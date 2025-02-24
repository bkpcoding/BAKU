#!/usr/bin/env python3

import warnings
import os
import wandb
from omegaconf import OmegaConf

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import cv2
import numpy as np
from collections import defaultdict

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightning_module import BCLightningModule

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory [{tag}] - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")


class LightningWorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        # We don't use the replay loader anymore since Lightning handles this
        # Just keep the dataset accessible for evaluation
        self.expert_dataset = dataset_iterable
        self.stats = dataset_iterable.stats

        # Initialize wandb if enabled
        if cfg.use_wandb:
            # Convert OmegaConf to dictionary for wandb, but don't resolve interpolations
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=cfg_dict,  # Log the entire config
            )

        # create logger for non-Lightning logging (e.g., evaluation)
        self.logger = Logger(
            self.work_dir,
            use_tb=self.cfg.use_tb,
            use_wandb=self.cfg.use_wandb
        )
        
        # create envs
        self.env = None
        self.task_descriptions = None
        if cfg.get('load_env', True):  # Default to True for backward compatibility
            self.cfg.suite.task_make_fn.max_episode_len = (
                dataset_iterable._max_episode_len
            )
            self.cfg.suite.task_make_fn.max_state_dim = (
                dataset_iterable._max_state_dim
            )
            if self.cfg.suite.name == "dmc":
                self.cfg.suite.task_make_fn.max_action_dim = (
                    dataset_iterable._max_action_dim
                )

            self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        if self.env is not None:
            self.agent = make_agent(
                self.env[0].observation_spec(), self.env[0].action_spec(), cfg
            )
        else:
            # You'll need to modify this based on your dataset's observation and action specs
            obs_spec = dataset_iterable.get_observation_spec()
            action_spec = dataset_iterable.get_action_spec()
            self.agent = make_agent(obs_spec, action_spec, cfg)
        print(f"*********** agent: {self.agent} ***********")
        try:
            self.envs_till_idx = dataset_iterable.envs_till_idx
        except AttributeError:
            self.envs_till_idx = 0
        # Discretizer for BeT
        if repr(self.agent) != "mtact":
            if repr(self.agent) == "rt1" or self.cfg.agent.policy_head in [
                "bet",
                "vqbet",
            ]:
                self.agent.discretize(
                    dataset_iterable.actions,
                    dataset_iterable.preprocess,
                )

        # Calculate global batch size based on number of GPUs
        cfg.batch_size = cfg.per_gpu_batch_size * cfg.num_gpus

        # Create Lightning module
        self.model = BCLightningModule(
            agent=self.agent,
            expert_dataset=dataset_iterable,
            batch_size=cfg.batch_size,  # Now using the calculated batch size
            stats=self.stats,
            cfg=cfg
        )

        # Setup Lightning loggers
        self.lightning_loggers = []
        if self.cfg.use_tb:
            tb_logger = TensorBoardLogger(
                save_dir=str(self.work_dir),
                name="tb_logs"
            )
            self.lightning_loggers.append(tb_logger)
        
        if self.cfg.use_wandb:
            wandb_logger = WandbLogger(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                save_dir=str(self.work_dir),
                log_model=True
            )
            self.lightning_loggers.append(wandb_logger)
        
        # Setup callbacks
        checkpoint_dir = str(self.work_dir / (self.cfg.checkpoint_dir if hasattr(self.cfg, "checkpoint_dir") else "checkpoints"))
        monitor_metric = self.cfg.monitor_metric if hasattr(self.cfg, "monitor_metric") else "val/actor_loss"
        monitor_mode = self.cfg.monitor_mode if hasattr(self.cfg, "monitor_mode") else "min"
        save_top_k = self.cfg.save_top_k if hasattr(self.cfg, "save_top_k") else 3
        save_last = self.cfg.save_last if hasattr(self.cfg, "save_last") else True
        
        self.callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{" + monitor_metric.replace("/", "_") + ":.4f}",
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=save_top_k,
                save_last=save_last,
                every_n_train_steps=self.cfg.suite.save_every_steps,
            ),
            LearningRateMonitor(logging_interval="step")
        ]
        
        # Create PyTorch Lightning Trainer
        # Note: this will handle DDP automatically when multiple GPUs are available
        trainer_kwargs = {
            "max_steps": self.cfg.suite.num_train_steps,
            "logger": self.lightning_loggers,
            "callbacks": self.callbacks,
            "default_root_dir": str(self.work_dir),
            "val_check_interval": self.cfg.val_check_interval if hasattr(self.cfg, "val_check_interval") else self.cfg.validate_every_steps,
            "log_every_n_steps": self.cfg.log_every_n_steps if hasattr(self.cfg, "log_every_n_steps") else self.cfg.suite.log_every_steps,
            "accelerator": "gpu",
            "strategy": "ddp" if int(cfg.num_gpus) > 1 else "auto",
            "devices": int(cfg.num_gpus),
            "precision": cfg.precision if hasattr(cfg, "precision") else 32,
            "deterministic": cfg.deterministic if hasattr(cfg, "deterministic") else False,
            "gradient_clip_val": cfg.gradient_clip_val if hasattr(cfg, "gradient_clip_val") else 0.0
        }
        
        self.trainer = pl.Trainer(**trainer_kwargs)
        
        # For evaluation
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self.model.global_step

    @property
    def global_episode(self):
        return self.model.global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        """Only run evaluation if environment is loaded"""
        if self.env is None:
            print("Skipping evaluation as no environment is loaded")
            return
            
        self.agent.train(False)
        episode_rewards = []
        successes = []

        num_envs = self.envs_till_idx

        for env_idx in range(num_envs):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_step}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - num_envs):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[:num_envs]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def train(self):
        """Train using PyTorch Lightning"""
        print(f"Starting training with {self.cfg.num_gpus} GPUs")
        
        # Start the Lightning training
        self.trainer.fit(self.model)
        
        # After training is complete, run evaluation if needed
        if self.cfg.eval and self.env is not None:
            self.eval()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=False)
        
        # Update the Lightning module's agent
        self.model.agent = self.agent

    def __del__(self):
        # Cleanup wandb
        if hasattr(self, 'cfg') and self.cfg.use_wandb:
            wandb.finish()

 
@hydra.main(config_path="cfgs", config_name="config_lightning")
def main(cfg):
    # Add multi-GPU configuration if not already present
    if not hasattr(cfg, "num_gpus"):
        cfg.num_gpus = torch.cuda.device_count()
    
    # Add precision setting if not already present (16 for mixed precision, 32 for full precision)
    if not hasattr(cfg, "precision"):
        cfg.precision = 32  # default to full precision

    # Print available GPUs
    print(f"Number of available GPUs: {cfg.num_gpus}")
    if int(cfg.num_gpus) > 0:
        for i in range(int(cfg.num_gpus)):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    root_dir = Path.cwd()
    workspace = LightningWorkspaceIL(cfg)

    # Load weights
    if cfg.load_bc:
        snapshots = {}
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)

    workspace.train()


if __name__ == "__main__":
    main()