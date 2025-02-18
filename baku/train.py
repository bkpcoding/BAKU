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
import torchinfo

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


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # Initialize wandb if enabled
        if cfg.use_wandb:
            # Convert OmegaConf to dictionary for wandb, but don't resolve interpolations
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=cfg_dict,  # Log the entire config
            )

        # create logger
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
                self.expert_replay_loader.dataset._max_episode_len
            )
            self.cfg.suite.task_make_fn.max_state_dim = (
                self.expert_replay_loader.dataset._max_state_dim
            )
            if self.cfg.suite.name == "dmc":
                self.cfg.suite.task_make_fn.max_action_dim = (
                    self.expert_replay_loader.dataset._max_action_dim
                )

            self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        if self.env is not None:
            self.agent = make_agent(
                self.env[0].observation_spec(), self.env[0].action_spec(), cfg
            )
        else:
            # You'll need to modify this based on your dataset's observation and action specs
            obs_spec = self.expert_replay_loader.dataset.get_observation_spec()
            action_spec = self.expert_replay_loader.dataset.get_action_spec()
            self.agent = make_agent(obs_spec, action_spec, cfg)

        try:
            self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx
        except AttributeError:
            self.envs_till_idx = 0
        # Discretizer for BeT
        if repr(self.agent) != "mtact":
            if repr(self.agent) == "rt1" or self.cfg.agent.policy_head in [
                "bet",
                "vqbet",
            ]:
                self.agent.discretize(
                    self.expert_replay_loader.dataset.actions,
                    self.expert_replay_loader.dataset.preprocess,
                )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

        # Add validation logging frequency and batch size
        self.validate_every_steps = cfg.suite.get('validate_every_steps', 1000)
        self.validation_batch_size = cfg.suite.get('validation_batch_size', 32)
        
        # Initialize best validation losses per skill
        if hasattr(self.expert_replay_loader.dataset, 'get_skill_names'):
            self.best_val_losses = {
                skill: float('inf') 
                for skill in self.expert_replay_loader.dataset.get_skill_names()
            }
        else:
            # For datasets without explicit skills, use a single overall validation loss
            self.best_val_losses = {'overall': float('inf')}

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

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
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
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

    def validate(self):
        """Compute validation loss, handling both skill-based and regular datasets."""
        self.agent.train(False)
        
        if hasattr(self.expert_replay_loader.dataset, 'get_validation_batch_per_skill'):
            # Use skill-specific validation for datasets that support it
            val_batches_per_skill = self.expert_replay_loader.dataset.get_validation_batch_per_skill(
                batch_size_per_skill=self.validation_batch_size
            )
            
            skill_losses = {}
            for skill_name, val_batches in val_batches_per_skill.items():
                total_skill_loss = 0
                num_valid_samples = 0
                
                # Collect all samples for this skill into a single batch
                batch_data = defaultdict(list)
                for val_batch in val_batches:
                    for k, v in val_batch.items():
                        batch_data[k].append(v)
                
                # Convert lists to appropriate format
                processed_batch = {}
                for k, v in batch_data.items():
                    if isinstance(v[0], (np.ndarray, torch.Tensor)):
                        processed_batch[k] = np.stack(v, axis=0)
                    else:
                        processed_batch[k] = v
                
                try:
                    metrics = self.agent.compute_loss(processed_batch, self.global_step)
                    total_skill_loss += metrics['actor_loss']
                    num_valid_samples += 1
                except Exception as e:
                    print(f"Warning: Error processing validation batch for skill {skill_name}: {str(e)}")
                    continue
                
                if num_valid_samples > 0:
                    avg_skill_loss = total_skill_loss / num_valid_samples
                    skill_losses[skill_name] = avg_skill_loss
                    
                    if avg_skill_loss < self.best_val_losses[skill_name]:
                        self.best_val_losses[skill_name] = avg_skill_loss
            
            if skill_losses:
                overall_val_loss = sum(skill_losses.values()) / len(skill_losses)
            else:
                print("Warning: No valid skill losses computed during validation")
                overall_val_loss = float('inf')
            
        else:
            # For datasets without skill-specific validation, compute overall validation loss
            total_val_loss = 0
            num_valid_batches = 0
            
            for _ in range(5):  # Try 5 validation batches
                try:
                    if hasattr(self.expert_replay_loader.dataset, 'get_validation_batch'):
                        val_batch = self.expert_replay_loader.dataset.get_validation_batch(
                            self.validation_batch_size
                        )
                    else:
                        val_batch = next(self.expert_replay_iter)
                    
                    metrics = self.agent.compute_loss(val_batch, self.global_step)
                    total_val_loss += metrics['actor_loss']
                    num_valid_batches += 1
                except Exception as e:
                    print(f"Warning: Error during validation: {str(e)}")
                    continue
            
            if num_valid_batches > 0:
                overall_val_loss = total_val_loss / num_valid_batches
                skill_losses = {'overall': overall_val_loss}
                
                if overall_val_loss < self.best_val_losses['overall']:
                    self.best_val_losses['overall'] = overall_val_loss
            else:
                print("Warning: No valid batches processed during validation")
                overall_val_loss = float('inf')
                skill_losses = {'overall': overall_val_loss}
        
        # Log validation metrics
        with self.logger.log_and_dump_ctx(self.global_frame, ty="validation") as log:
            log("val_actor_loss_overall", overall_val_loss)
            for skill_name, loss in skill_losses.items():
                log(f"val_actor_loss_{skill_name}", loss)
                log(f"val_actor_loss_{skill_name}_best", self.best_val_losses[skill_name])
            log("step", self.global_step)
            
        if self.cfg.use_wandb:
            wandb_log_dict = {
                "validation/actor_loss_overall": overall_val_loss,
                "step": self.global_step
            }
            for skill_name, loss in skill_losses.items():
                wandb_log_dict.update({
                    f"validation/actor_loss_{skill_name}": loss,
                    f"validation/actor_loss_{skill_name}_best": self.best_val_losses[skill_name]
                })
            wandb.log(wandb_log_dict)
            
        self.agent.train(True)
        return skill_losses, overall_val_loss

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)
        validate_every_step = utils.Every(self.validate_every_steps, 1)

        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate - only if env is loaded and eval is enabled
            if (
                self.cfg.eval
                and self.env is not None
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # update
            metrics = self.agent.update(self.expert_replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # log
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("actor_loss", metrics["actor_loss"])
                    log("step", self.global_step)

            # save snapshot
            if save_every_step(self.global_step):
                self.save_snapshot()

            # Add validation check with per-skill reporting
            if validate_every_step(self.global_step):
                skill_losses, overall_loss = self.validate()
                print(f"\nStep {self.global_step} Validation Losses:")
                print(f"Overall: {overall_loss:.4f}")
                for skill_name, loss in skill_losses.items():
                    print(f"{skill_name}: {loss:.4f} (Best: {self.best_val_losses[skill_name]:.4f})")
                print()  # Empty line for readability

            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode", "stats"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()
        
        if self.cfg.use_wandb:
            wandb.save(str(snapshot))  # Save model checkpoint to wandb

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

    def __del__(self):
        # Cleanup wandb
        if self.cfg.use_wandb:
            wandb.finish()

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

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
