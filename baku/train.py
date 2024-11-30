#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import cv2
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

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

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        self.dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        test_dataset_iterable = hydra.utils.call(self.cfg.test_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            self.dataset_iterable, self.cfg.batch_size
        )
        self.test_replayer_loader = make_expert_replay_loader(
            test_dataset_iterable, 1 # keeping the batch size 1 since we are evaluating on only one test point for now
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.test_replayer_iter = iter(self.test_replayer_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
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
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx

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

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)

        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            if (
                self.cfg.eval
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # update
            metrics, shapley_values = self.agent.update(self.expert_replay_iter, self.global_step, calculate_shapley=True, val_replay_iter=self.test_replayer_iter)
            if self.global_step % 100 == 0:
                positve_samples, negative_samples = self.analyze_shapley_values(shapley_values, self.dataset_iterable)
                self.print_shapley_analysis(positve_samples, negative_samples)
                self.analyze_environment_influence(shapley_values, self.dataset_iterable)
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

            self._global_step += 1


    def analyze_shapley_values(self, shapley_values, dataset, top_k=10):
        """
        Analyzes Shapley values to find most influential training examples.
        
        Args:
            shapley_values (dict): Dictionary mapping global indices to their Shapley values
            dataset (BCDataset): The training dataset instance
            top_k (int): Number of top positive and negative examples to return
        
        Returns:
            tuple: (top_positive_samples, top_negative_samples) where each is a list of 
            (env_idx, episode_idx, sample_idx, shapley_value) tuples
        """
        # Convert dictionary to list of (index, value) pairs
        shapley_items = [(idx, np.mean(values) if isinstance(values, list) else values) 
                        for idx, values in shapley_values.items()]
        
        # Sort by Shapley value
        sorted_items = sorted(shapley_items, key=lambda x: x[1])
        
        # Get top negative and positive influences
        top_negative = sorted_items[:top_k]
        top_positive = sorted_items[-top_k:][::-1]  # Reverse to get highest first
        
        def map_global_to_local(global_idx):
            # Find the corresponding (env_idx, episode_idx, sample_idx)
            for (env_idx, episode_idx, sample_idx), idx in dataset._global_indices.items():
                if idx == global_idx:
                    return (env_idx, episode_idx, sample_idx)
            return None
        
        # Map global indices to local indices
        negative_samples = []
        for global_idx, shapley_value in top_negative:
            local_indices = map_global_to_local(global_idx)
            if local_indices:
                negative_samples.append((*local_indices, shapley_value))
        
        positive_samples = []
        for global_idx, shapley_value in top_positive:
            local_indices = map_global_to_local(global_idx)
            if local_indices:
                positive_samples.append((*local_indices, shapley_value))
                
        return positive_samples, negative_samples


    def analyze_environment_influence(self, shapley_values, dataset):
        """
        Analyzes total and average Shapley values per environment.
        
        Args:
            shapley_values (dict): Dictionary mapping global indices to their Shapley values
            dataset (BCDataset): The training dataset instance
        
        Returns:
            tuple: Dictionaries containing environment-wise total and average influences
        """
        # Initialize dictionaries to store environment-wise statistics
        env_total_influence = {}  # Sum of Shapley values per environment
        env_count = {}           # Count of samples per environment
        
        # Map each global index to its environment and accumulate values
        for global_idx, values in shapley_values.items():
            # Find corresponding environment index
            for (env_idx, episode_idx, sample_idx), idx in dataset._global_indices.items():
                if idx == global_idx:
                    shapley_value = np.mean(values) if isinstance(values, list) else values
                    
                    if env_idx not in env_total_influence:
                        env_total_influence[env_idx] = 0
                        env_count[env_idx] = 0
                        
                    env_total_influence[env_idx] += shapley_value
                    env_count[env_idx] += 1
                    break
        
        # Calculate average influence per environment
        env_avg_influence = {
            env_idx: total / env_count[env_idx]
            for env_idx, total in env_total_influence.items()
        }
        
        # Find environments with max and min influence
        max_env = max(env_avg_influence.items(), key=lambda x: x[1])
        min_env = min(env_avg_influence.items(), key=lambda x: x[1])
        
        print("\nEnvironment-wise Influence Analysis:")
        print("===================================")
        print(f"\nMost Influential Environment:")
        print(f"Environment {max_env[0]}: Average Shapley Value = {max_env[1]:.6f}")
        print(f"Total Influence = {env_total_influence[max_env[0]]:.6f}")
        print(f"Number of samples = {env_count[max_env[0]]}")
        
        print(f"\nLeast Influential Environment:")
        print(f"Environment {min_env[0]}: Average Shapley Value = {min_env[1]:.6f}")
        print(f"Total Influence = {env_total_influence[min_env[0]]:.6f}")
        print(f"Number of samples = {env_count[min_env[0]]}")
        
        print("\nAll Environments Summary:")
        print("------------------------")
        for env_idx in sorted(env_avg_influence.keys()):
            print(f"Environment {env_idx:2d}: "
                f"Avg = {env_avg_influence[env_idx]:10.6f}, "
                f"Total = {env_total_influence[env_idx]:10.6f}, "
                f"Samples = {env_count[env_idx]:4d}")
        
        return env_total_influence, env_avg_influence

    
    def print_shapley_analysis(self, positive_samples, negative_samples):
        """Prints the analysis results in a formatted way"""
        print("\nTop Positive Influential Samples:")
        print("=================================")
        print(f"{'Env':>5} {'Episode':>8} {'Sample':>8} {'Shapley Value':>15}")
        print("-" * 40)
        for env_idx, episode_idx, sample_idx, value in positive_samples:
            print(f"{env_idx:5d} {episode_idx:8d} {sample_idx:8d} {value:15.6f}")
        
        print("\nTop Negative Influential Samples:")
        print("=================================")
        print(f"{'Env':>5} {'Episode':>8} {'Sample':>8} {'Shapley Value':>15}")
        print("-" * 40)
        for env_idx, episode_idx, sample_idx, value in negative_samples:
            print(f"{env_idx:5d} {episode_idx:8d} {sample_idx:8d} {value:15.6f}")

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