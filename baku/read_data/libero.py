import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from typing import List, Dict

class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        scenes,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        intermediate_goal_step=50,
        store_actions=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step

        # temporal_aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # Convert task_names, which is a list, to a dictionary
        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}

        # Get relevant task names
        task_name = []
        for scene in scenes:
            task_name.extend([task_name for task_name in tasks[scene]])

        # get data paths
        self._paths = []
        # for suite in suites:
        self._paths.extend(list((Path(path) / suite).glob("*")))

        if task_name is not None:
            paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split(".")[0].split("/")[-1]
                if task in task_name:
                    # get idx of task in task_name
                    idx = task_name.index(task)
                    paths[idx] = path
                    idx2name[idx] = task
            del self._paths
            self._paths = paths

        # store actions
        if store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._min_episode_len = float("inf")
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = (
                data["observations"] if self._obs_type == "pixels" else data["states"]
            )
            actions = data["actions"]
            task_emb = data["task_emb"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i]["pixels"])
                    ),
                )
                self._min_episode_len = min(
                    self._min_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i]["pixels"])
                    ),
                )
                # if obs_type == 'features':
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1]
                )
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i]["pixels"])
                )

                # store actions
                if store_actions:
                    self.actions.append(actions[i])

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
            "proprioceptive": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        self._global_indices = {}  # Maps (env_idx, episode_idx, sample_idx) to global index
        current_idx = 0
        
        # Assign fixed indices during data loading
        for env_idx in self._episodes:
            for episode_idx in range(len(self._episodes[env_idx])):
                episode = self._episodes[env_idx][episode_idx]
                obs_length = (len(episode['observation']) if self._obs_type == 'features' 
                            else len(episode['observation']['pixels']))
                for sample_idx in range(obs_length - self._history_len):
                    self._global_indices[(env_idx, episode_idx, sample_idx)] = current_idx
                    current_idx += 1

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx
        episode_idx = random.randrange(len(self._episodes[idx]))
        episode = self._episodes[idx][episode_idx]
        return (episode, idx, episode_idx) if env_idx is None else episode

    def _sample(self):
        env_idx = random.randint(0, self.envs_till_idx - 1)
        episode_idx = random.randrange(len(self._episodes[env_idx]))
        episodes = self._episodes[env_idx][episode_idx]
        observations = episodes["observation"]
        actions = episodes["action"]
        task_emb = episodes["task_emb"]

        if self._obs_type == "pixels":
            # Sample obs, action
            sample_idx = np.random.randint(
                0, len(observations["pixels"]) - self._history_len
            )
            global_idx = self._global_indices[(env_idx, episode_idx, sample_idx)]
            sampled_pixel = observations["pixels"][
                sample_idx : sample_idx + self._history_len
            ]
            sampled_pixel_egocentric = observations["pixels_egocentric"][
                sample_idx : sample_idx + self._history_len
            ]
            sampled_pixel = torch.stack(
                [self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))]
            )
            sampled_pixel_egocentric = torch.stack(
                [
                    self.aug(sampled_pixel_egocentric[i])
                    for i in range(len(sampled_pixel_egocentric))
                ]
            )
            sampled_proprioceptive_state = np.concatenate(
                [
                    observations["joint_states"][
                        sample_idx : sample_idx + self._history_len
                    ],
                    observations["gripper_states"][
                        sample_idx : sample_idx + self._history_len
                    ],
                ],
                axis=-1,
            )
            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                sampled_action = np.zeros(
                    (self._history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = (
                    self._history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, actions.shape[-1]))
                act[
                    : min(len(actions), sample_idx + num_actions) - sample_idx
                ] = actions[sample_idx : sample_idx + num_actions]
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx : sample_idx + self._history_len]

            # prompt
            if self._prompt == "text":
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](
                        sampled_proprioceptive_state
                    ),
                    "actions": self.preprocess["actions"](sampled_action),
                    "task_emb": task_emb,
                    "global_idx": global_idx,
                }
            elif self._prompt == "goal":
                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode["observation"]
                prompt_pixel = self.aug(prompt_observations["pixels"][-1])[None]
                prompt_pixel_egocentric = self.aug(
                    prompt_observations["pixels_egocentric"][-1]
                )[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["joint_states"][-1:],
                        prompt_observations["gripper_states"][-1:],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_episode["action"][-1:]
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](
                        sampled_proprioceptive_state
                    ),
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](
                        prompt_proprioceptive_state
                    ),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                    "global_idx": global_idx,
                }
            elif self._prompt == "intermediate_goal":
                prompt_episode = episodes
                prompt_observations = prompt_episode["observation"]
                intermediate_goal_step = (
                    self.intermediate_goal_step + np.random.randint(-30, 30)
                )
                goal_idx = min(
                    sample_idx + intermediate_goal_step,
                    len(prompt_observations["pixels"]) - 1,
                )
                prompt_pixel = self.aug(prompt_observations["pixels"][goal_idx])[None]
                prompt_pixel_egocentric = self.aug(
                    prompt_observations["pixels_egocentric"][goal_idx]
                )[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["joint_states"][goal_idx : goal_idx + 1],
                        prompt_observations["gripper_states"][goal_idx : goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](
                        sampled_proprioceptive_state
                    ),
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](
                        prompt_proprioceptive_state
                    ),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                    "global_idx": global_idx,
                }

        elif self._obs_type == "features":
            # Sample obs, action
            sample_idx = np.random.randint(0, len(observations) - self._history_len)
            global_idx = self._global_indices[(env_idx, episode_idx, sample_idx)]
            sampled_obs = np.array(
                observations[sample_idx : sample_idx + self._history_len]
            )
            sampled_action = actions[sample_idx : sample_idx + self._history_len]
            # pad obs to match self._max_state_dim
            obs = np.zeros((self._history_len, self._max_state_dim))
            state_dim = sampled_obs.shape[-1]
            obs[:, :state_dim] = sampled_obs
            sampled_obs = obs

            # prompt obs, action
            if self._prompt == "text":
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "task_emb": task_emb,
                    "global_idx": global_idx,

                }
            elif self._prompt == "goal":
                prompt_episode = self._sample_episode(env_idx)
                prompt_obs = np.array(prompt_episode["observation"][-1:])
                prompt_action = prompt_episode["action"][-1:]
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                    "global_idx": global_idx
                }
            elif self._prompt == "intermediate_goal":
                prompt_episode = self._sample_episode(env_idx)
                goal_idx = min(
                    sample_idx + self.intermediate_goal_step,
                    len(prompt_episode["observation"]) - 1,
                )
                prompt_obs = np.array(
                    prompt_episode["observation"][goal_idx : goal_idx + 1]
                )
                prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                    "global_idx": global_idx,
                }

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]
        actions = episode["action"]
        task_emb = episode["task_emb"]

        if self._obs_type == "pixels":
            pixels_shape = observations["pixels"].shape

            # observation
            if self._prompt == None or self._prompt == "text":
                prompt_pixel = None
                prompt_pixel_egocentric = None
                prompt_proprioceptive_state = None
                prompt_action = None
            elif self._prompt == "goal":
                prompt_pixel = np.transpose(observations["pixels"][-1:], (0, 3, 1, 2))
                prompt_pixel_egocentric = np.transpose(
                    observations["pixels_egocentric"][-1:], (0, 3, 1, 2)
                )
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["joint_states"][-1:],
                        observations["gripper_states"][-1:],
                    ],
                    axis=-1,
                )
                prompt_action = None
            elif self._prompt == "intermediate_goal":
                goal_idx = min(
                    step + self.intermediate_goal_step, len(observations["pixels"]) - 1
                )
                prompt_pixel = np.transpose(
                    observations["pixels"][goal_idx : goal_idx + 1], (0, 3, 1, 2)
                )
                prompt_pixel_egocentric = np.transpose(
                    observations["pixels_egocentric"][goal_idx : goal_idx + 1],
                    (0, 3, 1, 2),
                )
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["joint_states"][goal_idx : goal_idx + 1],
                        observations["gripper_states"][goal_idx : goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = None

            return {
                "prompt_pixels": prompt_pixel,
                "prompt_pixels_egocentric": prompt_pixel_egocentric,
                "prompt_proprioceptive": (
                    self.preprocess["proprioceptive"](prompt_proprioceptive_state)
                    if prompt_proprioceptive_state is not None
                    else None
                ),
                "prompt_actions": (
                    self.preprocess["actions"](prompt_action)
                    if prompt_action is not None
                    else None
                ),
                "task_emb": task_emb,
            }

        elif self._obs_type == "features":
            # observation
            if self._prompt == None or self._prompt == "text":
                prompt_obs, prompt_action = None, None
            elif self._prompt == "goal":
                prompt_obs = np.array(observations[-1:])
                prompt_action = None

            return {
                "prompt_features": prompt_obs,
                "prompt_actions": self.preprocess["actions"](prompt_action),
                "task_emb": task_emb,
            }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples

class BCDatasetSplit(BCDataset):
    """Extension of BCDataset that handles training/validation splitting"""
    def __init__(self, parent_dataset: BCDataset, indices: List[int], is_train: bool = True):
        """
        Create a split from a parent BCDataset.
        
        Args:
            parent_dataset: The original BCDataset
            indices: Indices for this split
            is_train: Whether this is the training split
        """
        # Copy parent dataset attributes without reinitializing
        self.__dict__.update(parent_dataset.__dict__)
        
        # Store split-specific information
        self.original_indices = indices
        self.is_train = is_train
        
        # Create episode mapping for this split with remapped indices
        self._split_episodes = {}
        for new_idx, original_idx in enumerate(self.original_indices):
            self._split_episodes[new_idx] = self._episodes[original_idx]
        
        # Override episodes with split episodes
        self._episodes = self._split_episodes
        
        # Recompute split-specific attributes
        self._num_samples = sum(
            len(episode['observation']) if self._obs_type == 'features' 
            else len(episode['observation']['pixels'])
            for episodes in self._episodes.values()
            for episode in episodes
        )
        
        # Update envs_till_idx for this split
        self.envs_till_idx = len(self._episodes)
        
        # For train split only, extract actions for discretization
        if is_train:
            self.actions = []
            for episodes in self._episodes.values():
                for episode in episodes:
                    self.actions.append(episode['action'])

    def _sample_episode(self, env_idx=None):
        """
        Override sample_episode to handle remapped indices
        """
        if env_idx is None:
            idx = random.randint(0, self.envs_till_idx - 1)
            episode = random.choice(self._episodes[idx])
            return (episode, idx)
        else:
            # For explicit env_idx requests, we need to handle remapping
            if env_idx >= self.envs_till_idx:
                raise IndexError(f"env_idx {env_idx} out of range for split with {self.envs_till_idx} environments")
            episode = random.choice(self._episodes[env_idx])
            return episode

def create_train_val_splits(dataset: BCDataset, val_ratio: float = 0.1, seed: int = None):
    """
    Create training and validation splits from a BCDataset.
    
    Args:
        dataset: Original BCDataset
        val_ratio: Proportion of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset (tuple of BCDatasetSplit)
    """
    if seed is not None:
        random.seed(seed)
    
    # Get all environment indices
    all_indices = list(dataset._episodes.keys())
    
    # Calculate split sizes
    total_envs = len(all_indices)
    val_size = int(total_envs * val_ratio)
    train_size = total_envs - val_size
    
    # Randomly shuffle indices
    random.shuffle(all_indices)
    
    # Split indices
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    # Create split datasets
    train_dataset = BCDatasetSplit(dataset, train_indices, is_train=True)
    val_dataset = BCDatasetSplit(dataset, val_indices, is_train=False)
    
    return train_dataset, val_dataset