import einops
import numpy as np
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T

import utils
from agent.networks.rgb_modules import BaseEncoder, ResnetEncoder
from agent.networks.policy_head import (
    DeterministicHead,
    GMMHead,
    BeTHead,
    VQBeTHead,
    DiffusionHead,
)
from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP
from agent.networks.kmeans_discretizer import KMeansDiscretizer
from opacus.grad_sample import GradSampleModule
from torchinfo import summary
from muon import Muon
import os
import torch.distributed as dist
class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_type="gpt",
        policy_head="deterministic",
        num_feat_per_step=1,
        device="cuda",
    ):
        super().__init__()

        self._policy_type = policy_type
        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        if policy_type == "gpt":
            self._policy = GPT(
                GPTConfig(
                    block_size=65,
                    input_dim=repr_dim,
                    output_dim=hidden_dim,
                    n_layer=8,
                    n_head=4,
                    n_embd=hidden_dim,
                    dropout=0.1,
                )
            )
        elif policy_type == "mlp":
            self._policy = nn.Sequential(
                nn.Linear(repr_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "gmm":
            self._action_head = GMMHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "bet":
            self._action_head = BeTHead(
                hidden_dim,
                self._act_dim,
                hidden_size=hidden_dim,
                num_layers=2,
            )
        elif policy_head == "vqbet":
            self._action_head = VQBeTHead(
                hidden_dim,
                self._act_dim,
                hidden_size=hidden_dim,
                device=device,
            )
            # self._action_head = GradSampleModule(self._action_head)
        elif policy_head == "diffusion":
            self._action_head = DiffusionHead(
                input_size=hidden_dim,
                output_size=self._act_dim,
                obs_horizon=10,  # 3 (dmc - diffusion)
                pred_horizon=10,  # 3 (dmc - diffusion)
                hidden_size=hidden_dim,
                num_layers=2,
                device=device,
            )

        self.apply(utils.weight_init)

    def forward(self, obs, num_prompt_feats, stddev, action=None, cluster_centers=None):
        B, T, D = obs.shape
        if self._policy_type == "mlp":
            if T * D < self._repr_dim:
                gt_num_time_steps = (
                    self._repr_dim // D - num_prompt_feats
                ) // self._num_feat_per_step
                num_repeat = (
                    gt_num_time_steps
                    - (T - num_prompt_feats) // self._num_feat_per_step
                )
                initial_obs = obs[
                    :, num_prompt_feats : num_prompt_feats + self._num_feat_per_step
                ]
                initial_obs = initial_obs.repeat(1, num_repeat, 1)
                obs = torch.cat(
                    [obs[:, :num_prompt_feats], initial_obs, obs[:, num_prompt_feats:]],
                    dim=1,
                )
                B, T, D = obs.shape
            obs = obs.view(B, 1, T * D)
            features = self._policy(obs)
        elif self._policy_type == "gpt":
            # insert action token at each self._num_feat_per_step interval
            prompt = obs[:, :num_prompt_feats]
            obs = obs[:, num_prompt_feats:]
            obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
            action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
            obs = torch.cat([prompt, obs], dim=1)

            # get action features
            features = self._policy(obs)
            features = features[:, num_prompt_feats:]
            num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
            features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        # action head
        pred_action = self._action_head(
            features,
            stddev,
            **{"cluster_centers": cluster_centers, "action_seq": action},
        )
        # shape of the input is ([96, 1, 256])
        for name, param in self._action_head.named_parameters():
            if hasattr(param, "grad_sample"):
                print(f"Param {name} has grad sample, yaya!! ")

        if action is None:
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
                **{"cluster_centers": cluster_centers},
            )
            return pred_action, loss[0] if isinstance(loss, tuple) else loss


def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory [{tag}] - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

class BCAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        stddev_clip,
        use_tb,
        augment,
        obs_type,
        encoder_type,
        policy_type,
        policy_head,
        pixel_keys,
        proprio_key,
        feature_key,
        use_proprio,
        train_encoder,
        norm,
        history,
        history_len,
        eval_history_len,
        separate_encoders,
        temporal_agg,
        max_episode_len,
        num_queries,
        prompt,
        use_language,
        film,
        use_muon,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.augment = augment
        self.obs_type = obs_type
        self.encoder_type = encoder_type
        self.policy_head = policy_head
        self.use_proprio = use_proprio if obs_type == "pixels" else False
        self.norm = norm
        self.train_encoder = train_encoder
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1
        self.separate_encoders = separate_encoders
        self.use_language = use_language
        self.language_proj_type = "mlp"  # mlp or identity
        self.prompt = prompt
        self.film = film
        self.use_muon = use_muon

        # Set environment variables for distributed processing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize distributed processing for Muon
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=1,
                rank=0
            )

        # language
        self.language_fusion = "none" if not self.use_language else "film"
        self.language_dim = 768
        self.lang_repr_dim = 512

        # actor parameters
        self._act_dim = action_shape[0]

        # keys
        if obs_type == "pixels":
            self.pixel_keys = pixel_keys
            self.proprio_key = proprio_key
        else:
            self.feature_key = feature_key

        # action chunking params
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.temporal_agg else 1

        # number of inputs per time step
        if obs_type == "features":
            num_feat_per_step = 1
        elif obs_type == "pixels":  # TODO: change to use_proprio
            num_feat_per_step = len(self.pixel_keys)
            if use_proprio:
                num_feat_per_step += 1

        # observation params
        if obs_type == "pixels":
            if use_proprio:
                proprio_shape = obs_shape[self.proprio_key]
            obs_shape = obs_shape[self.pixel_keys[0]]
        else:
            obs_shape = obs_shape[self.feature_key]

        # Track model size
        model_size = 0

        # encoder
        if obs_type == "pixels":
            if self.separate_encoders:
                self.encoder = {}
            if self.encoder_type == "base":
                if self.separate_encoders:
                    for key in self.pixel_keys:
                        self.encoder[key] = BaseEncoder(obs_shape).to(device)
                        self.repr_dim = self.encoder[key].repr_dim
                        model_size += sum(
                            p.numel()
                            for p in self.encoder[key].parameters()
                            if p.requires_grad
                        )
                else:
                    self.encoder = BaseEncoder(obs_shape).to(device)
                    self.repr_dim = self.encoder.repr_dim
                    model_size += sum(
                        p.numel() for p in self.encoder.parameters() if p.requires_grad
                    )
            elif self.encoder_type == "resnet":
                if self.separate_encoders:
                    for key in self.pixel_keys:
                        self.encoder[key] = ResnetEncoder(
                            obs_shape,
                            512,
                            language_dim=self.lang_repr_dim,
                            language_fusion=self.language_fusion,
                        ).to(device)
                        model_size += sum(
                            p.numel()
                            for p in self.encoder[key].parameters()
                            if p.requires_grad
                        )
                else:
                    self.encoder = ResnetEncoder(
                        obs_shape,
                        512,
                        language_dim=self.lang_repr_dim,
                        language_fusion=self.language_fusion,
                    ).to(device)
                    model_size += sum(
                        p.numel() for p in self.encoder.parameters() if p.requires_grad
                    )
                self.repr_dim = 512
            elif self.encoder_type == "patch":
                pass
        else:
            self.encoder = MLP(obs_shape[0], hidden_channels=[512, 512]).to(device)
            model_size += sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            self.repr_dim = 512

        # language encoder
        if self.use_language:
            # projector
            if self.language_proj_type == "mlp":
                self.language_projector = MLP(
                    self.language_dim,
                    hidden_channels=[self.lang_repr_dim, self.lang_repr_dim],
                ).to(device)
            else:
                self.language_projector = nn.Identity()
            self.language_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.language_projector.parameters()
                if p.requires_grad
            )

        # projector for proprioceptive features
        if use_proprio:
            self.proprio_projector = MLP(
                proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]
            ).to(device)
            self.proprio_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.proprio_projector.parameters()
                if p.requires_grad
            )

        if policy_type == "mlp":
            repr_mult_factor = len(self.pixel_keys) if obs_type == "pixels" else 1
            if use_proprio:
                repr_mult_factor += 1
            if history:
                repr_mult_factor *= self.history_len
            if self.use_language:
                repr_mult_factor += 1
        else:
            repr_mult_factor = 1

        # discretizer (for BeT)
        if self.policy_head == "bet":
            nbins = 64
            niters = 200
            self.discretizer = KMeansDiscretizer(num_bins=nbins, kmeans_iters=niters)
        else:
            self.discretizer = None

        # actor
        action_dim = (
            self._act_dim * self.num_queries if self.temporal_agg else self._act_dim
        )
        self.actor = Actor(
            self.repr_dim * repr_mult_factor,
            action_dim,
            hidden_dim,
            policy_type,
            self.policy_head,
            num_feat_per_step,
            device,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        # optimizers
        # encoder
        if self.train_encoder:
            if self.separate_encoders:
                params = []
                for key in self.pixel_keys:
                    params += list(self.encoder[key].parameters())
            else:
                params = list(self.encoder.parameters())
            self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # proprio
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # language
        if self.use_language:
            self.language_opt = torch.optim.AdamW(
                self.language_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # actor
        if self.use_muon:
            muon_params = [p for p in self.actor.parameters() if p.ndim >= 2]
            adamw_params = [p for p in self.actor.parameters() if p.ndim < 2]
            self.actor_opt = [
                Muon(muon_params, lr=lr, momentum=0.95),
                torch.optim.AdamW(adamw_params, lr=lr, weight_decay=1e-4)
            ]
        else:
            self.actor_opt = torch.optim.AdamW(
                self.actor.parameters(), lr=lr, weight_decay=1e-4
            )

        # augmentations
        if obs_type == "pixels" and self.norm:
            if self.encoder_type == "small":
                MEAN = torch.tensor([0.0, 0.0, 0.0])
                STD = torch.tensor([1.0, 1.0, 1.0])
            elif self.encoder_type == "resnet" or self.norm:
                MEAN = torch.tensor([0.485, 0.456, 0.406])
                STD = torch.tensor([0.229, 0.224, 0.225])
            self.customAug = T.Compose([T.Normalize(mean=MEAN, std=STD)])

        # data augmentation
        if obs_type == "pixels" and self.augment:
            self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "bc"

    def train(self, training=True):
        self.training = training
        if training:
            if self.separate_encoders:
                for key in self.pixel_keys:
                    if self.train_encoder:
                        self.encoder[key].train(training)
                    else:
                        self.encoder[key].eval()
            else:
                if self.train_encoder:
                    self.encoder.train(training)
                else:
                    self.encoder.eval()
            if self.use_language:
                self.language_projector.train(training)
            if self.obs_type == "pixels" and self.use_proprio:
                self.proprio_projector.train(training)
            self.actor.train(training)
        else:
            if self.separate_encoders:
                for key in self.pixel_keys:
                    self.encoder[key].eval()
            else:
                self.encoder.eval()
            if self.use_language:
                self.language_projector.eval()
            if self.obs_type == "pixels" and self.use_proprio:
                self.proprio_projector.eval()
            self.actor.eval()

    def buffer_reset(self):
        if self.obs_type == "pixels":
            self.observation_buffer = {}
            for key in self.pixel_keys:
                self.observation_buffer[key] = deque(maxlen=self.eval_history_len)
            if self.use_proprio:
                self.proprio_buffer = deque(maxlen=self.eval_history_len)
        else:
            self.observation_buffer = deque(maxlen=self.eval_history_len)

        # temporal aggregation
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [
                    self.max_episode_len,
                    self.max_episode_len + self.num_queries,
                    self._act_dim,
                ]
            ).to(self.device)

    def clear_buffers(self):
        del self.observation_buffer
        if self.obs_type == "pixels" and self.use_proprio:
            del self.proprio_buffer
        if self.temporal_agg:
            del self.all_time_actions

    def discretize(self, actions, preprocess):
        print("Discretizing actions ...")
        # organize actions into shape (N, A * num_queries)
        reshaped_actions = []
        for action in actions:
            action = preprocess["actions"](action)
            action = np.lib.stride_tricks.sliding_window_view(
                action,
                (self.num_queries, action.shape[-1]),
            )[:, 0]
            action = einops.rearrange(action, "n t a -> n (t a)")
            reshaped_actions.extend(action)
        reshaped_actions = np.array(reshaped_actions)

        if self.policy_head == "bet":
            actions = torch.as_tensor(reshaped_actions, device="cpu").float()
            self.discretizer.fit(actions)
            self._cluster_centers = self.discretizer.bin_centers.float().to(self.device)
        elif self.policy_head == "vqbet":
            config = {
                "epochs": 501, # default was 2001
                "batch_size": 2048,
                "save_every": 50,
            }
            self.actor._action_head.discretize(config, reshaped_actions)

        print("Discretization complete.")

    def reinit_optimizers(self):
        if self.train_encoder:
            if self.separate_encoders:
                params = []
                for key in self.pixel_keys:
                    params += list(self.encoder[key].parameters())
            else:
                params = list(self.encoder.parameters())
            self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        if self.use_language:
            self.language_opt = torch.optim.AdamW(
                self.language_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        params = list(self.actor.parameters())
        if self.use_muon:
            muon_params = [p for p in self.actor.parameters() if p.ndim >= 2]
            adamw_params = [p for p in self.actor.parameters() if p.ndim < 2]
            self.actor_opt = [
                Muon(muon_params, lr=self.lr, momentum=0.95),
                torch.optim.AdamW(adamw_params, lr=self.lr, weight_decay=1e-4)
            ]
        else:
            self.actor_opt = torch.optim.AdamW(
                self.actor.parameters(), lr=self.lr, weight_decay=1e-4
            )

    def act(self, obs, prompt, norm_stats, step, global_step, eval_mode=False):
        if norm_stats is not None:
            pre_process = lambda s_qpos: (
                s_qpos - norm_stats[self.proprio_key]["min"]
            ) / (
                norm_stats[self.proprio_key]["max"]
                - norm_stats[self.proprio_key]["min"]
                + 1e-5
            )
            post_process = (
                lambda a: a
                * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                + norm_stats["actions"]["min"]
            )

        # lang projection
        if self.use_language:
            key = self.pixel_keys[0] if self.obs_type == "pixels" else self.feature_key
            repeat_len = (
                min(len(self.observation_buffer[key]) + 1, self.eval_history_len)
                if self.obs_type == "pixels"
                else min(len(self.observation_buffer) + 1, self.eval_history_len)
            )
            lang_features = (
                torch.as_tensor(prompt["task_emb"], device=self.device)
                .float()[None].repeat(repeat_len, 1)
            )
            lang_features = self.language_projector(lang_features)
        else:
            lang_features = None

        if self.obs_type == "pixels":
            # add to buffer
            features = []
            for key in self.pixel_keys:
                self.observation_buffer[key].append(
                    self.test_aug(obs[key].transpose(1, 2, 0)).numpy()
                )
                pixels = torch.as_tensor(
                    np.array(self.observation_buffer[key]), device=self.device
                ).float()
                pixels = self.customAug(pixels / 255.0) if self.norm else pixels
                # encoder
                lang = lang_features if self.film else None
                pixels = (
                    self.encoder[key](pixels, lang=lang)
                    if self.separate_encoders
                    else self.encoder(pixels, lang=lang)
                )
                features.append(pixels)
            if self.use_proprio:
                obs[self.proprio_key] = pre_process(obs[self.proprio_key])
                self.proprio_buffer.append(obs[self.proprio_key])
                proprio = torch.as_tensor(
                    np.array(self.proprio_buffer), device=self.device
                ).float()
                proprio = self.proprio_projector(proprio)
                features.append(proprio)
            features = torch.cat(features, dim=-1).view(-1, self.repr_dim)
        else:
            self.observation_buffer.append(obs[self.feature_key])
            features = torch.as_tensor(
                np.array(self.observation_buffer), device=self.device
            ).float()
            features = self.encoder(features)

        # prompt
        prompt_features = []
        if self.use_language:
            prompt_features.append(lang_features[-1:])
        if self.prompt not in [None, "text", "one_hot"]:
            if self.use_language:
                prompt_lang_features = lang_features[-1:]
                reshape_lang = True
            else:
                prompt_lang_features = None

            if self.obs_type == "pixels":
                for key in self.pixel_keys:
                    pixel = torch.as_tensor(
                        prompt[f"prompt_{key}"], device=self.device
                    ).float()
                    shape = pixel.shape
                    # reshape lang features
                    if self.use_language and reshape_lang:
                        prompt_lang_features = prompt_lang_features.repeat(shape[0], 1)
                        reshape_lang = False
                    # augment
                    pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                    # encode
                    pixel = (
                        self.encoder[key](pixel, lang=prompt_lang_features)
                        if self.separate_encoders
                        else self.encoder(pixel, lang=prompt_lang_features)
                    )
                    prompt_features.append(pixel)
                if self.use_proprio:
                    proprio = torch.as_tensor(
                        prompt[f"prompt_{self.proprio_key}"], device=self.device
                    ).float()
                    proprio = self.proprio_projector(proprio)
                    prompt_features.append(proprio)
            else:
                prompt_feat = torch.as_tensor(
                    prompt[f"prompt_{self.feature_key}"], device=self.device
                ).float()
                prompt_feat = self.encoder(prompt_feat)
                prompt_features.append(prompt_feat)
        num_prompt_feats = len(prompt_features)
        if num_prompt_feats > 0:
            prompt_features = torch.cat(prompt_features, dim=-1).view(-1, self.repr_dim)
            features = torch.cat([prompt_features, features], dim=0)

        # Pass cluster center to actor for bet
        kwargs = {}
        if self.policy_head == "bet":
            kwargs["cluster_centers"] = self._cluster_centers

        stddev = utils.schedule(self.stddev_schedule, global_step)
        action = self.actor(features.unsqueeze(0), num_prompt_feats, stddev, **kwargs)

        if self.policy_head == "bet":
            _, offset, base_action = action
            action = base_action + offset
        elif self.policy_head == "vqbet":
            action = action["predicted_action"]
        elif self.policy_head == "diffusion":
            action = action[0]
        else:
            if eval_mode:
                action = action.mean
            else:
                action = action.sample()

        if self.temporal_agg:
            action = action.view(-1, self.num_queries, self._act_dim)
            self.all_time_actions[[step], step : step + self.num_queries] = action[-1:]
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0])
            return action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0, -1])
            return action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step, update=True):
        # print_gpu_memory("Start of update")
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)
        # print_gpu_memory("After data to device")
        
        action = data["actions"].float()
        
        # lang projection
        if self.use_language:
            lang_features = (
                data["task_emb"].float()[:, None].repeat(1, self.history_len, 1)
            )
            lang_features = self.language_projector(lang_features)
            lang_features = einops.rearrange(lang_features, "b t d -> (b t) d")
            # print_gpu_memory("After language projection")
        else:
            lang_features = None

        # features
        if self.obs_type == "pixels":
            features = []
            for key in self.pixel_keys:
                pixel = data[key].float()
                shape = pixel.shape
                pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
                pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                lang = lang_features if self.film else None
                if self.train_encoder:
                    # print_gpu_memory("Before encoder forward pass")
                    pixel = (
                        self.encoder[key](pixel, lang=lang)
                        if self.separate_encoders
                        else self.encoder(pixel, lang=lang)
                    )
                    # print_gpu_memory("After encoder forward pass")
                else:
                    with torch.no_grad():
                        pixel = (
                            self.encoder[key](pixel, lang=lang)
                            if self.separate_encoders
                            else self.encoder(pixel, lang=lang)
                        )
                pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
                features.append(pixel)
                # print_gpu_memory(f"After processing {key}")
            
            if self.use_proprio:
                proprio = data[self.proprio_key].float()
                proprio = self.proprio_projector(proprio)
                features.append(proprio)
            # concatenate
            features = torch.cat(features, dim=-1).view(
                action.shape[0], -1, self.repr_dim
            )  # (B, T * num_feat_per_step, D)
        else:
            features = data[self.feature_key].float()
            shape = features.shape
            if self.train_encoder:
                features = self.encoder(features)
            else:
                with torch.no_grad():
                    features = self.encoder(features)

        # prompt
        prompt_features = []
        if self.use_language:
            lang_features = einops.rearrange(
                lang_features, "(b t) d -> b t d", t=shape[1]
            )
            prompt_features.append(lang_features[:, -1:])
        if self.prompt not in [None, "text", "one_hot"]:
            if self.use_language:
                prompt_lang_features = lang_features[:, -1:]
                reshape_lang = True
            else:
                prompt_lang_features = None

            if self.obs_type == "pixels":
                for key in self.pixel_keys:
                    pixel = data[f"prompt_{key}"].float()
                    shape = pixel.shape
                    # reshape lang features
                    if self.use_language and reshape_lang:
                        prompt_lang_features = prompt_lang_features.repeat(
                            1, shape[1], 1
                        )
                        prompt_lang_features = einops.rearrange(
                            prompt_lang_features, "b t d -> (b t) d"
                        )
                        reshape_lang = False
                    # rearrange
                    pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
                    # augment
                    pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                    # encode
                    if self.train_encoder:
                        pixel = (
                            self.encoder[key](pixel, lang=prompt_lang_features)
                            if self.separate_encoders
                            else self.encoder(pixel, lang=prompt_lang_features)
                        )
                    else:
                        with torch.no_grad():
                            pixel = (
                                self.encoder[key](pixel, lang=prompt_lang_features)
                                if self.separate_encoders
                                else self.encoder(pixel, lang=prompt_lang_features)
                            )
                    pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
                    prompt_features.append(pixel)
                if self.use_proprio:
                    proprio = data[f"prompt_{self.proprio_key}"].float()
                    proprio = self.proprio_projector(proprio)
                    prompt_features.append(proprio)
            else:
                prompt_feat = data[f"prompt_{self.feature_key}"].float()
                if self.train_encoder:
                    prompt_feat = self.encoder(prompt_feat)
                else:
                    with torch.no_grad():
                        prompt_feat = self.encoder(prompt_feat)
                prompt_features.append(prompt_feat)
        num_prompt_feats = len(prompt_features) if len(prompt_features) > 0 else 0
        if num_prompt_feats > 0:
            prompt_features = torch.cat(prompt_features, dim=-1).view(
                action.shape[0], -1, self.repr_dim
            )
            # prepend prompt features
            features = torch.cat([prompt_features, features], dim=1)

        # rearrange action
        if self.temporal_agg:
            action = einops.rearrange(action, "b t1 t2 d -> b t1 (t2 d)")

        # Pass cluster center to actor for bet
        kwargs = {}
        if self.policy_head == "bet":
            kwargs["cluster_centers"] = self._cluster_centers

        # print_gpu_memory("Before actor forward pass")
        
        if update:
            # actor loss
            stddev = utils.schedule(self.stddev_schedule, step)
            _, actor_loss = self.actor(
                features, num_prompt_feats, stddev, action, **kwargs
            )
            # print_gpu_memory("After actor forward pass")

            if self.train_encoder:
                self.encoder_opt.zero_grad(set_to_none=True)
            if self.obs_type == "pixels" and self.use_proprio:
                self.proprio_opt.zero_grad(set_to_none=True)
            if self.use_language:
                self.language_opt.zero_grad(set_to_none=True)
            # actor_opt can be a list of optimizers when using Muon
            if isinstance(self.actor_opt, list):
                for opt in self.actor_opt:
                    opt.zero_grad(set_to_none=True)
            else:
                self.actor_opt.zero_grad(set_to_none=True)
            # print_gpu_memory("After zero_grad")

            actor_loss["actor_loss"].backward()
            # print_gpu_memory("After backward pass")

            if self.train_encoder:
                self.encoder_opt.step()
            if self.obs_type == "pixels" and self.use_proprio:
                self.proprio_opt.step()
            if self.use_language:
                self.language_opt.step()
            # actor_opt can be a list of optimizers when using Muon
            if isinstance(self.actor_opt, list):
                for opt in self.actor_opt:
                    opt.step()
            else:
                self.actor_opt.step()
            # print_gpu_memory("After optimizer steps")

            if self.policy_head == "diffusion" and step % 10 == 0:
                self.actor._action_head.net.ema_step()

            if self.use_tb:
                for key, value in actor_loss.items():
                    metrics[key] = value.item()

            # Clear some memory
            if step % 100 == 0:  # Every 100 steps
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # print_gpu_memory("After memory cleanup")

            return metrics

        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            pred_action, actor_loss = self.actor(
                features, num_prompt_feats, stddev, action, **kwargs
            )
            # print_gpu_memory("After actor forward pass (eval)")
            
            if self.use_tb:
                for key, value in actor_loss.items():
                    metrics[key] = value.item()
            metrics["gt_action"] = action.cpu().numpy()
            # predicted action
            if self.policy_head == "bet":
                _, offset, base_action = pred_action
                pred_action = base_action + offset
            elif self.policy_head == "vqbet":
                pred_action = pred_action["predicted_action"]
            elif self.policy_head == "diffusion":
                pred_action = pred_action[0]
            else:
                pred_action = pred_action.mean
            metrics["pred_action"] = pred_action.cpu().numpy()

            return metrics

    def save_snapshot(self):
        model_keys = ["actor", "encoder"]
        opt_keys = ["actor_opt"]
        if self.train_encoder:
            opt_keys += ["encoder_opt"]
        if self.obs_type == "pixels" and self.use_proprio:
            model_keys += ["proprio_projector"]
            opt_keys += ["proprio_opt"]
        if self.use_language:
            model_keys += ["language_projector"]
            opt_keys += ["language_opt"]
        # models
        payload = {
            k: self.__dict__[k].state_dict() for k in model_keys if k != "encoder"
        }
        if "encoder" in model_keys:
            if self.separate_encoders:
                for key in self.pixel_keys:
                    payload[f"encoder_{key}"] = self.encoder[key].state_dict()
            else:
                payload["encoder"] = self.encoder.state_dict()
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})
        if "actor_opt" in payload and isinstance(self.actor_opt, list):
            for i, opt in enumerate(self.actor_opt):
                payload[f"actor_opt_{i}"] = opt.state_dict()
            payload["use_muon_actor_opt"] = True
        elif "actor_opt" in payload:
            payload["actor_opt"] = self.actor_opt.state_dict()
            payload["use_muon_actor_opt"] = False
        del payload["actor_opt"] # remove the original actor_opt key as it might be ambiguous now
        # augmentations
        if self.obs_type == "pixels" and self.norm:
            if self.encoder_type == "small":
                MEAN = torch.tensor([0.0, 0.0, 0.0])
                STD = torch.tensor([1.0, 1.0, 1.0])
            elif self.encoder_type == "resnet" or self.norm:
                MEAN = torch.tensor([0.485, 0.456, 0.406])
                STD = torch.tensor([0.229, 0.224, 0.225])
            self.customAug = T.Compose([T.Normalize(mean=MEAN, std=STD)])

        # data augmentation
        if self.obs_type == "pixels" and self.augment:
            self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        if self.policy_head == "bet":
            payload["cluster_centers"] = self._cluster_centers
        elif self.policy_head == "vqbet":
            payload["vqvae"] = self.actor._action_head._vqvae_model.state_dict()

        others = [
            "use_proprio",
            "use_language",
            "max_episode_len",
        ]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, eval=False, load_opt=False):
        # models
        model_keys = ["actor", "encoder"]
        if self.obs_type == "pixels" and self.use_proprio:
            model_keys += ["proprio_projector"]
        if self.use_language:
            model_keys += ["language_projector"]
        for k in model_keys:
            if k == "encoder" and self.separate_encoders:
                for key in self.pixel_keys:
                    self.encoder[key].load_state_dict(payload[f"encoder_{key}"], strict=False)
            else:
                self.__dict__[k].load_state_dict(payload[k], strict=False)

        if self.policy_head == "bet":
            assert "cluster_centers" in payload
            "Cluster centers must be provided for BeT"
            self._cluster_centers = payload["cluster_centers"]
        elif self.policy_head == "vqbet":
            assert "vqvae" in payload
            "VQ-VAE model must be provided for VQ-BET"
            self.actor._action_head._vqvae_model.load_state_dict(payload["vqvae"])

        if eval:
            self.train(False)
            return

        # if not eval
        if not load_opt:
            self.reinit_optimizers()
        else:
            opt_keys = ["actor_opt"]
            if self.train_encoder:
                opt_keys += ["encoder_opt"]
            if self.obs_type == "pixels" and self.use_proprio:
                opt_keys += ["proprio_opt"]
            if self.use_language:
                opt_keys += ["language_opt"]
            if payload["use_muon_actor_opt"]:
                self.actor_opt = []
                i = 0
                while f"actor_opt_{i}" in payload:
                    muon_adamw_opt_state_dict = payload[f"actor_opt_{i}"]
                    if i == 0: # Muon optimizer
                        muon_params = [p for p in self.actor.parameters() if p.ndim >= 2]
                        opt = Muon(muon_params, lr=self.lr, momentum=0.95) # need to re-init optimizer and load state_dict
                    elif i == 1: # AdamW optimizer
                        adamw_params = [p for p in self.actor.parameters() if p.ndim < 2]
                        opt = torch.optim.AdamW(adamw_params, lr=self.lr, weight_decay=1e-4) # need to re-init optimizer and load state_dict
                    opt.load_state_dict(muon_adamw_opt_state_dict)
                    self.actor_opt.append(opt)
                    i += 1
            else:
                actor_opt_state_dict = payload["actor_opt"]
                self.actor_opt = torch.optim.AdamW(
                    self.actor.parameters(), lr=self.lr, weight_decay=1e-4
                ) # need to re-init optimizer and load state_dict
                self.actor_opt.load_state_dict(actor_opt_state_dict)
            del payload["actor_opt_0"]
            del payload["actor_opt_1"]
            del payload["use_muon_actor_opt"]
        self.train(True)

    def compute_loss(self, data, step):
        """Compute loss without performing optimization steps."""
        # Convert data to torch tensors
        data = utils.to_torch(data, self.device)
        action = data["actions"].float()

        # lang projection
        if self.use_language:
            lang_features = (
                data["task_emb"].float()[:, None].repeat(1, self.history_len, 1)
            )
            lang_features = self.language_projector(lang_features)
            lang_features = einops.rearrange(lang_features, "b t d -> (b t) d")
        else:
            lang_features = None

        # features
        if self.obs_type == "pixels":
            features = []
            for key in self.pixel_keys:
                pixel = data[key].float()
                shape = pixel.shape
                # rearrange
                pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
                # augment
                pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                # encode
                lang = lang_features if self.film else None
                with torch.no_grad():  # Don't compute gradients during validation
                    pixel = (
                        self.encoder[key](pixel, lang=lang)
                        if self.separate_encoders
                        else self.encoder(pixel, lang=lang)
                    )
                pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
                features.append(pixel)
            if self.use_proprio:
                proprio = data[self.proprio_key].float()
                with torch.no_grad():
                    proprio = self.proprio_projector(proprio)
                features.append(proprio)
            # concatenate
            features = torch.cat(features, dim=-1).view(
                action.shape[0], -1, self.repr_dim
            )
        else:
            features = data[self.feature_key].float()
            shape = features.shape
            with torch.no_grad():
                features = self.encoder(features)

        # prompt
        prompt_features = []
        if self.use_language:
            lang_features = einops.rearrange(
                lang_features, "(b t) d -> b t d", t=shape[1]
            )
            prompt_features.append(lang_features[:, -1:])
        if self.prompt not in [None, "text", "one_hot"]:
            if self.use_language:
                prompt_lang_features = lang_features[:, -1:]
                reshape_lang = True
            else:
                prompt_lang_features = None

            if self.obs_type == "pixels":
                for key in self.pixel_keys:
                    pixel = data[f"prompt_{key}"].float()
                    shape = pixel.shape
                    if self.use_language and reshape_lang:
                        prompt_lang_features = prompt_lang_features.repeat(
                            1, shape[1], 1
                        )
                        prompt_lang_features = einops.rearrange(
                            prompt_lang_features, "b t d -> (b t) d"
                        )
                        reshape_lang = False
                    pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
                    pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                    with torch.no_grad():
                        pixel = (
                            self.encoder[key](pixel, lang=prompt_lang_features)
                            if self.separate_encoders
                            else self.encoder(pixel, lang=prompt_lang_features)
                        )
                    pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
                    prompt_features.append(pixel)
                if self.use_proprio:
                    proprio = data[f"prompt_{self.proprio_key}"].float()
                    with torch.no_grad():
                        proprio = self.proprio_projector(proprio)
                    prompt_features.append(proprio)
            else:
                prompt_feat = data[f"prompt_{self.feature_key}"].float()
                with torch.no_grad():
                    prompt_feat = self.encoder(prompt_feat)
                prompt_features.append(prompt_feat)

        num_prompt_feats = len(prompt_features) if len(prompt_features) > 0 else 0
        if num_prompt_feats > 0:
            prompt_features = torch.cat(prompt_features, dim=-1).view(
                action.shape[0], -1, self.repr_dim
            )
            features = torch.cat([prompt_features, features], dim=1)

        if self.temporal_agg:
            action = einops.rearrange(action, "b t1 t2 d -> b t1 (t2 d)")

        # Pass cluster center to actor for bet
        kwargs = {}
        if self.policy_head == "bet":
            kwargs["cluster_centers"] = self._cluster_centers

        # Compute actor loss without gradients
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            _, actor_loss = self.actor(
                features, num_prompt_feats, stddev, action, **kwargs
            )

        return actor_loss
