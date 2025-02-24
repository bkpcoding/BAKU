import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# def to_torch(xs, device):
#     return tuple(torch.as_tensor(x, device=device) for x in xs)
def to_torch(xs, device):
    for key, value in xs.items():
        try:
            xs[key] = torch.as_tensor(value, device=device)
        except ValueError:
            continue

    return xs


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()
        # Keep track of evaluation time so that total time only includes train time
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        return time.time() - self._start_time - self._eval_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """

    # num_channels: num_groups
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(
                        GROUP_NORM_LOOKUP[num_channels], num_channels
                    )
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


import torch
import torch.nn as nn

class ActionAutoencoder(nn.Module):
   def __init__(self, action_dim, latent_dim, seq_length, hidden_dim=256):
       super().__init__()
       self.seq_length = seq_length
       self.latent_dim = latent_dim

       # Encoder
       self.encoder = nn.Sequential(
           nn.Linear(action_dim * seq_length, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU()
       )
       
       # Latent projections for VAE
       self.fc_mu = nn.Linear(hidden_dim, latent_dim)
       self.fc_var = nn.Linear(hidden_dim, latent_dim)

       # Decoder
       self.decoder = nn.Sequential(
           nn.Linear(latent_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, action_dim * seq_length)
       )

   def encode(self, x):
       # x shape: [batch_size, seq_length, action_dim]
       batch_size = x.shape[0]
       x = x.reshape(batch_size, -1)  # Flatten sequence
       h = self.encoder(x)
       return self.fc_mu(h), self.fc_var(h)

   def reparameterize(self, mu, logvar):
       std = torch.exp(0.5 * logvar)
       eps = torch.randn_like(std)
       return mu + eps * std

   def decode(self, z):
       # z shape: [batch_size, latent_dim]
       output = self.decoder(z)
       return output.reshape(-1, self.seq_length, output.shape[-1]//self.seq_length)

   def forward(self, x):
       mu, logvar = self.encode(x)
       z = self.reparameterize(mu, logvar)
       return self.decode(z), mu, logvar
   
   def loss_function(self, recon_x, x, mu, logvar, kl_weight=0.01):
       MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
       KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
       return MSE + kl_weight * KLD



import torch
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def load_pkl_data(data_dir):
   all_data = []
   labels = []
   skill_names = []
   
   for i, file in enumerate(sorted(os.listdir(data_dir))):
       if file.endswith('.pkl'):
           with open(os.path.join(data_dir, file), 'rb') as f:
               data = pickle.load(f)
               all_data.append(data)
               labels.extend([i] * len(data))
               skill_names.append(file.split('.')[0])
           
   return np.concatenate(all_data), np.array(labels), skill_names

# Setup training
def train_model(model, data, n_epochs=100, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        # Shuffle data
        idx = np.random.permutation(len(data))
        total_loss = 0
        
        for i in range(0, len(data), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch = torch.FloatTensor(data[batch_idx])
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(data)}")

# Visualize latent space
def visualize_latent(model, data, labels, skill_names):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.FloatTensor(data))
        latent = mu.numpy()
    
    # Use t-SNE for visualization
    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(latent)
    
    # Plot
    plt.figure(figsize=(10, 10))
    for i, skill in enumerate(skill_names):
        mask = labels == i
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], label=skill)
    
    plt.legend()
    plt.title("Action Latent Space")
    plt.show()

def main():
    # Load and preprocess data
    data_dir = "/home/shared_data/data_attrib_data/pkl_libero/libero_90/skill_datasets/"
    data, labels, skill_names = load_pkl_data(data_dir)

    # Create and train model
    seq_length = data.shape[1]  # Get sequence length from data
    action_dim = data.shape[2]  # Get action dimension from data
    latent_dim = 32

    model = ActionAutoencoder(action_dim, latent_dim, seq_length)
    train_model(model, data)

    # Visualize results
    visualize_latent(model, data, labels, skill_names)

if __name__=="__main__":
    main()
    
class TransformerActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim, seq_length, num_layers=3, nhead=8):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # Action embedding
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        
        # Position encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length + 1, latent_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=4*latent_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent projections
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

class TransformerActionDecoder(nn.Module):
    def __init__(self, action_dim, latent_dim, seq_length, num_layers=6, nhead=8):
        super().__init__()
        self.seq_length = seq_length
        
        # Position embeddings for decoder
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, latent_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=4*latent_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(latent_dim, action_dim)

class ActionTransformerAE(nn.Module):
    def __init__(self, action_dim, latent_dim, seq_length):
        super().__init__()
        self.encoder = TransformerActionEncoder(action_dim, latent_dim, seq_length)
        self.decoder = TransformerActionDecoder(action_dim, latent_dim, seq_length)
        
    def encode(self, x):
        # x shape: [batch_size, seq_length, action_dim]
        batch_size = x.shape[0]
        
        # Embed actions
        x = self.encoder.action_embedding(x)
        
        # Add CLS token
        cls_tokens = self.encoder.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.encoder.pos_embedding
        
        # Transform
        encoded = self.encoder.transformer(x)
        
        # Get CLS token output
        cls_output = encoded[:, 0]
        
        # Project to latent distribution
        mu = self.encoder.fc_mu(cls_output)
        logvar = self.encoder.fc_var(cls_output)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        batch_size = z.shape[0]
        
        # Expand latent to sequence length
        query = self.decoder.pos_embedding.expand(batch_size, -1, -1)
        
        # Expand z for decoder
        memory = z.unsqueeze(1).expand(-1, self.decoder.seq_length, -1)
        
        # Decode
        decoded = self.decoder.transformer(query, memory)
        actions = self.decoder.output_projection(decoded)
        
        return actions
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kl_weight=0.01):
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + kl_weight * KLD