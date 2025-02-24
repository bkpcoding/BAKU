import random
import numpy as np
import torch


def _worker_init_fn(worker_id):
    # Use a smaller base seed value
    base_seed = worker_id + torch.initial_seed() % (2**31)
    np.random.seed(base_seed)
    print(f"Setting seed {base_seed} for worker {worker_id}")
    random.seed(base_seed)


def make_expert_replay_loader(iterable, batch_size):
    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
