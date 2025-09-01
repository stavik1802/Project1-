import numpy as np
import torch
import random
import os

def init_seeds(seed, envs=None):
    """Sets random seed for reproducibility."""
    seed = int(seed)
    
    if envs is not None:
        # Generate a sequence of seeds for each environment using NumPy's SeedSequence
        envs_seeds = np.random.SeedSequence(seed).generate_state(len(envs))
        for idx, env in enumerate(envs):
            env.seed(int(envs_seeds[idx]))
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    
    # If you are using CUDA, ensure deterministic behavior by setting seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set Python's built-in 'random' seed
    random.seed(seed)
    
    # Ensure reproducible hashing
    os.environ['PYTHONHASHSEED'] = str(seed)
