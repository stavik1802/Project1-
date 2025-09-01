"""
PCS Environment Factory for RL-Baselines3-Zoo integration.

This module provides factory functions that create PCS-focused environments
wrapped appropriately for training with RL-Baselines3-Zoo.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from energy_net.env import EnergyNetV0
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.td3.policies import TD3Policy
from energy_net.env.iso_env import make_iso_env_zoo
from MPAC.config_env_pcs import get_env_kwargs,get_env_setup_kwargs

class NormalizedPolicy:
    def __init__(self, policy, vec_norm):
        self.policy = policy
        self.vec_norm = vec_norm

    def predict(self, obs, **kwargs):
        obs = self.vec_norm.normalize_obs(obs)
        action = self.policy.predict(obs, **kwargs)
        return action

def make_pcs_env_zoo(
    norm_path=None,
    iso_policy_path=None,
    log_dir="logs",
    use_dispatch_action=True,
    dispatch_strategy="PROPORTIONAL",
    monitor=True,
    seed=None,
    **kwargs
):
    """
    Factory function for PCS environment compatible with RL-Baselines3-Zoo.
    
    Args:
        norm_path: Path to saved normalization statistics
        iso_policy_path: Path to a trained ISO policy to use during PCS training
        log_dir: Directory for saving logs
        use_dispatch_action: Whether to include dispatch in ISO action space
        dispatch_strategy: Strategy for dispatch when not controlled by agent
        monitor: Whether to wrap with Monitor for episode stats
        seed: Random seed
        **kwargs: Additional arguments to pass to EnergyNetV0
        
    Returns:
        A PCS-focused environment ready for training with RL-Baselines3-Zoo
    """
    # Create monitor directory if it doesn't exist
    if monitor:
        monitor_dir = os.path.join(log_dir, "pcs_monitor")
        os.makedirs(monitor_dir, exist_ok=True)
    
    # Create base environment
    env_kwargs = {
        "dispatch_config": {
            "use_dispatch_action": use_dispatch_action,
            "default_strategy": dispatch_strategy
        }
    }

    env_kwargs.update(kwargs)

    env = EnergyNetV0(**env_kwargs)
    
    # # Load ISO policy if provided
    # iso_policy = None
    # if iso_policy_path:
    #     try:
    #         print(f"Loading ISO policy from {iso_policy_path}")
    #         iso_policy = PPO.load(iso_policy_path,device='cpu')
    #     except Exception as e:
    #         print(f"Error loading ISO policy: {e}")
    # Load ISO policy if provided
    iso_policy = None
    if iso_policy_path:
        
        print(f"Loading ISO policy from {iso_policy_path}")
        env_kwargs = get_env_setup_kwargs()
        dummy_env = DummyVecEnv([lambda: make_iso_env_zoo(**env_kwargs)])
            
        try:
            vec_norm = VecNormalize.load(norm_path, venv=dummy_env)
            vec_norm.training = False
            vec_norm.norm_reward = False

            print("✅ VecNormalize loaded and applied.")
        except Exception as e:
            print(f"⚠️ Skipping normalization: {e}")
            vec_norm = dummy_env  # fallback

        # 3. Load TD3 model
        try:
            iso_model = TD3.load(iso_policy_path, device="cpu")
            print("✅ TD3 model loaded.")
        except Exception as e:
            raise RuntimeError(f"❌ Could not load TD3 model: {e}")
        iso_policy = NormalizedPolicy(iso_model.policy, vec_norm)
    
    # Import here to avoid circular imports
    from energy_net.alternating_wrappers import PCSEnvWrapper
    
    # Apply PCS wrapper
    env = PCSEnvWrapper(env, iso_policy=iso_policy)
    
    # Apply monitor wrapper if requested
    if monitor:
        env = Monitor(env, monitor_dir, allow_early_resets=True)
    
    # Set random seed if provided
    if seed is not None:
        env.seed(seed)
    
    return env 