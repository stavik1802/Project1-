import numpy as np
import os
import gym
from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict, Box as GymBox
import gymnasium
from collections import OrderedDict
from gymnasium import spaces
from gymnasium.spaces import Box, Dict
from dm_env import specs, StepType, TimeStep
# from MPAC.envs.wrappers.dmc_wrapper import DMCWrapper
import energy_net.envs as energy_net_envs
from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

class EnergyNetWrapper(gym.Env):
    """
         Wrapper for the EnergyNetV0 environment to work with MPAC training loop.
         Only processes PCS actions and observations but returns full env-compatible outputs.
    """
    def __init__(self, env_setup_kwargs):
        def convert_obs_space(space):
            if isinstance(space, gymnasium.spaces.Box):
                return GymBox(
                    low=space.low,
                    high=space.high,
                    shape=space.shape,
                    dtype=space.dtype
                )
            return space

        env_setup_kwargs.setdefault('pricing_policy', PricingPolicy.ONLINE)
        env_setup_kwargs.setdefault('cost_type', CostType.CONSTANT)
        env_setup_kwargs.setdefault('demand_pattern', DemandPattern.CONSTANT)

        project_root = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(project_root, "../../../"))
        env_setup_kwargs.setdefault('env_config_path', os.path.join(project_root, "energy-net/configs/environment_config.yaml"))
        env_setup_kwargs.setdefault('iso_config_path', os.path.join(project_root, "energy-net/configs/iso_config.yaml"))
        env_setup_kwargs.setdefault('pcs_unit_config_path', os.path.join(project_root, "energy-net/configs/pcs_unit_config.yaml"))
        env_setup_kwargs.setdefault('log_file', os.path.join(project_root, "energy-net/logs/environments.log"))
        safety_spec = create_safety_spec(env_setup_kwargs)
        self.env = EnergyNetV0(**env_setup_kwargs, safety_spec=safety_spec)
        self.r_dim = None
        pcs_box = self.env.action_space["pcs"]
        if isinstance(pcs_box, gymnasium.spaces.Box):
            self.action_space = GymBox(
                low=pcs_box.low,
                high=pcs_box.high,
                shape=pcs_box.shape,
                dtype=pcs_box.dtype
            )
        else:
            self.action_space = pcs_box  # fallback, maybe already a gym Box
        # self.action_space = self.env.action_space["pcs"]
        self.observation_space = convert_obs_space(self.env.observation_space["pcs"])


    def reset(self):
        return self.env.reset()[0]["pcs"]

    def step(self,action):
                #full_action = {key:(action[key] if key == "pcs" else np.zeros(space.shape,dtype=space.dtype)) for key,space in self.env.action_space.items()}
        full_action = {"iso": np.zeros(3,dtype=np.float32),
                       "pcs": action}
        #print("fixed step")
       # print(self.env.step(full_action))
        obs_raw, reward, terminated, truncated, info = self.env.step(full_action)
        done = terminated["pcs"]
        obs = obs_raw["pcs"]
        reward_scalar = reward["pcs"] if isinstance(reward, dict) else reward
        #print(reward["pcs"] if isinstance(reward, dict) else reward)
        #reward_vector = np.array([reward_scalar], dtype=np.float32)
        info['reward'] = reward_scalar
        constraints = info.get('constraints', np.array([True]))
        cost = 1.0 - np.all(constraints)
        info['cost'] = cost


        return obs, reward_scalar, done, info

    def render(self):
        return self.env.reder()

    def close(self):
        return self.env.close()
# class DummyTask:
#     def __init__(self):
#         self._random = np.random.RandomState()
#
# class EnergyNetWrapper(DMCWrapper):
#     """
#     Wrapper for the EnergyNetV0 environment to work with MPAC training loop.
#     Only processes PCS actions and observations but returns full env-compatible outputs.
#     """
#
#     def _flatten_obs(self, obs):
#         def _flatten(v):
#             if isinstance(v, dict):
#                 return [x for sub in v.values() for x in _flatten(sub)]
#             elif isinstance(v, (list, tuple, np.ndarray)):
#                 return np.ravel(v).tolist()
#             else:
#                 return [float(v)]
#
#         flat = _flatten(obs)
#
#         # Keep full structure, but zero ISO part (first 3 dims)
#         flat = np.asarray(flat, dtype=np.float32)
#         flat[:3] = 0.0
#         return flat
#
#     def __init__(self, env_setup_kwargs):
#         env_setup_kwargs.setdefault('pricing_policy', PricingPolicy.CONSTANT)
#         env_setup_kwargs.setdefault('cost_type', CostType.CONSTANT)
#         env_setup_kwargs.setdefault('demand_pattern', DemandPattern.CONSTANT)
#
#         project_root = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.abspath(os.path.join(project_root, "../../../"))
#         env_setup_kwargs.setdefault('env_config_path', os.path.join(project_root, "energy-net/configs/environment_config.yaml"))
#         env_setup_kwargs.setdefault('iso_config_path', os.path.join(project_root, "energy-net/configs/iso_config.yaml"))
#         env_setup_kwargs.setdefault('pcs_unit_config_path', os.path.join(project_root, "energy-net/configs/pcs_unit_config.yaml"))
#         env_setup_kwargs.setdefault('log_file', os.path.join(project_root, "energy-net/logs/environments.log"))
#
#         env = EnergyNetV0(**env_setup_kwargs)
#         print(env)
#
#         def observation_spec():
#             obs = env.observation_space
#             if isinstance(obs, spaces.Dict):
#                 return OrderedDict({
#                     k: specs.BoundedArray(
#                         shape=v.shape,
#                         dtype=v.dtype,
#                         minimum=np.clip(v.low, -1e4, 1e4),
#                         maximum=np.clip(v.high, -1e4, 1e4),
#                         name=k
#                     )
#                     for k, v in obs.spaces.items()
#                 })
#             else:
#                 return specs.BoundedArray(
#                     shape=obs.shape,
#                     dtype=obs.dtype,
#                     minimum=np.clip(obs.low, -1e4, 1e4),
#                     maximum=np.clip(obs.high, -1e4, 1e4),
#                     name="obs"
#                 )
#
#         def action_spec():
#             action = env.action_space
#             if isinstance(action, Dict):
#                 lows = []
#                 highs = []
#                 shapes = []
#                 dtypes = []
#                 for space in action.spaces.values():
#                     assert isinstance(space, Box), "Only Box subspaces supported"
#                     lows.append(np.ravel(space.low))
#                     highs.append(np.ravel(space.high))
#                     shapes.append(space.shape)
#                     dtypes.append(space.dtype)
#
#                 low = np.concatenate(lows)
#                 high = np.concatenate(highs)
#                 dtype = np.result_type(*dtypes)
#
#                 return specs.BoundedArray(
#                     shape=low.shape,
#                     dtype=dtype,
#                     minimum=low,
#                     maximum=high,
#                     name="action"
#                 )
#             else:
#                 return specs.BoundedArray(
#                     shape=action.shape,
#                     dtype=action.dtype,
#                     minimum=np.clip(action.low, -1e4, 1e4),
#                     maximum=np.clip(action.high, -1e4, 1e4),
#                     name="action"
#                 )
#
#         env.observation_spec = observation_spec
#         env.action_spec = action_spec
#         env.task = DummyTask()
#
#         exclude_keys = []
#         self._setup(env, exclude_keys)
#         self.r_dim = None
#
#     def step(self, action):
#         iso_dim = 3
#
#         # If action is a single scalar or 1D array with one element
#         pcs_action = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
#
#         action_dict = {
#             "iso": np.zeros(iso_dim, dtype=np.float32),
#             "pcs": pcs_action  # Send scalar
#         }
#
#         obs_raw, reward, terminated, truncated, info = self.env._env.step(action_dict)
#         done = terminated["iso"] or terminated["pcs"]
#
#         obs = self._flatten_obs(obs_raw)
#
#         constraints = info.get('constraints', None)
#         if constraints is not None:
#             info['cost'] = np.float32(1.0 - int(np.all(constraints)))
#
#         reward_scalar = reward["pcs"] if isinstance(reward, dict) else reward
#         reward_vector = np.array([reward_scalar], dtype=np.float32)
#         info['reward'] = reward_vector
#
#         return obs, reward_vector, done, info
#
#     def seed(self, seed):
#         self.env.task._random = np.random.RandomState(seed)
#
#     def reset(self):
#         obs_raw, info = self.env.reset()
#         obs = self._flatten_obs(obs_raw)
#         return TimeStep(
#             step_type=StepType.FIRST,
#             reward=None,
#             discount=1.0,
#             observation=obs
#         )
#
#
#

# class EnergyNetWrapper(DMCWrapper):
#     """
#     Wrapper for the EnergyNetV0 environment to work with MPAC training loop.
#     """
#
#     def _flatten_obs(self, obs):
#         def _flatten(v):
#             if isinstance(v, dict):
#                 return [x for sub in v.values() for x in _flatten(sub)]
#             elif isinstance(v, (list, tuple, np.ndarray)):
#                 return np.ravel(v).tolist()
#             else:
#                 return [float(v)]
#
#         flat = _flatten(obs)
#         return np.asarray(flat, dtype=np.float32)
#     def __init__(self, env_setup_kwargs):
#         # Instantiate EnergyNetV0 using the config kwargs
#         env_setup_kwargs.setdefault('pricing_policy', PricingPolicy.CONSTANT)
#         env_setup_kwargs.setdefault('cost_type', CostType.CONSTANT)
#         env_setup_kwargs.setdefault('demand_pattern', DemandPattern.CONSTANT)
#         project_root = os.path.dirname(os.path.abspath(__file__))  # path to MPAC/envs/wrappers
#         project_root = os.path.abspath(os.path.join(project_root, "../../../"))  # now at project root
#
#         env_setup_kwargs.setdefault('env_config_path',
#                                     os.path.join(project_root, "energy-net/configs/environment_config.yaml"))
#         env_setup_kwargs.setdefault('iso_config_path', os.path.join(project_root, "energy-net/configs/iso_config.yaml"))
#         env_setup_kwargs.setdefault('pcs_unit_config_path',
#                                     os.path.join(project_root, "energy-net/configs/pcs_unit_config.yaml"))
#         env_setup_kwargs.setdefault('log_file', os.path.join(project_root, "energy-net/logs/environments.log"))
#         env = EnergyNetV0(**env_setup_kwargs)
#         #convert observation and action spec to dmc
#         def observation_spec():
#             obs = env.observation_space
#             if isinstance(obs, spaces.Dict):
#                 return OrderedDict({
#                     k: specs.BoundedArray(
#                         shape=v.shape,
#                         dtype=v.dtype,
#                         minimum=np.clip(v.low, -1e4, 1e4),
#                         maximum=np.clip(v.high, -1e4, 1e4),
#                         name=k
#                     )
#                     for k, v in obs.spaces.items()
#                 })
#             else:
#                 return specs.BoundedArray(
#                     shape=obs.shape,
#                     dtype=obs.dtype,
#                     minimum=np.clip(obs.low, -1e4, 1e4),
#                     maximum=np.clip(obs.high, -1e4, 1e4),
#                     name="obs"
#                 )
#
#         def action_spec():
#             if isinstance(env.action_space, Dict):
#                 # Flatten and concatenate bounds
#                 lows = []
#                 highs = []
#                 shapes = []
#                 dtypes = []
#
#                 for space in env.action_space.spaces.values():
#                     assert isinstance(space, Box), "Only Box subspaces supported"
#                     lows.append(np.ravel(space.low))
#                     highs.append(np.ravel(space.high))
#                     shapes.append(space.shape)
#                     dtypes.append(space.dtype)
#
#                 low = np.concatenate(lows)
#                 high = np.concatenate(highs)
#                 dtype = np.result_type(*dtypes)
#
#                 return specs.BoundedArray(
#                     shape=low.shape,
#                     dtype=dtype,
#                     minimum=low,
#                     maximum=high,
#                     name="action"
#                 )
#             else:
#                 space = env.action_space
#                 return specs.BoundedArray(
#                     shape=space.shape,
#                     dtype=space.dtype,
#                     minimum=space.low,
#                     maximum=space.high,
#                     name="action"
#                 )
#
#         env.observation_spec = observation_spec
#         env.action_spec = action_spec
#         env.task = DummyTask()
#         # You can exclude specific observation keys if needed
#         exclude_keys = []  # or e.g., ['constraints'] if your controller provides them
#         self._setup(env, exclude_keys)
#
#         # Setup using base DMCWrapper logic
#         self._setup(env, exclude_keys)
#
#         # Define reward dimension if using multi-objective (optional)
#         self.r_dim = 1  # For example, one for each agent — adjust if needed
#
#     def step(self, action):
#         iso_dim = 3  # change this to your ISO action dimension
#         iso_action = action[:iso_dim]
#         pcs_action = action[iso_dim:]
#
#         action_dict = {
#             "iso": iso_action,
#             "pcs": pcs_action
#         }
#
#         obs_raw, reward, terminated, truncated, info = self.env._env.step(action_dict)
#         done = terminated["iso"] or terminated["pcs"]
#
#         obs = self._flatten_obs(obs_raw)
#
#         constraints = info.get('constraints', None)
#         if constraints is not None:
#             info['cost'] = 1.0 - int(np.all(constraints))
#         reward_scalar = sum(reward.values())
#         info['reward'] = reward_scalar
#
#         return obs, reward, done, info
#
#     def seed(self, seed):
#         self.env.task._random = np.random.RandomState(seed)
#
#     def reset(self):
#         obs_raw, info = self.env.reset()
#         obs = self._flatten_obs(obs_raw)
#         return TimeStep(
#             step_type=StepType.FIRST,
#             reward=None,
#             discount=1.0,
#             observation=obs
#          )
#


from collections import OrderedDict

def create_safety_spec(env_setup_kwargs):
    """Creates safety_spec dictionary."""
    safety_spec = {
        'enable':       True,
        'observations': True,
        'safety_coeff': env_setup_kwargs['safety_coeff'],
    }

    constraints_list = env_setup_kwargs['energy_constraints']

    if env_setup_kwargs.get('constraints_all', False):
        constraints_list = list(energy_net.keys())

    if constraints_list:
        constraints = OrderedDict()
        for constraint in constraints_list:
            constraints[constraint] = energy_net[constraint]
        safety_spec['constraints'] = constraints

    return safety_spec


#### Constrain dicts ####

energy_net = {
    'battery':       energy_net_envs.energy_net_v0.battery_level_constraint,
}

