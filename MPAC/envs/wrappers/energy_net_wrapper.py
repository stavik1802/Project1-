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
from energy_net.env.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing_policy import PricingPolicy
from energy_net.market.iso.cost_types import CostType
from energy_net.market.iso.demand_patterns import DemandPattern

class EnergyNetWrapper(gym.Env):
    def __init__(self, gymnasium_env):
        def convert_obs_space(space):
            if isinstance(space, gymnasium.spaces.Box):
                return GymBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
            return space
        self.env = gymnasium_env
        self.r_dim = None

        # Use correct space
        space = self.env.action_space
        self.action_space = GymBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        self.observation_space = convert_obs_space(self.env.observation_space)

    def reset(self):
        obs_all = self.env.reset()[0]
        return obs_all

    def step(self, action):
        # Decide partner action
        # if self.agent_type == "pcs":
        #     iso_action = self.partner_policy(self.env.observation_space["iso"]) if self.partner_policy else np.zeros(3, dtype=np.float32)
        #     full_action = {"iso": iso_action, "pcs": action}
        # else:
        #     pcs_action = self.partner_policy(self.env.observation_space["pcs"]) if self.partner_policy else 0.0
        #     full_action = {"iso": action, "pcs": pcs_action}

        obs_raw, reward, terminated, truncated, info = self.env.step(action)
        done = terminated
        obs = obs_raw
        reward_scalar = reward if isinstance(reward, dict) else reward
        info['reward'] = reward_scalar
        return obs, reward_scalar, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

# class EnergyNetWrapper(gym.Env):
#     def __init__(self, env_setup_kwargs, agent_type: str = "pcs", partner_policy=None):
#         def convert_obs_space(space):
#             if isinstance(space, gymnasium.spaces.Box):
#                 return GymBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
#             return space
#
#         self.agent_type = agent_type.lower()
#         self.partner_policy = partner_policy
#         # Default config setup
#         env_setup_kwargs.setdefault('pricing_policy', PricingPolicy.ONLINE)
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
#         self.env = EnergyNetV0(**env_setup_kwargs)
#         self.r_dim = None
#
#         # Use correct space
#         space = self.env.action_space[self.agent_type]
#         self.action_space = GymBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
#         self.observation_space = convert_obs_space(self.env.observation_space[self.agent_type])
#
#     def reset(self):
#         obs_all = self.env.reset()[0]
#         return obs_all[self.agent_type]
#
#     def step(self, action):
#         # Decide partner action
#         if self.agent_type == "pcs":
#             iso_action = self.partner_policy(self.env.observation_space["iso"]) if self.partner_policy else np.zeros(3, dtype=np.float32)
#             full_action = {"iso": iso_action, "pcs": action}
#         else:
#             pcs_action = self.partner_policy(self.env.observation_space["pcs"]) if self.partner_policy else 0.0
#             full_action = {"iso": action, "pcs": pcs_action}
#
#         obs_raw, reward, terminated, truncated, info = self.env.step(full_action)
#         done = terminated[self.agent_type]
#         obs = obs_raw[self.agent_type]
#         reward_scalar = reward[self.agent_type] if isinstance(reward, dict) else reward
#         info['reward'] = reward_scalar
#         return obs, reward_scalar, done, info
#
#     def render(self):
#         return self.env.render()
#
#     def close(self):
#         return self.env.close()

####### verse that switches between pcs and iso with hardcoded price when iso
# class EnergyNetWrapper(gym.Env):
#     """
#          Wrapper for the EnergyNetV0 environment to work with MPAC training loop.
#          Only processes PCS actions and observations but returns full env-compatible outputs.
#     """
#     def __init__(self, env_setup_kwargs,agent_type: str = "pcs"):
#         def convert_obs_space(space):
#             if isinstance(space, gymnasium.spaces.Box):
#                 return GymBox(
#                     low=space.low,
#                     high=space.high,
#                     shape=space.shape,
#                     dtype=space.dtype
#                 )
#             return space
#
#         print("@@@@@@@@@@@@@@2 this is train in train_agent")
#         print(agent_type)
#         self.agent_type = agent_type.lower()  # either "iso" or "pcs"
#         env_setup_kwargs.setdefault('pricing_policy', PricingPolicy.ONLINE)
#         env_setup_kwargs.setdefault('cost_type', CostType.CONSTANT)
#         env_setup_kwargs.setdefault('demand_pattern', DemandPattern.CONSTANT)
#
#         project_root = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.abspath(os.path.join(project_root, "../../../"))
#         env_setup_kwargs.setdefault('env_config_path', os.path.join(project_root, "energy-net/configs/environment_config.yaml"))
#         env_setup_kwargs.setdefault('iso_config_path', os.path.join(project_root, "energy-net/configs/iso_config.yaml"))
#         env_setup_kwargs.setdefault('pcs_unit_config_path', os.path.join(project_root, "energy-net/configs/pcs_unit_config.yaml"))
#         env_setup_kwargs.setdefault('log_file', os.path.join(project_root, "energy-net/logs/environments.log"))
#         self.env = EnergyNetV0(**env_setup_kwargs)
#         self.r_dim = None
#         pcs_box = self.env.action_space[self.agent_type]
#         if isinstance(pcs_box, gymnasium.spaces.Box):
#             self.action_space = GymBox(
#                 low=pcs_box.low,
#                 high=pcs_box.high,
#                 shape=pcs_box.shape,
#                 dtype=pcs_box.dtype
#             )
#         else:
#             self.action_space = pcs_box  # fallback, maybe already a gym Box
#         # self.action_space = self.env.action_space["pcs"]
#         self.observation_space = convert_obs_space(self.env.observation_space[self.agent_type])
#
#
#     def reset(self):
#         return self.env.reset()[0][self.agent_type]
#
#     def step(self,action):
#         print("this is agent type####################")
#         print(self.agent_type)
#         if self.agent_type == "pcs":
#             full_action = {"iso": np.zeros(3,dtype=np.float32),
#                             "pcs": action}
#         else:
#             full_action = {"iso": action,
#                            #"pcs": np.zeros(4, dtype=np.float32)
#                             "pcs": 0.0}
#         #print("fixed step")
#        # print(self.env.step(full_action))
#         obs_raw, reward, terminated, truncated, info = self.env.step(full_action)
#         done = terminated[self.agent_type]
#         obs = obs_raw[self.agent_type]
#         reward_scalar = reward[self.agent_type] if isinstance(reward, dict) else reward
#         #print(reward["pcs"] if isinstance(reward, dict) else reward)
#         #reward_vector = np.array([reward_scalar], dtype=np.float32)
#         info['reward'] = reward_scalar
#         return obs, reward_scalar, done, info
#
#     def render(self):
#         return self.env.reder()
#
#     def close(self):
#         return self.env.close()
######## verse of only pcs that works #################3
# class EnergyNetWrapper(gym.Env):
#     """
#          Wrapper for the EnergyNetV0 environment to work with MPAC training loop.
#          Only processes PCS actions and observations but returns full env-compatible outputs.
#     """
#     def __init__(self, env_setup_kwargs):
#         def convert_obs_space(space):
#             if isinstance(space, gymnasium.spaces.Box):
#                 return GymBox(
#                     low=space.low,
#                     high=space.high,
#                     shape=space.shape,
#                     dtype=space.dtype
#                 )
#             return space
#
#         env_setup_kwargs.setdefault('pricing_policy', PricingPolicy.ONLINE)
#         env_setup_kwargs.setdefault('cost_type', CostType.CONSTANT)
#         env_setup_kwargs.setdefault('demand_pattern', DemandPattern.CONSTANT)
#
#         project_root = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.abspath(os.path.join(project_root, "../../../"))
#         env_setup_kwargs.setdefault('env_config_path', os.path.join(project_root, "energy-net/configs/environment_config.yaml"))
#         env_setup_kwargs.setdefault('iso_config_path', os.path.join(project_root, "energy-net/configs/iso_config.yaml"))
#         env_setup_kwargs.setdefault('pcs_unit_config_path', os.path.join(project_root, "energy-net/configs/pcs_unit_config.yaml"))
#         env_setup_kwargs.setdefault('log_file', os.path.join(project_root, "energy-net/logs/environments.log"))
#         self.env = EnergyNetV0(**env_setup_kwargs)
#         self.r_dim = None
#         pcs_box = self.env.action_space["pcs"]
#         if isinstance(pcs_box, gymnasium.spaces.Box):
#             self.action_space = GymBox(
#                 low=pcs_box.low,
#                 high=pcs_box.high,
#                 shape=pcs_box.shape,
#                 dtype=pcs_box.dtype
#             )
#         else:
#             self.action_space = pcs_box  # fallback, maybe already a gym Box
#         # self.action_space = self.env.action_space["pcs"]
#         self.observation_space = convert_obs_space(self.env.observation_space["pcs"])
#
#
#     def reset(self):
#         return self.env.reset()[0]["pcs"]
#
#     def step(self,action):
#                 #full_action = {key:(action[key] if key == "pcs" else np.zeros(space.shape,dtype=space.dtype)) for key,space in self.env.action_space.items()}
#         full_action = {"iso": np.zeros(3,dtype=np.float32),
#                        "pcs": action}
#         #print("fixed step")
#        # print(self.env.step(full_action))
#         obs_raw, reward, terminated, truncated, info = self.env.step(full_action)
#         done = terminated["pcs"]
#         obs = obs_raw["pcs"]
#         reward_scalar = reward["pcs"] if isinstance(reward, dict) else reward
#         #print(reward["pcs"] if isinstance(reward, dict) else reward)
#         #reward_vector = np.array([reward_scalar], dtype=np.float32)
#         info['reward'] = reward_scalar
#         return obs, reward_scalar, done, info
#
#     def render(self):
#         return self.env.reder()
#
#     def close(self):
#         return self.env.close()
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
