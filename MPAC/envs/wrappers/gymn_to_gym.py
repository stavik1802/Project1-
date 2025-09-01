import gym
import gymnasium as gymnasium
from gymnasium.spaces import Box as GymnasiumBox, Dict as GymnasiumDict
from gym.spaces import Box as GymBox, Dict as GymDict
import numpy as np
from numpy.random import uniform,normal

from typing import Optional
from gym.spaces import Box as GymBox
from collections import OrderedDict
from typing import Optional, Dict, Any
from MPAC.envs.wrappers.safe_robust import create_safety_spec, create_perturb_spec



def compute_battery_diff_fraction(
    prev_level, curr_level,
    step_seconds=1800,
    capacity_mAh=9600.0,
    max_c_rate=1.0,          # 1C: can move at most capacity_Ah per hour
    levels_are_percent=True, # True if levels are 0..100; False if already 0..1
):
    """
    Returns battery_diff as a FRACTION of pack capacity moved in this step (0..1),
    """
    if prev_level is None or curr_level is None:
        return 0.0

    # normalize to 0..1
    if levels_are_percent:
        p = float(prev_level) / 100.0
        c = float(curr_level) / 100.0
    else:
        p = float(prev_level)
        c = float(curr_level)
    frac_raw = abs(p - c) 

    cap_Ah = capacity_mAh / 1000.0
    max_frac_this_step = max_c_rate * (step_seconds / 3600.0)
    frac_capped = min(frac_raw, max_frac_this_step)
    return max(0.0, min(1.0, frac_capped))



# ---- Thermal model (Joule heating + linear cooling) ----
STEP_SECONDS = 1800  # 30 minutes

def update_temperature_model_based(
    battery_temperature,            # °C
    battery_diff,                   
    capacity_mAh=9600.0,            # mAh
    internal_resistance=0.5,       
    mass_kg=0.2,                    # kg
    c_p=500.00,                      # J/(kg·K)
    ambient_C=25.0,                 # C
    cooling_coeff=3e-4,              
    max_c_rate=0.8,              
    debug=False,
):
    """
    battery_diff:
      - if diff_mode="fraction": fraction of capacity used this step (e.g., 0.02 = 2%/step)
      - if diff_mode="mAh":      absolute mAh drawn this step
    Temperature update:
      Joule heat ΔT_heat = (I^2 R Δt) / (m c_p)
      Cooling: T_next = T_amb + (T + ΔT_heat - T_amb) * exp(-k Δt)
    """
    cap_Ah = capacity_mAh / 1000.0
    frac = max(0.0, min(1.0, float(battery_diff)))
    I = (cap_Ah * 3600.0 * frac) / STEP_SECONDS 

    # Enforce C-rate cap
    I_max = max_c_rate * cap_Ah
    if I > I_max:
        if debug:
            print(f"[thermal] Current clipped: I={I:.2f}A -> {I_max:.2f}A (C-rate cap)")
        I = I_max

    # Safety clamps on physical params
    internal_resistance = max(1e-4, float(internal_resistance))
    mass_kg = max(1e-3, float(mass_kg))
    c_p = max(10.0, float(c_p))
    k = max(0.0, float(cooling_coeff))

    # Joule heating
    P_heat = (I ** 2) * internal_resistance             # W
    deltaT_heat = (P_heat * STEP_SECONDS) / (mass_kg * c_p)

    # Exponential cooling towards ambient
    # T_next = T_amb + (T + ΔT_heat - T_amb) * exp(-k Δt)
    from math import exp
    T_excess = (battery_temperature + deltaT_heat) - ambient_C
    T_next = ambient_C + T_excess * exp(-k * STEP_SECONDS)

    if debug:
        print(f"[thermal] I={I:.3f}A, P={P_heat:.3f}W, dT_heat={deltaT_heat:.4f}°C, T->{T_next:.2f}°C")


    return float(T_next)



class GymnasiumToGymWrapper(gym.Wrapper):
    def __init__(self, gymnasium_env, env_setup_kwargs=None, pert_flag = False, pert_min =0.9, pert_max = 1.1):
        super().__init__(gymnasium_env)
        def convert_obs_space(space):
            if isinstance(space, gymnasium.spaces.Box):
                return GymBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
            return space
        # Convert Gymnasium space to Gym space
        space = self.env.action_space
        self.action_space = GymBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        self.observation_space = convert_obs_space(self.env.observation_space)
        self.r_dim = 1 #stav added
        self.previous_battery = None
        self.previous_temp = None
        self.battery_temperature = 25.0

        # Initialize safety
        self.safety_enabled = True # TODO: From agrs
        self.safety_spec = create_safety_spec(env_setup_kwargs)

        # Initialize pertubs

        # battery_pert = lambda obs, info, pert: obs.__setitem__(0, obs[0] * pert)
        # self.pert_spec = {
        #     'enable': True,
        #     'min': pert_min,
        #     'max': pert_max,
        #     'scheduler': 'uniform',
        #     'perturbation': battery_pert,
        # }
        self.pert_spec = create_perturb_spec(env_setup_kwargs)

    def reset(self,seed: Optional[int] = None):
        obs_all = self.env.reset(seed=seed)[0]
        processed_obs = self._process_obs(obs_all)
        self.battery_temperature = 25.0
        self.previous_battery = None
        self.previous_temp = None
        return processed_obs

    def step(self, action):
        obs_raw, reward, truncated, terminated, info = self.env.step(action)
        done = terminated or truncated
        obs = np.asarray(obs_raw, dtype=np.float32).flatten()

        # ensure dict and expose battery level
        info = {} if not isinstance(info, dict) else dict(info)
        info['battery_level'] = float(obs[0])

        # temperature update
        frac = compute_battery_diff_fraction(
            prev_level=self.previous_battery, curr_level=obs[0],
            step_seconds=STEP_SECONDS, capacity_mAh=9600.0, max_c_rate=0.8
        )
        self.battery_temperature = float(update_temperature_model_based(self.battery_temperature, frac))
        info['battery_temperature'] = self.battery_temperature

        # safety: continuous satisfaction in info['constraints'], avg cost in info['cost']
        if self.safety_enabled:
            constraints_sat, avg_cost = self._check_constraints(
                obs, info, previous_battery=self.previous_battery, previous_temp=self.previous_temp
            )
            info['constraints'] = constraints_sat   # continuous [0,1], 1=safe
            info['cost'] = avg_cost                 # continuous [0,1], 0=safe
        # if np.random.rand() < 0.05:  # print ~5% of steps to avoid spam
        #     print(
        #         f"[safety] "
        #         f"battery={info.get('battery_level', None):.1f}% "
        #         f"temp={info.get('battery_temperature', None):.1f}°C | "
        #         f"constraints={info['constraints']} | "
        #         f"cost={info['cost']:.3f}"
        #     )

        # optional perturbations
        if self.pert_spec['enable'] == True:
            self._apply_perturbation(obs, info)

        reward_scalar = float(np.asarray(reward if isinstance(reward, dict) else reward).squeeze())

        # update memory
        self.previous_battery = obs[0]
        self.previous_temp = info['battery_temperature']
        return obs, reward_scalar, done, info


    def _check_constraints(self, observation, info, previous_battery, previous_temp):
        """
        Constraint fns return COST in [0,1] (0=safe, 1=violation).
        We return:
        - constraints: continuous SATISFACTION in [0,1] = (1 - cost)
        - cost: average cost across constraints (continuous [0,1])
        """
        spec = self.safety_spec
        constraint_dict = spec.get('constraints', None)
        safety_coeff = float(spec.get('safety_coeff', 1.0))

        if not constraint_dict:
            return np.array([1.0], dtype=np.float32), 0.0

        costs = []
        sats  = []

        for name, func in constraint_dict.items():
            try:
                if name == 'battery':
                    c = float(func(observation, info, previous_battery))
                elif name == 'thermal':
                    c = float(func(observation, info, previous_temp))
                else:
                    c = float(func(observation, info))
            except Exception as e:
                print(f"Warning: Constraint {name} failed with error: {e}")
                c = 1.0  # worst-case if eval fails

            # scale and clamp cost
            c = max(0.0, min(1.0, safety_coeff * c))
            s = 1.0 - c  # continuous satisfaction

            costs.append(c)
            sats.append(s)

        costs = np.array(costs, dtype=np.float32)
        sats  = np.array(sats,  dtype=np.float32)
        avg_cost = float(costs.mean())

        return sats, avg_cost


    def _apply_perturbation(self, observation, info):
        pert_fun = self.pert_spec['perturbation']
        if self.pert_spec['scheduler'] == 'uniform':
            pert = np.random.uniform(self.pert_spec['min'], self.pert_spec['max'])
        else:
            pert = self.pert_spec['start']

        pert_fun(observation, info, pert)



    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def _process_obs(self, obs):
        """Convert obs to flat NumPy array if needed"""
        if isinstance(obs, dict):
            # Flatten dict of arrays
            return np.concatenate([np.asarray(v).flatten() for v in obs.values()]).astype(np.float32)
        return np.asarray(obs, dtype=np.float32)


    def seed(self, seed: Optional[int] = None):
        try:
            self.env.reset(seed=seed)
        except Exception as e:
            print(f"[WARNING] Failed to seed underlying env: {e}")





def extract_env_info(wrapped_env):
        """
        Recursively unwrap the environment and extract key information.
        Supports VecEnv (like DummyVecEnv) as well as standard Gym wrappers.
        """
        # If it's a VecEnv (e.g., DummyVecEnv), unwrap first env
        if hasattr(wrapped_env, 'envs'):
            env = wrapped_env.envs[0]
        else:
            env = wrapped_env

        # Recursively unwrap
        while hasattr(env, 'env'):
            env = env.env

        # Extract info
        info = {
            'class': env.__class__.__name__,
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'reward_range': getattr(env, 'reward_range', None),
            '_max_episode_steps': getattr(env, '_max_episode_steps', None),
            'spec_id': getattr(env.spec, 'id', None) if getattr(env, 'spec', None) else None,
        }

        return info
