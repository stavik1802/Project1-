from gymnasium import Env as GymnasiumEnv
from gymnasium.spaces import Box as GSpaceBox, Dict as GSpaceDict

class GymToGymnasiumWrapper(GymnasiumEnv):
    def __init__(self, gym_env):
        self.env = gym_env
        print("this is type")
        print(gym_env.agent_type)
        self.agent = gym_env.agent_type
        self.partner_policy = gym_env.partner_policy
        self.metadata = getattr(gym_env, "metadata", {})
        self.reward_range = getattr(gym_env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(gym_env, "spec", None)
        self.render_mode = getattr(gym_env, "render_mode", None)

        # Convert spaces to gymnasium-compatible ones
        from gymnasium import spaces as gspaces
        self.action_space = gspaces.Box(
            low=gym_env.action_space.low,
            high=gym_env.action_space.high,
            shape=gym_env.action_space.shape,
            dtype=gym_env.action_space.dtype,
        )
        self.observation_space = gspaces.Box(
            low=gym_env.observation_space.low,
            high=gym_env.observation_space.high,
            shape=gym_env.observation_space.shape,
            dtype=gym_env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}  # Gymnasium expects a tuple: (obs, info)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info  # gymnasium expects 5-tuple

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
