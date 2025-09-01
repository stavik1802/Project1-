import logging
from abc import ABC, abstractmethod
from typing import List, Any, Mapping

import numpy as np

from energy_net.defs import Bounds
LOGGER = logging.getLogger()


class Agent(ABC):
    @abstractmethod
    def get_action(self, observation, deterministic=False):
        pass

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def eval(self):
        pass

    @property
    def observation_names(self) -> List[List[str]]:
        """Names of active observations that can be used to map observation values."""

        return self.__observation_names

    @property
    def action_names(self) -> List[List[str]]:
        """Names of active actions that can be used to map action values."""

        return self.__action_names

    @property
    def observation_space(self) -> List[Bounds]:
        """Format of valid observations."""

        return self.__observation_space

    @property
    def action_space(self) -> List[Bounds]:
        """Format of valid actions."""

        return self.__action_space

    @property
    def episode_time_steps(self) -> int:
        return self.__episode_time_steps

    @property
    def action_dimension(self) -> List[int]:
        """Number of returned actions."""

        return [s.shape[0] for s in self.action_space]

    @property
    def actions(self) -> List[List[List[Any]]]:
        """Action history/time series."""

        return self.__actions

    # @env.setter
    # def env(self, env: EnergyNetEnv):
    #     self.__env = env

    @observation_names.setter
    def observation_names(self, observation_names: List[List[str]]):
        self.__observation_names = observation_names

    @action_names.setter
    def action_names(self, action_names: List[List[str]]):
        self.__action_names = action_names

    @observation_space.setter
    def observation_space(self, observation_space: List[Bounds]):
        self.__observation_space = observation_space

    @action_space.setter
    def action_space(self, action_space: List[Bounds]):
        self.__action_space = action_space

    @episode_time_steps.setter
    def episode_time_steps(self, episode_time_steps: int):
        """Number of time steps in one episode."""

        self.__episode_time_steps = episode_time_steps

    @actions.setter
    def actions(self, actions: List[List[Any]]):
        for i in range(len(self.action_space)):
            self.__actions[i][self.time_step] = actions[i]

    def learn(self, episodes: int = None, deterministic: bool = None, deterministic_finish: bool = None,
              logging_level: int = None):
        """Train agent.

        Parameters
        ----------
        episodes: int, default: 1
            Number of training episode >= 1.
        deterministic: bool, default: False
            Indicator to take deterministic actions i.e. strictly exploit the learned policy.
        deterministic_finish: bool, default: False
            Indicator to take deterministic actions in the final episode.
        logging_level: int, default: 30
            Logging level where increasing the number silences lower level information.
        """

        episodes = 1 if episodes is None else episodes
        deterministic_finish = False if deterministic_finish is None else deterministic_finish
        deterministic = False if deterministic is None else deterministic
        self.__set_logger(logging_level)

        for episode in range(episodes):
            deterministic = deterministic or (deterministic_finish and episode >= episodes - 1)
            observations, _ = self.network.reset()
            self.episode_time_steps = self.episode_tracker.episode_time_steps
            terminated = False
            time_step = 0
            rewards_list = []

            while not terminated:
                actions = self.select_actions(observations, deterministic=deterministic)

                # apply actions
                next_observations, rewards, terminated, truncated, _ = self.network.step(actions)
                rewards_list.append(rewards)

                # update
                if not deterministic:
                    self.update(observations, actions, rewards, next_observations, terminated=terminated,
                                truncated=truncated)
                else:
                    pass

                observations = [o for o in next_observations]

                logging.debug(
                    f'Time step: {time_step + 1}/{self.episode_time_steps},' \
                    f' Episode: {episode + 1}/{episodes},' \
                    f' Actions: {actions},' \
                    f' Rewards: {rewards}'
                )

                time_step += 1

            rewards = np.array(rewards_list, dtype='float')
            rewards_summary = {
                'min': rewards.min(axis=0),
                'max': rewards.max(axis=0),
                'sum': rewards.sum(axis=0),
                'mean': rewards.mean(axis=0)
            }
            logging.info(f'Completed episode: {episode + 1}/{episodes}, Reward: {rewards_summary}')

    def select_actions(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[List[float]]
            Action values
        """

        actions = [list(s.sample()) for s in self.action_space]
        self.actions = actions
        self.next_time_step()
        return actions

    def __set_logger(self, logging_level: int = None):
        """Set logging level."""

        logging_level = 30 if logging_level is None else logging_level
        assert logging_level >= 0, 'logging_level must be >= 0'
        LOGGER.setLevel(logging_level)

    def update(self, *args, **kwargs):
        """Update replay buffer and networks.

        Notes
        -----
        This implementation does nothing but is kept to keep the API for all agents similar during simulation.
        """

        pass

    def next_time_step(self):
        super().next_time_step()

        for i in range(len(self.action_space)):
            self.__actions[i].append([])

    def reset(self):
        super().reset()
        self.__actions = [[[]] for _ in self.action_space]

    def is_done(self) -> bool:
        """Check if the episode is done."""

        return False

    def get_info(self) -> Mapping[str, Any]:
        """Get additional information about the stratigic entity."""

        return {}



