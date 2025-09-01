
import random
from typing import Any, List, Mapping, Tuple, Union
import uuid
import numpy as np

class EpisodeTracker:
    """Class for keeping track of current episode time steps for reading observations from data files.

    An EpisodeTracker object is shared amongst the environment and all the components.

    Parameters
    ----------
    simulation_start_time_step: int
        Time step to start reading from data files. 
    simulation_end_time_step: int
        Time step to end reading from data files.
    """
    
    def __init__(self, simulation_start_time_step: int, simulation_end_time_step: int):
        self.__episode = None
        self.__episode_start_time_step = None
        self.__episode_end_time_step = None
        self.__simulation_start_time_step = simulation_start_time_step
        self.__simulation_end_time_step = simulation_end_time_step
        self.reset_episode_index()

    @property
    def episode(self):
        """Current episode index"""

        return self.__episode
    
    @property
    def episode_time_steps(self):
        """Number of time steps in current episode split."""

        return (self.episode_end_time_step - self.episode_start_time_step) + 1
    
    @property
    def simulation_time_steps(self):
        """Number of time steps between `simulation_start_time_step` and `simulation_end_time_step`."""

        return (self.__simulation_end_time_step - self.__simulation_start_time_step) + 1
    
    @property
    def simulation_start_time_step(self):
        """Time step to start reading from data files."""

        return self.__simulation_start_time_step
    
    @property
    def simulation_end_time_step(self):
        """Time step to end reading from data files."""

        return self.__simulation_end_time_step
    
    @property
    def episode_start_time_step(self):
        """Start time step in current episode split."""

        return self.__episode_start_time_step
    
    @property
    def episode_end_time_step(self):
        """End time step in current episode split."""

        return self.__episode_end_time_step

    def next_episode(self, episode_time_steps: Union[int, List[Tuple[int, int]]], rolling_episode_split: bool, random_episode_split: bool, random_seed: int):
        """Advance to next episode and set `episode_start_time_step` and `episode_end_time_step` for reading data files.
        
        Parameters
        ----------
        episode_time_steps: Union[int, List[Tuple[int, int]]], optional
            If type is `int`, it is the number of time steps in an episode. If type is `List[Tuple[int, int]]]` is provided, it is a list of 
            episode start and end time steps between `simulation_start_time_step` and `simulation_end_time_step`. Defaults to (`simulation_end_time_step` 
        - `simulation_start_time_step`) + 1. Will ignore `rolling_episode_split` if `episode_splits` is of type `List[Tuple[int, int]]]`.
        rolling_episode_split: bool, default: False
            True if episode sequences are split such that each time step is a candidate for `episode_start_time_step` otherwise, False to split episodes 
            in steps of `episode_time_steps`.
        random_episode_split: bool, default: False
            True if episode splits are to be selected at random during training otherwise, False to select sequentially.
        """

        self.__episode += 1
        self.__next_episode_time_steps(
            episode_time_steps,
            rolling_episode_split,
            random_episode_split,
            random_seed,
        )
        
    def __next_episode_time_steps(self, episode_time_steps: Union[int, List[Tuple[int, int]]], rolling_episode_split: bool, random_episode_split: bool, random_seed: int):
        """Sets `episode_start_time_step` and `episode_end_time_step` for reading data files."""

        splits = None

        if isinstance(episode_time_steps, List):
            splits = episode_time_steps

        else:
            earliest_start_time_step = self.__simulation_start_time_step 
            latest_start_time_step = (self.__simulation_end_time_step + 1) - episode_time_steps
            
            if rolling_episode_split:
                start_time_steps = range(earliest_start_time_step, latest_start_time_step + 1)
            else:
                start_time_steps = range(earliest_start_time_step, latest_start_time_step + 1, episode_time_steps)

            end_time_steps = np.array(start_time_steps, dtype=int) + episode_time_steps - 1
            splits = np.array([start_time_steps, end_time_steps], dtype=int).T
            splits = splits.tolist()

        if random_episode_split:
            seed = int(random_seed*(self.episode + 1))
            nprs = np.random.RandomState(seed)
            ix = nprs.choice(len(splits) - 1)

        else:
            ix = self.episode%len(splits)

        self.__episode_start_time_step, self.__episode_end_time_step = splits[ix]

    def reset_episode_index(self):
        """Resets episode index to -1 before any simulation."""

        self.__episode = -1

class Environment:
    """Base class for the environment.

    Parameters
    ----------
    seconds_per_time_step: float, default: 3600.0
        Number of seconds in 1 `time_step` and must be set to >= 1.
    random_seed : int, optional
        Pseudorandom number generator seed for repeatable results.
    simulation_start_time_step: int, optional
        Time step to start the simulation. 
    simulation_end_time_step: int, optional
        Time step to end the simulation.
    episode_tracker: EpisodeTracker, optional
        :py:class:`.base.EpisodeTracker` object used to keep track of current episode time steps for reading observations from data files.
    """

    DEFAULT_SECONDS_PER_TIME_STEP = 3600.0
    DEFAULT_RANDOM_SEED_RANGE = (0, 100_000_000)
    
    def __init__(self, seconds_per_time_step: float = None, random_seed: int = None, episode_tracker: EpisodeTracker = None):
        self.seconds_per_time_step = seconds_per_time_step
        self.__uid = uuid.uuid4().hex
        self.random_seed = random_seed
        self.__time_step = None
        self.episode_tracker = episode_tracker

    @property
    def uid(self) -> str:
        r"""Unique environment ID."""

        return self.__uid
    
    @property
    def random_seed(self) -> int:
        """Pseudorandom number generator seed for repeatable results."""

        return self.__random_seed
    
    @property
    def episode_tracker(self) -> EpisodeTracker:
        """:py:class:`citylearn.base.EpisodeTracker` object used to keep track of 
        current episode time steps for reading observations from data files."""

        return self.__episode_tracker

    @property
    def time_step(self) -> int:
        r"""Current environment time step."""

        return self.__time_step

    @property
    def seconds_per_time_step(self) -> float:
        r"""Number of seconds in 1 time step."""

        return self.__seconds_per_time_step
    
    @random_seed.setter
    def random_seed(self, random_seed: int):
        random_seed = random.randint(*self.DEFAULT_RANDOM_SEED_RANGE) if random_seed is None else random_seed
        self.__random_seed = random_seed

    @seconds_per_time_step.setter
    def seconds_per_time_step(self, seconds_per_time_step: float):
        if seconds_per_time_step is None:
            self.seconds_per_time_step = self.DEFAULT_SECONDS_PER_TIME_STEP
        else:
            assert seconds_per_time_step >= 1, 'seconds_per_time_step >= 1'
            self.__seconds_per_time_step = seconds_per_time_step

    @episode_tracker.setter
    def episode_tracker(self, episode_tracker: EpisodeTracker):
        self.__episode_tracker = episode_tracker

    def get_metadata(self) -> Mapping[str, Any]:
        """Returns general static information."""

        return {
            'uid': self.uid,
            'random_seed': self.random_seed,
            'simulation_time_steps': self.episode_tracker.simulation_time_steps,
            'seconds_per_time_step': self.seconds_per_time_step
        }

    def next_time_step(self):
        r"""Advance to next `time_step` value.

        Notes
        -----
        Override in subclass for custom implementation when advancing to next `time_step`.
        """

        self.__time_step += 1
        

    def reset(self):
        r"""Reset environment to initial state.

        Calls `reset_time_step`.

        Notes
        -----
        Override in subclass for custom implementation when reseting environment.
        """

        self.reset_time_step()

    def reset_time_step(self):
        r"""Reset `time_step` to initial state.

        Sets `time_step` to 0.
        """

        self.__time_step = 0