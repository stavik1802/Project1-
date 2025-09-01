from realworldrl_suite.utils.multiobj_objectives import Objective
from dm_control.utils import rewards
import numpy as np

_STAND_HEIGHT = 1.4

class WalkerObjectives(Objective):
    
    def get_objectives(self, task_obj):
        """Returns the safety objective: sum of satisfied constraints."""
        if task_obj.safety_enabled:
            num_constraints = float(task_obj.constraints_obs.shape[0])
            num_satisfied = task_obj.constraints_obs.sum()
            s_reward = num_satisfied / num_constraints
            return np.array([s_reward])
        else:
            raise Exception('Safety not enabled.  Safety-based multi-objective reward'
                            ' requires safety spec to be enabled.')
    @staticmethod
    def get_dim_objectives():
        return 3
    
    def merge_reward(self, task_obj, physics, base_reward, alpha):
        """Decomposes walker reward into meaningful components."""
        # The agent’s head must be above _STAND_HEIGHT
        standing = rewards.tolerance(physics.torso_height(),
                                bounds=(_STAND_HEIGHT, float('inf')),
                                margin=_STAND_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4
        
        return np.array([base_reward, upright.mean(), stand_reward])