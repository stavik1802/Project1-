from realworldrl_suite.utils.multiobj_objectives import Objective
from dm_control.utils import rewards
import numpy as np

class QuadrupedObjectives(Objective):
    
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
        
        move_reward = rewards.tolerance(
            physics.torso_velocity()[0],
            bounds=(task_obj._desired_speed, float('inf')),
            margin=task_obj._desired_speed,
            value_at_margin=0.5,
            sigmoid='linear')
        
        deviation = np.cos(np.deg2rad(0))
        
        upright = rewards.tolerance(
            physics.torso_upright(),
            bounds=(deviation, float('inf')),
            sigmoid='linear',
            margin=1 + deviation,
            value_at_margin=0)
        
        return np.array([move_reward, upright, move_reward * upright])