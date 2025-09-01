from realworldrl_suite.utils.multiobj_objectives import Objective
from dm_control.utils import rewards
import numpy as np

class ManipulatorObjectives(Objective):
    
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
        return 2
    
    def merge_reward(self, task_obj, physics, base_reward, alpha):
        """Decomposes walker reward into meaningful components."""
        
        reward_vector = np.zeros(2)
    
        # 1. Distance to target
        ball_dist = physics.site_distance('ball', 'target_ball')
        reward_vector[0] = task_obj._is_close(ball_dist)
        
        if hasattr(task_obj, '_prev_ball_dist'):
            progress = task_obj._prev_ball_dist - ball_dist  # Positive if getting closer
            reward_vector[1] = np.clip(progress * 5, -1, 1)  # Scale and clip the progress reward
        else:
            reward_vector[1] = 0
        task_obj._prev_ball_dist = ball_dist
        
        
        return reward_vector
        