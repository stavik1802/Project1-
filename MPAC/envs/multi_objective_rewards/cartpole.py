from realworldrl_suite.utils.multiobj_objectives import Objective
from dm_control.utils import rewards
import numpy as np

class CartpoleObjectives(Objective):
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
        """Decomposes cartpole reward into meaningful components.
    
        Returns vector of rewards where maximizing each component helps achieve better control.
        """
    
    
        # 1. Pole angle reward: Encourage keeping pole upright
        
        upright = (physics.pole_angle_cosine() + 1) / 2
        
        # 2. Cart position reward: Encourage staying centered
        centered = rewards.tolerance(physics.cart_position(), task_obj._CART_RANGE, margin=2)
        centered = (1 + centered) / 2
        
        
        small_control = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic')[0]
        small_control = (4 + small_control) / 5
        
        
        small_velocity = rewards.tolerance(physics.angular_vel(), margin=5).min()
        small_velocity = (1 + small_velocity) / 2
        
        
        return np.array([upright.mean() * small_control, small_velocity * centered, base_reward])