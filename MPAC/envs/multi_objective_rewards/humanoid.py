from realworldrl_suite.utils.multiobj_objectives import Objective
from dm_control.utils import rewards
import numpy as np

_STAND_HEIGHT = 1.4

class HumanoidObjectives(Objective):
    
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
        return 4
            
    def merge_reward(self, task_obj, physics, base_reward, alpha):
        reward_vector = np.zeros(4)
    
        # 1. Forward progress reward
        # Get horizontal velocity (x-axis movement)
        horizontal_vel = physics.center_of_mass_velocity()[0]
        # Reward positive forward motion, normalize to reasonable range
        reward_vector[0] = np.clip(horizontal_vel / 5.0, -1.0, 1.0)
        
        # 2. Control cost (negative reward for large actions)
        actions = physics.control()
        # Quadratic control cost, normalized between 0 and 1
        control_cost = -np.sum(np.square(actions))
        reward_vector[1] = np.exp(control_cost / 30.0)  # Scale to reasonable range
        
        # 3. Healthy reward (based on basic stability conditions)
        height = physics.center_of_mass_position()[2]  # z-axis position
        orientation = physics.torso_upright()
        is_healthy = (height > 0.8 and  # Minimum height
                    height < 2.0 and   # Maximum height
                    orientation > 0.2)  # Minimum uprightness
        reward_vector[2] = 1.0 if is_healthy else 0.0
        
        # 4. Original base reward for reference
        reward_vector[3] = base_reward
            
        return reward_vector