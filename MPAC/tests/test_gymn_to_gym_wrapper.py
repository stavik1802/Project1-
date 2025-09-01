import unittest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from MPAC.envs.wrappers.gymn_to_gym import (
    GymnasiumToGymWrapper, 
    create_safety_spec,
    battery_level_constraint
)

class MockGymnasiumEnv:
    """Mock Gymnasium environment for testing."""
    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(3,))
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.reset_called = False
        self.step_called = False
        self.current_time = 0
        self.count = 0
        self.battery_level = 0.5

    def reset(self):
        self.reset_called = True
        return np.array([0.0, 0.1, 0.2]), {}

    def step(self, action):
        self.step_called = True
        # Create energy-net style info dictionary
        info = {
            # Battery/PCS info
            'battery_level': self.battery_level,
            'battery_action': action[0],
            'net_exchange': 0.1,
            'pcs_exchange_cost': 0.5,
            'pcs_action': action,
            
            # Market/ISO info
            'iso_buy_price': 1.0,
            'iso_sell_price': 1.5,
            'predicted_demand': 2.0,
            'realized_demand': 2.1,
            'net_demand': 2.1,
            'dispatch': 2.0,
            'shortfall': 0.1,
            'dispatch_cost': 2.0,
            'reserve_cost': 0.15,
            'price_spread': 0.5,
            
            # General info
            'time': self.current_time,
            'step': self.count,
            'terminated': False,
            'truncated': False
        }
        return np.array([0.1, 0.2, 0.3]), 1.0, False, False, info

class TestGymnasiumToGymWrapper(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.mock_env = MockGymnasiumEnv()
        self.env_setup_kwargs = {
            'safety_coeff': 1.0,
            'energy_constraints': ['battery'],
            'constraints_all': False
        }
        self.safety_config = create_safety_spec(self.env_setup_kwargs)
        self.wrapped_env = GymnasiumToGymWrapper(self.mock_env, self.safety_config)

    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertTrue(self.wrapped_env.safety_enabled)
        self.assertEqual(self.wrapped_env.safety_coeff, 1.0)
        self.assertIn('battery', self.wrapped_env.constraints)

    def test_reset(self):
        """Test reset functionality."""
        obs = self.wrapped_env.reset()
        self.assertTrue(self.mock_env.reset_called)
        self.assertTrue(isinstance(obs, np.ndarray))
        # Should be base observation (3) + constraint observation (1)
        self.assertEqual(obs.shape, (4,))  # Updated from (3,) to include constraint observation

    def test_step(self):
        """Test step functionality."""
        action = np.array([0.5, -0.5])
        obs, reward, done, info = self.wrapped_env.step(action)
        
        self.assertTrue(self.mock_env.step_called)
        self.assertTrue(isinstance(obs, np.ndarray))
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))
        self.assertTrue(isinstance(info, dict))
        
        # Check observation shape includes constraint
        self.assertEqual(obs.shape, (4,))  # Updated to include constraint observation
        
        # Check safety information in info dict
        self.assertIn('cost', info)
        self.assertIn('constraints', info)
        self.assertIn('reward', info)

    def test_constraint_evaluation(self):
        """Test constraint evaluation."""
        obs = np.array([0.1, 0.2, 0.3])
        info = {}
        constraints = self.wrapped_env._check_constraints(obs, info)
        self.assertTrue(isinstance(constraints, np.ndarray))
        self.assertEqual(len(constraints), 1)  # One constraint (battery)

    def test_observation_processing(self):
        """Test observation processing."""
        # Test with numpy array
        obs_array = np.array([0.1, 0.2, 0.3])
        processed_obs = self.wrapped_env._process_obs(obs_array)
        self.assertTrue(isinstance(processed_obs, np.ndarray))
        self.assertEqual(processed_obs.shape, (3,))

        # Test with dictionary
        obs_dict = {'pos': np.array([0.1, 0.2]), 'vel': np.array([0.3])}
        processed_obs = self.wrapped_env._process_obs(obs_dict)
        self.assertTrue(isinstance(processed_obs, np.ndarray))
        self.assertEqual(processed_obs.shape, (3,))

    def test_safety_spec_creation(self):
        """Test safety specification creation."""
        env_setup_kwargs = {
            'safety_coeff': 0.5,
            'energy_constraints': ['battery'],
            'constraints_all': False
        }
        safety_spec = create_safety_spec(env_setup_kwargs)
        
        self.assertTrue(safety_spec['enable'])
        self.assertTrue(safety_spec['observations'])
        self.assertEqual(safety_spec['safety_coeff'], 0.5)
        self.assertIn('constraints', safety_spec)
        self.assertIn('battery', safety_spec['constraints'])

    def test_battery_constraint(self):
        """Test battery constraint function with different battery levels."""
        obs = np.array([0.1, 0.2, 0.3])
        
        # Test battery level within limits (20% to 80%)
        info = {'battery_level': 0.5}  # 50% - should be safe
        result = battery_level_constraint(obs, info)
        self.assertTrue(result)
        
        info = {'battery_level': 0.2}  # 20% - should be safe (boundary)
        result = battery_level_constraint(obs, info)
        self.assertTrue(result)
        
        info = {'battery_level': 0.8}  # 80% - should be safe (boundary)
        result = battery_level_constraint(obs, info)
        self.assertTrue(result)
        
        # Test battery level outside limits
        info = {'battery_level': 0.1}  # 10% - should violate
        result = battery_level_constraint(obs, info)
        self.assertFalse(result)
        
        info = {'battery_level': 0.9}  # 90% - should violate
        result = battery_level_constraint(obs, info)
        self.assertFalse(result)
        
        # Test missing battery level
        info = {}  # No battery level - should violate
        result = battery_level_constraint(obs, info)
        self.assertFalse(result)
        
        # Test integration with wrapper
        self.mock_env.battery_level = 0.15  # Set battery level to violate constraint
        _, _, _, info = self.wrapped_env.step(np.array([0.5, -0.5]))
        self.assertEqual(info['cost'], 1.0)  # Should have cost of 1.0 due to violation
        self.assertFalse(info['constraints'][0])  # Constraint should be violated
        
        self.mock_env.battery_level = 0.5  # Set battery level to safe value
        _, _, _, info = self.wrapped_env.step(np.array([0.5, -0.5]))
        self.assertEqual(info['cost'], 0.0)  # Should have cost of 0.0 (no violation)
        self.assertTrue(info['constraints'][0])  # Constraint should be satisfied

    def test_cost_calculation(self):
        """Test cost calculation when constraints are violated."""
        # Override battery constraint to return False (violated)
        self.wrapped_env.constraints['battery'] = lambda obs, info: False
        
        action = np.array([0.5, -0.5])
        _, _, _, info = self.wrapped_env.step(action)
        
        self.assertEqual(info['cost'], 1.0)  # Cost should be 1.0 when constraint is violated
        self.assertFalse(info['constraints'][0])  # Constraint should be False

    def test_observation_space_conversion(self):
        """Test observation space conversion from Gymnasium to Gym."""
        self.assertTrue(hasattr(self.wrapped_env, 'observation_space'))
        self.assertTrue(hasattr(self.wrapped_env.observation_space, 'low'))
        self.assertTrue(hasattr(self.wrapped_env.observation_space, 'high'))

    def test_info_dictionary_contents(self):
        """Test the contents of the info dictionary returned by step."""
        action = np.array([0.5, -0.5])
        _, _, _, info = self.wrapped_env.step(action)
        
        # Check basic safety-related contents
        self.assertIn('cost', info)
        self.assertTrue(isinstance(info['cost'], float))
        self.assertGreaterEqual(info['cost'], 0.0)
        self.assertLessEqual(info['cost'], 1.0)
        
        # Check constraint information
        self.assertIn('constraints', info)
        self.assertTrue(isinstance(info['constraints'], np.ndarray))
        self.assertEqual(len(info['constraints']), 1)  # One constraint (battery)
        
        # Check energy-net specific info
        # Battery/PCS related
        self.assertIn('battery_level', info)
        self.assertIn('battery_action', info)
        self.assertIn('net_exchange', info)
        self.assertIn('pcs_exchange_cost', info)
        self.assertIn('pcs_action', info)
        
        # Market/ISO related
        self.assertIn('iso_buy_price', info)
        self.assertIn('iso_sell_price', info)
        self.assertIn('predicted_demand', info)
        self.assertIn('realized_demand', info)
        self.assertIn('net_demand', info)
        self.assertIn('dispatch', info)
        self.assertIn('shortfall', info)
        self.assertIn('dispatch_cost', info)
        self.assertIn('reserve_cost', info)
        self.assertIn('price_spread', info)
        
        # General info
        self.assertIn('time', info)
        self.assertIn('step', info)
        self.assertIn('terminated', info)
        self.assertIn('truncated', info)
        self.assertIn('reward', info)
        
        # Test types of key values
        self.assertTrue(isinstance(info['battery_level'], float))
        self.assertTrue(isinstance(info['net_exchange'], float))
        self.assertTrue(isinstance(info['iso_buy_price'], float))
        self.assertTrue(isinstance(info['predicted_demand'], float))
        self.assertTrue(isinstance(info['time'], (int, float)))
        self.assertTrue(isinstance(info['step'], int))

if __name__ == '__main__':
    unittest.main() 