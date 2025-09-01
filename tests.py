from MPAC.envs.wrappers.energy_net_wrapper import EnergyNetWrapper
from enum import Enum
from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern


# Define the dictionary with valid references
env_kargs = {
    'safety_coeff': None,
    'rwrl_constraints': None,
    'rwrl_constraints_all': False,
    'perturb_param_name': None,
    'perturb_param_value': None,
    'perturb_param_min': None,
    'perturb_param_max': None,
    'action_noise_std': 0.0,
    'observation_noise_std': 0.0,
    'multiobj_coeff': 0.0,
    'multiobj_enable': False,
    'pricing_policy': PricingPolicy.CONSTANT,
    'cost_type': CostType.CONSTANT,
    'demand_pattern': DemandPattern.CONSTANT,
    'env_config_path': '/home/stav.karasik/MPAC/energy-net/configs/environment_config.yaml',
    'iso_config_path': '/home/stav.karasik/MPAC/energy-net/configs/iso_config.yaml',
    'pcs_unit_config_path': '/home/stav.karasik/MPAC/energy-net/configs/pcs_unit_config.yaml',
    'log_file': '/home/stav.karasik/MPAC/energy-net/logs/environments.log'
}
env = EnergyNetWrapper(env_kargs)
#print(env.action_space)
action = env.action_space.sample()
print(env.reset())
#print(env.step(action))