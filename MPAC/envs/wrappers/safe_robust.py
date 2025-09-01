from collections import OrderedDict
from typing import Optional, Dict, Any

THERMAL_LOWER_BOUND = 15.0
THERMAL_UPPER_BOUND = 35.0
BATTARY_LOWER_BOUND = 20.0
BATTARY_UPPER_BOUND = 80.0


# ============ Safe functions ===============
# ---- Constraint: returns cost in [0,1] (0 == safe) ----
def thermal_constraint(obs, info, previous=None):
    level = info.get('battery_level', None)
    temp = info.get('battery_temperature', None)
    if temp is None:
        return 1.0  # worst if unknown

    # penalize if battery level did not change
    if previous is not None and level == previous:
        return 1.0

    if temp < THERMAL_LOWER_BOUND:
        diff = THERMAL_LOWER_BOUND - temp
        rng  = THERMAL_LOWER_BOUND
        return min(1.0, diff / rng)
    elif temp > THERMAL_UPPER_BOUND:
        diff = temp - THERMAL_UPPER_BOUND
        rng  = max(1.0, 50.0 - THERMAL_UPPER_BOUND)
        return min(1.0, diff / rng)
    else:
        return 0.0   # safe

def battery_level_constraint(obs, info, previous=None):
    level = info.get('battery_level', None)
    if level is None:
        return 1.0
    if previous is not None and previous == level:
        return 1.0

    if level < BATTARY_LOWER_BOUND:
        diff = BATTARY_LOWER_BOUND - level
        rng  = BATTARY_LOWER_BOUND
        return min(1.0, diff / rng)
    elif level > BATTARY_UPPER_BOUND:
        diff = level - BATTARY_UPPER_BOUND
        rng  = 100.0 - BATTARY_UPPER_BOUND
        return min(1.0, diff / rng)
    else:
        return 0.0   # safe


# # ============ Safe functions ===============
# # ---- Constraint: returns score in [0,1] (1 == safe) ----
# def thermal_constraint(obs, info, previous=None):
#     level = info.get('battery_level', None)
#     temp = info.get('battery_temperature', None)
#     if temp is None:
#         return 0.0  # worst if unknown

#     # if "previous" is the same we want to penalize cost
#     if previous is not None and level == previous:
#         return 0.0

#     if temp < THERMAL_LOWER_BOUND:
#         diff = THERMAL_LOWER_BOUND - temp
#         rng  = THERMAL_LOWER_BOUND                # normalize by distance to bound from 0C
#     elif temp > THERMAL_UPPER_BOUND:
#         diff = temp - THERMAL_UPPER_BOUND
#         rng  = max(1.0, 50.0 - THERMAL_UPPER_BOUND)  # assume 50C is "very unsafe"
#     else:
#         return 1.0

#     return max(0.0, 1.0 - diff / rng)

# # def compute_thermal_cost(obs, info) -> float:
# #     battery_temperature = info.get('battery_temperature', None)
# #     if battery_temperature is None:
# #         return False
# #     print('battery_temperature', battery_temperature)
# #     if battery_temperature < THERMAL_LOWER_BOUND:
# #         return 1 - (THERMAL_LOWER_BOUND - self.battery_temperature) ** 2
# #     elif battery_temperature > THERMAL_UPPER_BOUND:
# #         return 1 - (battery_temperature - THERMAL_UPPER_BOUND) ** 2
# #     return 1.0


# def battery_level_constraint(obs, info,previous = None):
#     level = info.get('battery_level', None)
#     if level is None:
#         return 0.0
#     if previous is not None and previous == level:
#         return 1.0
#     if level < BATTARY_LOWER_BOUND:
#         diff = BATTARY_LOWER_BOUND - level
#         range_ = BATTARY_LOWER_BOUND
#     elif level > BATTARY_UPPER_BOUND:
#         diff = level - BATTARY_UPPER_BOUND
#         range_ = 100.0 - BATTARY_UPPER_BOUND
#     else:
#         return 1.0

#     return max(0.0, 1.0 - diff / range_)


# ============ Pert functions ===============
def battery_pert(obs, info, pert):
    obs[0] *= pert
    if (obs[0] > 100.0):
        obs[0] = 100.0

# ============ Constraints dict =============
energy_net_constraints = {
    'battery': battery_level_constraint,
    'thermal' : thermal_constraint,
    # 'termal_cost' : compute_thermal_cost,
}

# =========== Pertub names ==================
energy_net_pertubs = {
    'battery'   : battery_pert,
}

# ============== Initielizers ===============
def create_safety_spec(env_setup_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Creates safety_spec dictionary.

    Args:
        env_setup_kwargs (dict): Environment setup parameters including:
            - safety_coeff (float): Safety coefficient
            - energy_constraints (list): List of constraint names to enable
            - constraints_all (bool): Whether to enable all constraints

    Returns:
        dict: Safety specification dictionary
    """
    safety_spec = {
        'enable': True,
        'observations': True,
        'safety_coeff': 1.0,
        'constraints' : None,
    }

    constraints_list = env_setup_kwargs['energy_constraints']

    if env_setup_kwargs['energy_constraints_all']:
        constraints_list = list(energy_net_constraints.keys())

    if constraints_list:
        constraints = OrderedDict()
        for constraint in constraints_list:
            constraints[constraint] = energy_net_constraints[constraint]
        safety_spec['constraints'] = constraints

    return safety_spec

def create_perturb_spec(env_setup_kwargs):
    """Creates perturb_spec dictionary."""
    perturb_spec = {
        'enable':       False, # Whether to enable perturbations
        'period':       1, # Number of episodes between perturbation changes.
        'scheduler':    'constant', # Specifies which scheduler to apply.
        'perturbation' : None
    }

    if not env_setup_kwargs['perturb_param_name']:
        return perturb_spec

    perturb_spec['param'] = env_setup_kwargs['perturb_param_name']

    assert (perturb_spec['param'] in energy_net_pertubs)
    perturb_spec['perturbation'] = energy_net_pertubs[perturb_spec['param']]

    perturb_min = env_setup_kwargs['perturb_param_min']
    perturb_max = env_setup_kwargs['perturb_param_max']
    print(f"this is setup val {env_setup_kwargs['perturb_param_value']}")

    if env_setup_kwargs['perturb_param_value'] is not None:
        print("1")
        perturb_spec['enable'] = True
        perturb_spec['start'] = env_setup_kwargs['perturb_param_value'] # Indicates initial value of perturbed parameter
        perturb_spec['min'] = env_setup_kwargs['perturb_param_value']
        perturb_spec['max'] = env_setup_kwargs['perturb_param_value']
    elif (perturb_min is not None) and (perturb_max is not None):
        print("2")
        perturb_spec['enable'] = True
        perturb_spec['start'] = (perturb_min + perturb_max) / 2
        perturb_spec['min'] = perturb_min
        perturb_spec['max'] = perturb_max
        perturb_spec['scheduler'] = 'uniform' # set the perturbation to a uniform random value within [min, max]

    return perturb_spec