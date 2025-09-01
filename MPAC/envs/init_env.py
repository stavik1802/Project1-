"""Interface to environments."""
import copy
from MPAC.envs.wrappers.gymn_to_gym import GymnasiumToGymWrapper
from energy_net.env.pcs_unit_v0 import PCSUnitEnv
from energy_net.market.pricing_policy import PricingPolicy
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
#changed init_env to support energy_net env
def init_env(env_type,env_name,task_name,env_setup_kwargs):
    """Creates environments.
    
    Args:
        env_type (str): environment type (rwrl, dmc)
        env_name (str): environment / domain name
        task_name (str): task name
        env_setup_kwargs (dict): setup parameters
    
    Returns:
        Training environment and evaluation environment.
    """
    env_eval_setup_kwargs = copy.deepcopy(env_setup_kwargs)
    env_eval_setup_kwargs['perturb_param_min'] = None
    env_eval_setup_kwargs['perturb_param_max'] = None

    if env_type == 'rwrl':
        from MPAC.envs.wrappers.rwrl_wrapper import make_rwrl_env
        env = make_rwrl_env(env_name,task_name,env_setup_kwargs)
        env_eval_nominal = make_rwrl_env(env_name,task_name,env_eval_setup_kwargs)
    elif env_type == 'dmc':
        from MPAC.envs.wrappers.dmc_wrapper import make_dmc_env
        env = make_dmc_env(env_name,task_name)
        env_eval_nominal = make_dmc_env(env_name,task_name)
    elif env_type == 'energy_net':
        energy_env_kwargs = copy.deepcopy(env_setup_kwargs)
        from energy_net.env.pcs_env import make_pcs_env_zoo
        from MPAC.config_env_pcs import get_env_kwargs,get_env_setup_kwargs
        env_kwargs = get_env_setup_kwargs()
        env_setup_kwargs = get_env_setup_kwargs()
        iso_policy_path = "MPAC/model_iso/td3_iso_best.zip"
        norm_path = "MPAC/model_iso/td3_iso_best_norm.pkl"
        env = PCSUnitEnv(
            norm_path=norm_path,
            trained_iso_model_path=iso_policy_path,
            cost_type=CostType.CONSTANT,
            demand_pattern=DemandPattern.DOUBLE_PEAK,
            render_mode=None,
        )
        env_eval_nominal = PCSUnitEnv(
            norm_path=norm_path,
            trained_iso_model_path=iso_policy_path,
            cost_type=CostType.CONSTANT,
            demand_pattern=DemandPattern.DOUBLE_PEAK,
            render_mode=None,
        )
        #configure perturbation
        pert_flag = False
        pert_min = 1.8
        pert_max = 2.0
        env = GymnasiumToGymWrapper(env, energy_env_kwargs, pert_flag,pert_min,pert_max)
        env_eval_nominal = GymnasiumToGymWrapper(env_eval_nominal, energy_env_kwargs, pert_flag,pert_min,pert_max)
        # perturb_min = 0.8 #configure for perturbation
        # perturb_max = 1.2 #configure for perturbation
        # if perturb_min != 1.0 or perturb_max != 1.0:
        #     from MPAC.pert_wrap import BatteryActionPerturbationWrapper
        #     env = BatteryActionPerturbationWrapper(env, perturbation_min=perturb_min, perturbation_max=perturb_max)
   

    else:
        raise ValueError('Supported env_type: rwrl, dmc, energy_net')
    
    return env, env_eval_nominal