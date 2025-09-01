import numpy as np
import gym

from MPAC.envs import init_env
from MPAC.actors import init_actor
from MPAC.common.normalizer import RunningNormalizers
from MPAC.common.samplers import trajectory_sampler
from MPAC.common.seeding import init_seeds
from MPAC.common.train_parser import env_kwargs as env_kwargs_keys
from MPAC.common.train_parser import actor_kwargs as actor_kwargs_keys

from MPAC.envs.wrappers.safe_robust import create_safety_spec, create_perturb_spec

def show_perturbations(import_logs):
    try:
        env_kwargs = import_logs[0]['param']['env_kwargs']
        domain = env_kwargs['env_name'].replace('_', '')
        domain = "energy_net"
        if domain not in rwrl_defaults_nominal:
            raise ValueError(f"Unknown domain: {domain}. Add to rwrl_defaults_nominal.")
        rwrl_nominal = rwrl_defaults_nominal[domain]
    except:
        raise ValueError('evaluation not supported on this domain')
    return rwrl_nominal

def set_perturb_defaults(import_logs, inputs_dict):
    setup_kwargs = inputs_dict['setup_kwargs']
    perturb_param_min = setup_kwargs['perturb_param_min']
    perturb_param_max = setup_kwargs['perturb_param_max']

    if (perturb_param_min is None) and (perturb_param_max is None):
        try:
            env_kwargs = import_logs[0]['param']['env_kwargs']
            domain_raw = env_kwargs['env_name']
            domain = "energy_net"
            if domain not in rwrl_defaults_eval:
                raise KeyError(domain)
            rwrl_defaults = rwrl_defaults_eval[domain]
            setup_kwargs.update(rwrl_defaults)
        except KeyError:
            raise ValueError(f"Unknown domain '{domain_raw}' — must define defaults in rwrl_defaults_eval")
        except:
            raise ValueError('Must set perturb_param_min, perturb_param_max')

    return inputs_dict

def create_rwrl_kwargs(perturb_param_value, setup_kwargs):
    # Ensure defaults for EnergyNet safety and perturbation constraints
    setup_defaults = {
        'energy_constraints': ['thermal', 'battery'],
        'energy_constraints_all': True
    }
    for k, v in setup_defaults.items():
        setup_kwargs.setdefault(k, v)
    env_setup_kwargs = dict(setup_kwargs)  # shallow copy
    print(f"env_setup_kwargs: {env_setup_kwargs}")
    env_setup_kwargs['perturb_param_value'] = perturb_param_value
    env_setup_kwargs['safety_spec'] = create_safety_spec(env_setup_kwargs)
    env_setup_kwargs['perturb_spec'] = create_perturb_spec(env_setup_kwargs)
    return env_setup_kwargs

def eval_setup(perturb_param_value, setup_kwargs, import_logs):
    rwrl_kwargs = create_rwrl_kwargs(perturb_param_value, setup_kwargs)

    import_objects_all = []
    for import_log in import_logs:
        import_log_param = import_log['param']
        import_log_final = import_log['final']

        env_kwargs = import_log_param['env_kwargs']
        env_kwargs = {k: v for k, v in env_kwargs.items() if k in env_kwargs_keys}

        try:
            env_setup_kwargs = import_log_param['env_setup_kwargs']
            env_setup_kwargs.update(rwrl_kwargs)
        except:
            env_setup_kwargs = rwrl_kwargs

        gamma = import_log_param['alg_kwargs']['gamma']

        actor_kwargs = import_log_param['actor_kwargs']
        actor_kwargs = {k: v for k, v in actor_kwargs.items() if k in actor_kwargs_keys}

        actor_weights = import_log_final['actor_weights']
        actor_kwargs['actor_weights'] = actor_weights

        if setup_kwargs['import_adversary']:
            actor_kwargs['adversary_weights'] = import_log_final['adversary_weights']
        else:
            actor_kwargs['actor_adversary_prob'] = 0.0

        import_rms_stats = import_log_final['rms_stats']

        env, _ = init_env(**env_kwargs, env_setup_kwargs=env_setup_kwargs)
        actor = init_actor(env, **actor_kwargs)

        s_dim = gym.spaces.utils.flatdim(env.observation_space)
        a_dim = gym.spaces.utils.flatdim(env.action_space)

        normalizer = RunningNormalizers(s_dim, a_dim, gamma, import_rms_stats)
        actor.set_rms(normalizer)

        import_objects_all.append({'env': env, 'actor': actor, 'gamma': gamma})

    return import_objects_all

def evaluate(env, actor, gamma=1.00, env_horizon=48, num_traj=150, deterministic=True, seed=0):
    init_seeds(seed, [env])
    J_tot_list, Jc_tot_list, Jc_vec_tot_list = [], [], []
    J_disc_list, Jc_disc_list, Jc_vec_disc_list = [], [], []
    env_horizon = 48
    num_traj = 30
    for _ in range(num_traj):
        _, J_all = trajectory_sampler(env, actor, env_horizon, deterministic=deterministic, gamma=gamma)
        J_tot, Jc_tot, Jc_vec_tot, J_disc, Jc_disc, Jc_vec_disc = J_all
        J_tot_list.append(J_tot)
        Jc_tot_list.append(Jc_tot)
        Jc_vec_tot_list.append(Jc_vec_tot)
        J_disc_list.append(J_disc)
        Jc_disc_list.append(Jc_disc)
        Jc_vec_disc_list.append(Jc_vec_disc)

    return (
        np.mean(J_tot_list),
        np.mean(Jc_tot_list),
        np.mean(Jc_vec_tot_list, axis=0),
        np.mean(J_disc_list),
        np.mean(Jc_disc_list),
        np.mean(Jc_vec_disc_list, axis=0),
    )

def evaluate_list(inputs_dict):
    import_logs = inputs_dict['import_logs']
    perturb_param_value = inputs_dict['perturb_param_value']
    setup_kwargs = inputs_dict['setup_kwargs']
    eval_kwargs = inputs_dict['eval_kwargs']

    init_seeds(eval_kwargs['seed'])
    import_objects_all = eval_setup(perturb_param_value, setup_kwargs, import_logs)

    J_tot_all, Jc_tot_all, Jc_vec_tot_all = [], [], []
    J_disc_all, Jc_disc_all, Jc_vec_disc_all = [], [], []

    for import_objects in import_objects_all:
        env, actor, gamma = import_objects['env'], import_objects['actor'], import_objects['gamma']
        J = evaluate(env, actor, gamma, **eval_kwargs)
        J_tot_all.append(J[0])
        Jc_tot_all.append(J[1])
        Jc_vec_tot_all.append(J[2])
        J_disc_all.append(J[3])
        Jc_disc_all.append(J[4])
        Jc_vec_disc_all.append(J[5])

    return (
        np.array(J_tot_all),
        np.array(Jc_tot_all),
        np.array(Jc_vec_tot_all).T,
        np.array(J_disc_all),
        np.array(Jc_disc_all),
        np.array(Jc_vec_disc_all).T,
    )

# Default Evaluation Settings
rwrl_defaults_eval_cartpole = {'perturb_param_name': 'pole_length', 'perturb_param_min': 0.75, 'perturb_param_max': 1.25}
rwrl_defaults_eval_walker = {'perturb_param_name': 'torso_length', 'perturb_param_min': 0.10, 'perturb_param_max': 0.50}
rwrl_defaults_eval_quadruped = {'perturb_param_name': 'torso_density', 'perturb_param_min': 500, 'perturb_param_max': 1500}
rwrl_defaults_eval_energynet = {'perturb_param_name': 'battery', 'perturb_param_min': 1.7, 'perturb_param_max': 2.0}

rwrl_defaults_eval = {
    'cartpole': rwrl_defaults_eval_cartpole,
    'walker': rwrl_defaults_eval_walker,
    'quadruped': rwrl_defaults_eval_quadruped,
    'energy_net': rwrl_defaults_eval_energynet
}

# Nominal Perturbation Values
rwrl_defaults_nominal_cartpole = {'pole_length': 1.0, 'pole_mass': 0.1, 'joint_damping': 2e-6, 'slider_damping': 5e-4}
rwrl_defaults_nominal_quadruped = {'shin_length': 0.25, 'torso_density': 1000., 'joint_damping': 30., 'contact_friction': 1.5}
rwrl_defaults_nominal_walker = {'thigh_length': 0.225, 'torso_length': 0.3, 'joint_damping': 0.1, 'contact_friction': 0.7}
rwrl_defaults_nominal_humanoid = {'contact_friction': 0.7, 'joint_damping': 0.2, 'head_size': 0.09}
rwrl_defaults_nominal_manipulator = {'lower_arm_length': 0.12, 'root_damping': 2.0, 'shoulder_damping': 1.5}
rwrl_defaults_nominal_energynet = {'battery': 1.0}


rwrl_defaults_nominal = {
    'cartpole': rwrl_defaults_nominal_cartpole,
    'quadruped': rwrl_defaults_nominal_quadruped,
    'walker': rwrl_defaults_nominal_walker,
    'humanoid': rwrl_defaults_nominal_humanoid,
    'manipulator': rwrl_defaults_nominal_manipulator,
    'energy_net': rwrl_defaults_nominal_energynet
}
