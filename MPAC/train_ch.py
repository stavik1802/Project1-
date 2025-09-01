"""Entry point for RL training."""
import os

# Set deterministic behavior for PyTorch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datetime import datetime
import pickle
import multiprocessing as mp
import numpy as np
import copy

from MPAC.envs import init_env
from MPAC.actors import init_actor
from MPAC.critics import init_critics
from MPAC.robust_methods import init_rob_net
from MPAC.algs import init_alg
from MPAC.common.seeding import init_seeds
from MPAC.common.train_parser import create_train_parser, all_kwargs
from MPAC.common.train_utils import gather_inputs, set_default_inputs
from MPAC.common.train_utils import import_inputs

def train(inputs_dict):
    """Training on given seed."""

    idx = inputs_dict['setup_kwargs']['idx']
    setup_seed = inputs_dict['setup_kwargs']['setup_seed']
    sim_seed = inputs_dict['setup_kwargs']['sim_seed']
    eval_seed = inputs_dict['setup_kwargs']['eval_seed']

    env_kwargs = inputs_dict['env_kwargs']
    env_setup_kwargs = inputs_dict['env_setup_kwargs']
    safety_kwargs = inputs_dict['safety_kwargs']
    actor_kwargs = inputs_dict['actor_kwargs']
    critic_kwargs = inputs_dict['critic_kwargs']
    rob_kwargs = inputs_dict['rob_kwargs']
    rob_setup_kwargs = inputs_dict['rob_setup_kwargs']
    alg_kwargs = inputs_dict['alg_kwargs']
    rl_update_kwargs = inputs_dict['rl_update_kwargs']

    total_timesteps = inputs_dict['alg_kwargs']['total_timesteps']
    multiobj_enable = env_setup_kwargs.get('multiobj_enable', None)
    init_seeds(setup_seed)
    env, env_eval = init_env(**env_kwargs,env_setup_kwargs=env_setup_kwargs,agent_type=inputs_dict["setup_kwargs"]["agent_type"] )
    actor = init_actor(env,**actor_kwargs)
    critic, cost_critic = init_critics(env, multiobj_enable=multiobj_enable, num_objectives=env.r_dim, **critic_kwargs)
    rob_net = init_rob_net(env,**rob_kwargs,
        rob_setup_kwargs=rob_setup_kwargs,safety_kwargs=safety_kwargs)

    init_seeds(eval_seed,[env_eval])
    init_seeds(sim_seed,[env])
    alg = init_alg(idx,env,env_eval,actor,critic,cost_critic,rob_net,
        alg_kwargs,safety_kwargs,rl_update_kwargs)

    log_name = alg.train(total_timesteps, inputs_dict)

    return log_name

def main():
    """Parses inputs, runs simulations, saves data."""
    start_time = datetime.now()

    parser = create_train_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args,all_kwargs)
    inputs_dict = set_default_inputs(inputs_dict)

    # Checkpoint file
    checkpoint_file_base = inputs_dict['alg_kwargs']['checkpoint_file']
    checkpoint_date = datetime.today().strftime('%m%d%y_%H%M%S')
    checkpoint_file = '%s_%s'%(checkpoint_file_base,checkpoint_date)
    inputs_dict['alg_kwargs']['checkpoint_file'] = checkpoint_file

    # Seeds
    seeds = np.random.SeedSequence(args.seed).generate_state(3)
    setup_seeds = np.random.SeedSequence(seeds[0]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    sim_seeds = np.random.SeedSequence(seeds[1]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    eval_seeds = np.random.SeedSequence(seeds[2]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]

    inputs_list = []
    for run in range(args.runs):
        inputs_dict['setup_kwargs']['idx'] = run + args.runs_start
        if args.setup_seed is None:
            inputs_dict['setup_kwargs']['setup_seed'] = int(setup_seeds[run])
        if args.sim_seed is None:
            inputs_dict['setup_kwargs']['sim_seed'] = int(sim_seeds[run])
        if args.eval_seed is None:
            inputs_dict['setup_kwargs']['eval_seed'] = int(eval_seeds[run])

        inputs_dict = import_inputs(inputs_dict)

        inputs_list.append(copy.deepcopy(inputs_dict))

    if args.cores is None:
        cpu_count = mp.cpu_count()
        if args.runs > cpu_count:
            raise ValueError((
                "WARNING. Number of runs (%d) "
                "exceeds number of CPUs (%d). "
                "Specify number of parallel processes using --cores. "
                "CPU and GPU memory should also be considered when "
                "setting --cores."
                )%(args.runs,cpu_count)
            )
        else:
            args.cores = args.runs

    # Train
    with mp.get_context('spawn').Pool(args.cores) as pool:
        log_names = pool.map(train,inputs_list)

    # Aggregate results
    outputs = []
    for log_name in log_names:
        os.makedirs(args.save_path,exist_ok=True)
        filename = os.path.join(args.save_path,log_name)

        with open(filename,'rb') as f:
            log_data = pickle.load(f)

        outputs.append(log_data)

    # Save data
    save_env_type = args.env_type.lower()
    save_env = args.env_name.split('-')[0].lower()
    if args.task_name is not None:
        save_env = '%s_%s'%(save_env,args.task_name.lower())
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    if args.save_file is None:
        save_file = '%s_%s_%s_%s'%(save_env_type,save_env,
            args.rl_alg,save_date)
    else:
        save_file = '%s_%s_%s_%s_%s'%(save_env_type,save_env,
            args.rl_alg,args.save_file,save_date)

    os.makedirs(args.save_path,exist_ok=True)
    save_filefull = os.path.join(args.save_path,save_file)

    with open(save_filefull,'wb') as f:
        pickle.dump(outputs,f)

    for log_name in log_names:
        filename = os.path.join(args.save_path,log_name)
        os.remove(filename)

    end_time = datetime.now()
    print('Time Elapsed: %s'%(end_time-start_time))

if __name__=='__main__':
    main()