"""Entry point for evaluation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Helps in maintaining consistency across matrix operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128" # Limits the maximum memory block size to improve stability

from datetime import datetime
import pickle
import multiprocessing as mp
import numpy as np
import copy

from MPAC.common.train_utils import gather_inputs
from MPAC.analysis.eval_parser import create_eval_parser, all_eval_kwargs
from MPAC.analysis.eval_utils_energy import show_perturbations
from MPAC.analysis.eval_utils_energy import set_perturb_defaults, evaluate_list

def main():
    """Parses inputs, runs evaluations, saves data."""
    start_time = datetime.now()
    
    parser = create_eval_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args,all_eval_kwargs)

    import_filefull = os.path.join(args.import_path,args.import_file)
    with open(import_filefull,'rb') as f:
        import_logs = pickle.load(f)
    
    if args.show_perturbations:
        perturb_nominal = show_perturbations(import_logs)
        
        print('\n')
        print('Perturbation Names and Nominal Values:')
        print('--------------------------------------')
        print(perturb_nominal)
    else:
        inputs_dict = set_perturb_defaults(import_logs,inputs_dict)
        inputs_dict['import_logs'] = import_logs

        # Create range of perturbation values
        setup_kwargs = inputs_dict['setup_kwargs']
        perturb_param_min = setup_kwargs['perturb_param_min']
        perturb_param_max = setup_kwargs['perturb_param_max']
        perturb_param_count = setup_kwargs['perturb_param_count']
        
        perturb_param_values = np.linspace(
            perturb_param_min,perturb_param_max,perturb_param_count)

        # Create input list and run evaluations
        inputs_list = []
        for value in perturb_param_values:
            inputs_dict['perturb_param_value'] = value
            inputs_list.append(copy.deepcopy(inputs_dict))

        if args.cores is None:
            cpu_count = mp.cpu_count()
            if args.perturb_param_count > cpu_count:
                raise ValueError((
                    "WARNING. Number of test environments (%d) "
                    "exceeds number of CPUs (%d). "
                    "Specify number of parallel processes using --cores. "
                    "CPU and GPU memory should also be considered when "
                    "setting --cores."
                    )%(args.perturb_param_count,cpu_count)
                )
            else:
                args.cores = args.perturb_param_count

        with mp.get_context('spawn').Pool(args.cores) as pool:
            out_list = pool.map(evaluate_list,inputs_list)
        (J_tot_list, Jc_tot_list, Jc_vec_tot_list, J_disc_list, Jc_disc_list, 
            Jc_vec_disc_list) = zip(*out_list)

        J_tot_all = np.moveaxis(np.array(J_tot_list),-1,0)
        Jc_tot_all = np.moveaxis(np.array(Jc_tot_list),-1,0)
        Jc_vec_tot_all = np.moveaxis(np.array(Jc_vec_tot_list),-1,0)
        J_disc_all = np.moveaxis(np.array(J_disc_list),-1,0)
        Jc_disc_all = np.moveaxis(np.array(Jc_disc_list),-1,0)
        Jc_vec_disc_all = np.moveaxis(np.array(Jc_vec_disc_list),-1,0)

        output = {
            'eval': {
                'J_tot':                J_tot_all,
                'Jc_tot':               Jc_tot_all,
                'Jc_vec_tot':           Jc_vec_tot_all,
                'J_disc':               J_disc_all,
                'Jc_disc':              Jc_disc_all,
                'Jc_vec_disc':          Jc_vec_disc_all,
                'perturb_param_values': perturb_param_values,            
            },
            'param': vars(args)
        }

        # Save data
        if args.save_file is None:
            save_file = 'EVAL__%s'%(args.import_file)
        else:
            save_file = 'EVAL_%s__%s'%(args.save_file,args.import_file)

        os.makedirs(args.save_path,exist_ok=True)
        save_filefull = os.path.join(args.save_path,save_file)

        with open(save_filefull,'wb') as f:
            pickle.dump(output,f)
        
        end_time = datetime.now()
        print('Time Elapsed: %s'%(end_time-start_time))

if __name__=='__main__':
    main()