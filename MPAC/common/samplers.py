import numpy as np


def trajectory_sampler(env,actor,horizon,deterministic=False,gamma=1.00):
    """Generates single trajectory.
    
    Args:
        env (object): environment
        actor (object): policy
        horizon (int): length of rollout
        deterministic (bool): if True, use deterministic actor
        gamma (float): discount rate
    """
    s_traj = []
    a_traj = []
    r_traj = []
    sp_traj = []
    d_traj = []
    c_traj = []
    r_raw_traj = []
    
    J_tot = 0.0
    Jc_tot = 0.0
    Jc_vec_tot = 0.0
    J_disc = 0.0
    Jc_disc = 0.0
    Jc_vec_disc = 0.0
    gamma_t = 1.0
    
    s = env.reset()
    for t in range(horizon):
        s_old = s
        a = actor.sample(s_old,deterministic=deterministic).detach().cpu().numpy()
        s, r, d, info = env.step(actor.clip(a))
        # Override `done` flag if the only reason it's True is time limit
        c = info.get('cost',np.zeros_like(r))
        reward = info.get('reward',r)
        constraints = info.get('constraints',np.array([1.0]))
        J_tot += reward
        Jc_tot += c
        Jc_vec_tot += (1.0 - constraints)
        J_disc += gamma_t * reward
        Jc_disc += gamma_t * c
        Jc_vec_disc += gamma_t * (1.0 - constraints)
        gamma_t *= gamma

        if t == (horizon-1):
            d = False

        s_traj.append(s_old.observation if hasattr(s_old, 'observation') else s_old) #change
        a_traj.append(a)
        #stav added 
        # if isinstance(r, dict):
        #     r = np.mean(list(r.values()))
        r_traj.append(r)
        sp_traj.append(s)
        d_traj.append((bool(np.asarray(d).squeeze)))
        #stav added 
        # if isinstance(c, dict):
        #     c = np.mean(list(c.values()))
        c_traj.append(c)
        r_raw_traj.append(reward)

        if d:
            break
    s_traj = np.array(s_traj,dtype=np.float32)
    a_traj = np.array(a_traj,dtype=np.float32)
    r_traj = np.array(r_traj,dtype=np.float32)
    sp_traj = np.array(sp_traj,dtype=np.float32)
    d_traj = np.array(d_traj)
    c_traj = np.array(c_traj,dtype=np.float32)
    r_raw_traj = np.array(r_raw_traj,dtype=np.float32)

    data = (s_traj, a_traj, r_traj, sp_traj, d_traj, c_traj, r_raw_traj)
    eval = (J_tot, Jc_tot, Jc_vec_tot, J_disc, Jc_disc, Jc_vec_disc)

    return data, eval