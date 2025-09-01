"""Interface to robust perturbation classes."""
from MPAC.robust_methods.ramu import RAMU

def init_rob_net(env,robust_type,
    rob_weights,rob_setup_kwargs,safety_kwargs):
    """Creates robust perturbation class.

    Args:
        env (object): environment
        robust_type (str): robust perturbation type (otp, ramu)
        rob_weights (list): list of OTP NN weights

        rob_setup_kwargs (dict): robustness perturbation setup parameters
        safety_kwargs (dict): safety parameters
    
    Returns:
        Class that implements robust perturbations.
    """
    if robust_type == 'ramu':
        rob_net = RAMU(env,rob_setup_kwargs,safety_kwargs)
    else:
        raise ValueError('invalid perturbation type')

    if rob_weights is not None:
        rob_net.set_weights(rob_weights)

    return rob_net