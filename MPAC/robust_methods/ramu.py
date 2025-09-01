import numpy as np
import scipy.stats as stats
import torch


from MPAC.robust_methods.base_rob_net import RobustNet

class RAMU(RobustNet):
    """Class for Risk-Averse Model Uncertainty."""

    def __init__(self, env, rob_setup_kwargs, safety_kwargs):
        """Initializes RAMU class.

        Args:
            env (object): environment
            rob_setup_kwargs (dict): robustness perturbation setup parameters
            safety_kwargs (dict): safety parameters
        """
        super(RAMU, self).__init__(env, rob_setup_kwargs, safety_kwargs)

    def _setup(self, rob_setup_kwargs, safety_kwargs):
        """Sets up hyperparameters as class attributes."""
        super(RAMU, self)._setup(rob_setup_kwargs, safety_kwargs)

        self.ramu_critic_samples = rob_setup_kwargs['ramu_critic_samples']

        self.distortion_type = rob_setup_kwargs['distortion_type']
        self.distortion_param = rob_setup_kwargs['distortion_param']

    def sample(self, s, a, sp):
        """Returns perturbed next state samples and perturbation outputs."""
        delta_feat = self._transform_delta(s, sp)

        x_shape = (self.ramu_critic_samples,) + np.shape(delta_feat)
        # Generates random uniform samples in the range [0, 1]
        
        x = np.random.random(size=x_shape).astype('float32')
        
        # The uniform samples u are scaled and shifted to lie in the range [-2 * self.rob_magnitude, 2 * self.rob_magnitude]
        delta_out = torch.tensor(2 * self.rob_magnitude * (2 * x - 1), dtype=torch.float32)

        sp_final = self._calc_sp(s, delta_feat, delta_out)
        sp_final_flat = sp_final.view(-1, self.s_dim)

        return sp_final_flat, None, delta_out, delta_out

    def loss_and_targets(self, disc, rtg_next_all, ctg_next_all, delta_out, delta_out_cost):
        """Returns loss function and critic targets."""
        out_shape = (self.ramu_critic_samples, -1)
        rtg_next_all = rtg_next_all.view(out_shape)
        ctg_next_all = ctg_next_all.view(out_shape)

        rtg_weights = self.get_distortion_probs(rtg_next_all, weights=True, attitude=self.rob_reward_attitude)
        sorted_rtg = torch.sort(rtg_next_all, dim=0).values
        # Convert rtg_weights (NumPy array) to torch
        rtg_weights_torch = torch.tensor(rtg_weights, 
                                        dtype=sorted_rtg.dtype, 
                                        device=sorted_rtg.device)
        rtg_next_all = sorted_rtg * rtg_weights_torch

        if self.safe:
            ctg_weights = self.get_distortion_probs(ctg_next_all, weights=True)
            sorted_ctg = torch.sort(ctg_next_all, dim=0).values
            ctg_weights_torch = torch.tensor(ctg_weights, 
                                        dtype=sorted_ctg.dtype, 
                                        device=sorted_ctg.device)
            ctg_next_all = sorted_ctg * ctg_weights_torch
            
        
        
        # weighted average (or sum) ensures that the target is consistent with the properties of the distortion risk measure
        rtg_next_values = torch.mean(rtg_next_all, dim=0)
        ctg_next_values = torch.mean(ctg_next_all, dim=0)

        return 0.0, rtg_next_values, ctg_next_values

    def apply_updates(self,rob_loss):
        """Applies gradient updates."""
        pass

    def _get_quantile_probs(self, num_samples, neutral=False):
        """Returns sorted distortion probabilities of quantiles."""
        quantiles = np.linspace(0, 1 ,num_samples + 1)

        # compute g(x)
        if (self.distortion_type == 'expectation') or neutral:
            g_quantiles = quantiles
        elif self.distortion_type == 'cvar':
            assert 0 < self.distortion_param <= 1.0, 'invalid distortion_param'
            g_quantiles = np.minimum(quantiles / self.distortion_param,1.0)
        elif self.distortion_type == 'power':
            assert 0 < self.distortion_param <= 1.0, 'invalid distortion_param'
            g_quantiles = np.power(quantiles,self.distortion_param)
        elif self.distortion_type == 'dual_power':
            assert self.distortion_param >= 1.0, 'invalid distortion_param'
            g_quantiles = 1 - np.power(1-quantiles,self.distortion_param)
        elif self.distortion_type == 'wang':
            assert self.distortion_param >= 0.0, 'invalid distortion_param'
            g_quantiles = stats.norm.cdf(
                stats.norm.ppf(quantiles) + self.distortion_param)
        else:
            raise ValueError('invalid distortion_type')
        
        # Computes the weights by taking differences between adjacent values 
        quantile_probs = g_quantiles[1:] - g_quantiles[:-1]

        return quantile_probs.astype('float32')

    def get_distortion_probs(self,values,weights=False,attitude='robust'):
        """Returns distortion probabilities for each critic value."""

        if attitude == 'neutral':
            neutral = True
        else:
            neutral = False

        num_samples = len(values)
        quantile_probs = self._get_quantile_probs(num_samples,neutral)

        if attitude == 'optimistic':
            # Emphasizing high-value outcomes instead of low ones
            quantile_probs = quantile_probs[::-1]
        
        distortion_probs = np.expand_dims(quantile_probs, axis=-1)

        if weights:
            return distortion_probs * num_samples
        else:
            return distortion_probs

    def get_weights(self):
        """Returns parameter weights."""
        return None

    def set_weights(self,weights):
        """Sets parameter weights."""
        pass