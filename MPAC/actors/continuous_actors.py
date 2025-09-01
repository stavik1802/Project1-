import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MPAC.common.nn_utils import transform_features, SimpleNN, flat_to_list, list_to_flat

class BaseActor:
    """Base policy class."""
    def __init__(self, env):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)
        self.s_rms = None  # Will be set externally

    def set_rms(self, normalizer):
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, _, _, _ = all_rms

    def _transform_state(self, s):
        s_norm = self.s_rms.normalize(s)
        s_feat = transform_features(s_norm)
        return s_feat.to(self.device)

class GaussianActor(BaseActor):
    def __init__(self, env, layers, activations, init_type, gain, layer_norm,
                 std_mult=1.0, per_state_std=True, output_norm=False):
        super(GaussianActor, self).__init__(env)
        
        self.per_state_std = per_state_std
        self.output_norm = output_norm
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        
        # logstd_init
        if self.per_state_std:
            self.logstd_init = np.ones((1,)+env.action_space.shape,dtype='float32') * (np.log(std_mult) - np.log(np.log(2)))
        else:
            self.logstd_init = np.ones((1,)+env.action_space.shape,dtype='float32') * np.log(std_mult)

        # Convert logstd_init to torch tensor
        self.logstd_init_t = torch.tensor(self.logstd_init, dtype=torch.float32, device=self.device)

        # Create main network and target network
        if self.per_state_std:
            self._nn = SimpleNN(self.s_dim, 2*self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_targ = SimpleNN(self.s_dim, 2*self.a_dim, layers, activations, init_type, gain, layer_norm)
            # Copy weights
            self._nn_targ.load_state_dict(self._nn.state_dict())
            
            
            self._nn.to(self.device)
            self._nn_targ.to(self.device)
            
            # trainable are just model parameters
            self.trainable = list(self._nn.parameters())
        else:
            self._nn = SimpleNN(self.s_dim, self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_targ = SimpleNN(self.s_dim, self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_targ.load_state_dict(self._nn.state_dict())
            
            self._nn.to(self.device)
            self._nn_targ.to(self.device)

            # logstd as a parameter
            self.logstd = nn.Parameter(torch.zeros_like(self.logstd_init_t))
            self.logstd_targ = nn.Parameter(torch.zeros_like(self.logstd_init_t), requires_grad=False)

            self.trainable = list(self._nn.parameters()) + [self.logstd]

    def _output_normalization(self, out):
        out_max = torch.mean(torch.abs(out), dim=-1, keepdim=True)
        out_max = torch.maximum(out_max, torch.tensor(1.0, dtype=out.dtype, device=out.device))
        return out / out_max

    def _forward(self, s, targ=False, adversary=False):
        s_feat = self._transform_state(s)
        if targ:
            a_out = self._nn_targ(s_feat)
        else:
            a_out = self._nn(s_feat)

        if self.per_state_std:
            a_mean, a_std_out = torch.split(a_out, self.a_dim, dim=-1)
            a_std = F.softplus(a_std_out)
            a_logstd = torch.log(a_std)
        else:
            a_mean = a_out
            if targ:
                a_logstd = self.logstd_targ.expand_as(a_mean)
            else:
                a_logstd = self.logstd.expand_as(a_mean)

        a_logstd = a_logstd + self.logstd_init_t.to(a_logstd.device)
        a_logstd = torch.maximum(a_logstd, torch.log(torch.tensor(1e-3, dtype=a_logstd.dtype, device=a_logstd.device)))

        if self.output_norm:
            a_mean = self._output_normalization(a_mean)

        return a_mean, a_logstd

    def sample(self, s, deterministic=False, targ=False):
        a_mean, a_logstd = self._forward(s, targ=targ)
        
        if not deterministic:
            u = torch.randn_like(a_mean)
            a = a_mean + torch.exp(a_logstd)*u
        else:
            a = a_mean

        if a.shape[0] == 1:
            a = a.squeeze(0)
        return a

    def clip(self, a):
        return np.clip(a, self.act_low, self.act_high)

    def neglogp(self, s, a, targ=False, adversary=False):
        a_mean, a_logstd = self._forward(s, targ=targ, adversary=adversary)
        neglogp_vec = ( ((a - a_mean)/torch.exp(a_logstd))**2
                        + 2*a_logstd + torch.log(torch.tensor(2*np.pi)) )
        return 0.5 * torch.sum(neglogp_vec, dim=-1)

    def entropy(self, s, adversary=False):
        _, a_logstd = self._forward(s, adversary=adversary)
        ent_vec = 2*a_logstd + torch.log(torch.tensor(2*np.pi)) + 1
        return 0.5 * torch.sum(ent_vec, dim=-1)

    def kl(self, s, kl_info_ref, adversary=False):
        # kl_info_ref: shape [..., 2], last dimension: mean, logstd
        ref_mean = kl_info_ref[...,0]
        ref_logstd = kl_info_ref[...,1]
        ref_mean_t = torch.tensor(ref_mean, dtype=torch.float32, device=self.device)
        ref_logstd_t = torch.tensor(ref_logstd, dtype=torch.float32, device=self.device)

        a_mean, a_logstd = self._forward(s, adversary=adversary)
        num = (a_mean - ref_mean_t)**2 + torch.exp(2*ref_logstd_t)
        kl_vec = num/torch.exp(2*a_logstd) + 2*a_logstd - 2*ref_logstd_t - 1
        return 0.5 * torch.sum(kl_vec, dim=-1)

    def get_kl_info(self, s, adversary=False):
        a_mean, a_logstd = self._forward(s, adversary=adversary)
        # return numpy arrays
        return torch.stack([a_mean, a_logstd], dim=-1).detach().cpu().numpy()

    def kl_targ(self, s, separate=False, per_dim=False, adversary=False):
        a_mean, a_logstd = self._forward(s, adversary=adversary)
        a_mean_targ, a_logstd_targ = self._forward(s, targ=True, adversary=adversary)

        if separate:
            kl_mean_vec = (a_mean - a_mean_targ)**2 / torch.exp(2*a_logstd_targ)
            kl_std_vec = (torch.exp(2*a_logstd_targ)/torch.exp(2*a_logstd)
                          + 2*a_logstd - 2*a_logstd_targ - 1)
            
            if per_dim:
                kl_mean = 0.5 * kl_mean_vec
                kl_std = 0.5 * kl_std_vec
            else:
                kl_mean = 0.5 * torch.sum(kl_mean_vec, dim=-1)
                kl_std = 0.5 * torch.sum(kl_std_vec, dim=-1)
            return kl_mean, kl_std
        else:
            num = (a_mean - a_mean_targ)**2 + torch.exp(2*a_logstd_targ)
            kl_vec = num/torch.exp(2*a_logstd) + 2*a_logstd - 2*a_logstd_targ - 1
            if per_dim:
                kl = 0.5 * kl_vec
            else:
                kl = 0.5 * torch.sum(kl_vec, dim=-1)
            return kl

    def get_weights(self, flat=False):
        # Extract model weights and logstd if needed
        with torch.no_grad():
            if self.per_state_std:
                # just model weights
                weights = [p.detach().cpu().numpy() for p in self._nn.parameters()]
            else:
                weights = [p.detach().cpu().numpy() for p in self._nn.parameters()] + [self.logstd.detach().cpu().numpy()]
        if flat:
            weights = list_to_flat(weights)
        return weights

    def set_weights(self, weights, from_flat=False, increment=False):
        if from_flat:
            weights = flat_to_list(self.trainable, weights)

        if increment:
            cur_w = self.get_weights(flat=False)
            weights = [w+cw for w,cw in zip(weights, cur_w)]

        with torch.no_grad():
            if self.per_state_std:
                # Just load the model weights
                param_list = list(self._nn.parameters())
                for p, w in zip(param_list, weights):
                    p.copy_(torch.tensor(w, device=self.device))
            else:
                # Last is logstd
                model_weights = weights[:-1]
                logstd_weights = weights[-1]
                logstd_weights = np.maximum(logstd_weights, np.log(1e-3))

                param_list = list(self._nn.parameters())
                for p, w in zip(param_list, model_weights):
                    p.copy_(torch.tensor(w, device=self.device))
                self.logstd.copy_(torch.tensor(logstd_weights, device=self.device))

    def update_targ(self, tau):
        # Polyak averaging
        with torch.no_grad():
            for p_targ, p in zip(self._nn_targ.parameters(), self._nn.parameters()):
                p_targ.copy_((1-tau)*p_targ + tau*p)
            
            if not self.per_state_std:
                self.logstd_targ.copy_((1-tau)*self.logstd_targ + tau*self.logstd)
                
    

class GaussianActorwAdversary(GaussianActor):
    def __init__(self, env, layers, activations, init_type, gain, layer_norm,
                 std_mult=1.0, per_state_std=True, output_norm=False, adversary_prob=0.0):
        super(GaussianActorwAdversary, self).__init__(env, layers, activations, init_type, gain, layer_norm,
                                                      std_mult, per_state_std, output_norm)
        self.adversary_prob = adversary_prob

        # Create adversary networks
        if self.per_state_std:
            self._nn_adversary = SimpleNN(self.s_dim, 2*self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_adversary_targ = SimpleNN(self.s_dim, 2*self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_adversary_targ.load_state_dict(self._nn_adversary.state_dict())
            
            self._nn_adversary.to(self.device)
            self._nn_adversary_targ.to(self.device)
            
            
            self.adversary_trainable = list(self._nn_adversary.parameters())
        else:
            self._nn_adversary = SimpleNN(self.s_dim, self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_adversary_targ = SimpleNN(self.s_dim, self.a_dim, layers, activations, init_type, gain, layer_norm)
            self._nn_adversary_targ.load_state_dict(self._nn_adversary.state_dict())
            
            self._nn_adversary.to(self.device)
            self._nn_adversary_targ.to(self.device)
            
            self.logstd_adversary = nn.Parameter(torch.zeros_like(self.logstd_init_t))
            self.logstd_adversary_targ = nn.Parameter(torch.zeros_like(self.logstd_init_t), requires_grad=False)
            self.adversary_trainable = list(self._nn_adversary.parameters()) + [self.logstd_adversary]

    def _forward(self, s, targ=False, adversary=False):
        s_feat = self._transform_state(s)
        if adversary:
            net = self._nn_adversary_targ if targ else self._nn_adversary
        else:
            net = self._nn_targ if targ else self._nn
        
        a_out = net(s_feat)
        if self.per_state_std:
            a_mean, a_std_out = torch.split(a_out, self.a_dim, dim=-1)
            a_std = F.softplus(a_std_out)
            a_logstd = torch.log(a_std)
        else:
            a_mean = a_out
            if adversary:
                a_logstd_base = self.logstd_adversary_targ if targ else self.logstd_adversary
            else:
                a_logstd_base = self.logstd_targ if targ else self.logstd
            a_logstd = a_logstd_base.expand_as(a_mean)

        a_logstd = a_logstd + self.logstd_init_t.to(a_logstd.device)
        a_logstd = torch.maximum(a_logstd, torch.log(torch.tensor(1e-3, dtype=a_logstd.dtype, device=a_logstd.device)))

        if self.output_norm:
            a_mean = self._output_normalization(a_mean)
        return a_mean, a_logstd

    def sample(self, s, deterministic=False, targ=False):
        a_mean, a_logstd = self._forward(s, targ=targ, adversary=False)
        a = a_mean.clone()
        if self.adversary_prob > 0.0:
            a_mean_adv, a_logstd_adv = self._forward(s, targ=targ, adversary=True)
            u_adv = np.random.rand(a.shape[0],1)
            use_adv = torch.tensor((u_adv < self.adversary_prob).astype('float32'), device=self.device)
            use_adv = use_adv.to(a.device)
            a = (1-use_adv)*a + use_adv*a_mean_adv
            a_logstd = (1-use_adv)*a_logstd + use_adv*a_logstd_adv

        if not deterministic:
            u = torch.randn_like(a)
            a = a + torch.exp(a_logstd)*u

        if a.shape[0] == 1:
            a = a.squeeze(0)
        print(f"action: {a}")
        return a

    def sample_separate(self, s, deterministic=False, targ=False, adversary=False):
        a_mean, a_logstd = self._forward(s, targ=targ, adversary=adversary)
        a = a_mean.clone()
        if not deterministic:
            u = torch.randn_like(a)
            a = a + torch.exp(a_logstd)*u

        if a.shape[0] == 1:
            a = a.squeeze(0)
        return a

    def get_adversary_weights(self, flat=False):
        with torch.no_grad():
            if self.per_state_std:
                weights = [p.detach().cpu().numpy() for p in self._nn_adversary.parameters()]
            else:
                weights = [p.detach().cpu().numpy() for p in self._nn_adversary.parameters()] + [self.logstd_adversary.detach().cpu().numpy()]

        if flat:
            weights = list_to_flat(weights)
        return weights

    def set_adversary_weights(self, weights, from_flat=False, increment=False):
        if from_flat:
            weights = flat_to_list(self.adversary_trainable, weights)

        if increment:
            cur_w = self.get_adversary_weights(flat=False)
            weights = [w+cw for w,cw in zip(weights, cur_w)]

        with torch.no_grad():
            if self.per_state_std:
                param_list = list(self._nn_adversary.parameters())
                for p, w in zip(param_list, weights):
                    p.copy_(torch.tensor(w, device=self.device))
            else:
                model_weights = weights[:-1]
                logstd_weights = weights[-1]
                logstd_weights = np.maximum(logstd_weights, np.log(1e-3))
                param_list = list(self._nn_adversary.parameters())
                for p, w in zip(param_list, model_weights):
                    p.copy_(torch.tensor(w, device=self.device))
                self.logstd_adversary.copy_(torch.tensor(logstd_weights, device=self.device))

    def update_adversary_targ(self, tau):
        with torch.no_grad():
            for p_targ, p in zip(self._nn_adversary_targ.parameters(), self._nn_adversary.parameters()):
                p_targ.copy_((1-tau)*p_targ + tau*p)

            if not self.per_state_std:
                self.logstd_adversary_targ.copy_((1-tau)*self.logstd_adversary_targ + tau*self.logstd_adversary)
