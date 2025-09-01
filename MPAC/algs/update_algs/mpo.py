import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy as sp

from MPAC.algs.update_algs.base_update_alg import BaseUpdate
from MPAC.common.nn_utils import soft_value

class MPO(BaseUpdate):
    """Algorithm class for MPO actor updates."""

    def __init__(self, actor, critic, cost_critic, rob_net,
                update_kwargs, safety_kwargs):
        """Initializes MPO class."""
        super(MPO, self).__init__(actor, critic, cost_critic, rob_net,
                                update_kwargs, safety_kwargs)
    
    def _setup(self, update_kwargs, safety_kwargs):
        """Sets up hyperparameters as class attributes."""
        # Call parent setup to initialize common parameters
        super(MPO, self)._setup(update_kwargs, safety_kwargs)

        # Set actor learning rate and Adam optimizer for actor parameters
        self.actor_lr = update_kwargs['actor_lr']
        self.actor_optimizer = optim.Adam(self.actor.trainable, lr=self.actor_lr)
        
        # Maximum gradient norm for clipping
        self.max_grad_norm = update_kwargs['max_grad_norm']

        # Initialize temperature parameters
        self.temp_init = update_kwargs['temp_init']
        self.soft_temp_init = soft_value(self.temp_init)
        # Create torch parameter for soft_temp
        self.soft_temp = torch.nn.Parameter(torch.tensor(self.soft_temp_init, dtype=torch.float32, device=self.device))
        
        # Whether to apply action penalties
        self.act_penalty = update_kwargs['act_penalty']
        self.act_temp_init = update_kwargs['act_temp_init']
        self.soft_act_temp_init = soft_value(self.act_temp_init)
        # Create parameter for soft_act_temp if action penalty is used
        self.soft_act_temp = torch.nn.Parameter(torch.tensor(self.soft_act_temp_init, dtype=torch.float32, device=self.device))

        # Whether to solve temperature closed-form or via gradient descent
        self.temp_closed = update_kwargs['temp_closed']
        self.temp_tol = update_kwargs['temp_tol']
        self.temp_bounds = [(1e-6, 1e3)]
        self.temp_opt_type = update_kwargs['temp_opt_type']
        # Set optimization options depending on method
        if self.temp_opt_type == 'SLSQP':
            self.temp_options = {"maxiter": 10}
        elif self.temp_opt_type == 'Powell':
            self.temp_options = None
        else:
            raise ValueError('temp_opt_type not valid')
        
        # Learning rate for temperature updates
        self.temp_lr = update_kwargs['temp_lr']
        self.temp_optimizer = optim.Adam([self.soft_temp] + ([self.soft_act_temp] if self.act_penalty else []), 
                                         lr=self.temp_lr)
        
        # Store references to temperature variables
        self.temp_vars = [self.soft_temp]
        if self.act_penalty:
            self.temp_vars.append(self.soft_act_temp)

        # KL settings (separate mean and std constraints or not)
        self.kl_separate = update_kwargs['kl_separate']
        self.kl_per_dim = update_kwargs['kl_per_dim']
        self.a_dim = self.actor.a_dim
        
        # Set up dual variables for KL constraints
        if self.kl_separate:
            self.dual_mean_init = update_kwargs['dual_mean_init']
            self.soft_dual_mean_init = soft_value(self.dual_mean_init)
            if self.kl_per_dim:
                self.soft_dual_mean_init = np.ones(self.a_dim) * self.soft_dual_mean_init
            # Dual variables for mean KL
            self.soft_dual_mean = torch.nn.Parameter(torch.tensor(self.soft_dual_mean_init, dtype=torch.float32, device=self.device))

            self.dual_std_init = update_kwargs['dual_std_init']
            self.soft_dual_std_init = soft_value(self.dual_std_init)
            if self.kl_per_dim:
                self.soft_dual_std_init = np.ones(self.a_dim) * self.soft_dual_std_init
            # Dual variables for std KL
            self.soft_dual_std = torch.nn.Parameter(torch.tensor(self.soft_dual_std_init, dtype=torch.float32, device=self.device))
            self.dual_vars = [self.soft_dual_mean, self.soft_dual_std]
        else:
            self.dual_init = update_kwargs['dual_init']
            self.soft_dual_init = soft_value(self.dual_init)
            if self.kl_per_dim:
                self.soft_dual_init = np.ones(self.a_dim) * self.soft_dual_init
            self.soft_dual = torch.nn.Parameter(torch.tensor(self.soft_dual_init, dtype=torch.float32, device=self.device))
            self.dual_vars = [self.soft_dual]
        
        # Dual optimizer for KL constraints
        self.dual_lr = update_kwargs['dual_lr']
        self.dual_optimizer = optim.Adam(self.dual_vars, lr=self.dual_lr)

        # Number of actions considered by MPO
        self.mpo_num_actions = update_kwargs['mpo_num_actions']

        # KL and moment constraints
        self.mpo_delta_E = update_kwargs['mpo_delta_E']
        self.mpo_delta_E_penalty = update_kwargs['mpo_delta_E_penalty']
        self.mpo_delta_M = update_kwargs['mpo_delta_M']
        self.mpo_delta_M_mean = update_kwargs['mpo_delta_M_mean']
        self.mpo_delta_M_std = update_kwargs['mpo_delta_M_std']

        # Adjust constraints per dimension if needed
        if self.kl_per_dim:
            self.mpo_delta_M = self.mpo_delta_M / self.a_dim
            self.mpo_delta_M_mean = self.mpo_delta_M_mean / self.a_dim
            self.mpo_delta_M_std = self.mpo_delta_M_std / self.a_dim

        # If using adversary, set up separate variables and optimizers
        if self.use_adversary:
            self.adversary_optimizer = optim.Adam(self.actor.adversary_trainable, lr=self.actor_lr)
            self.soft_temp_adversary = torch.nn.Parameter(torch.tensor(self.soft_temp_init, dtype=torch.float32, device=self.device))
            self.soft_act_temp_adversary = torch.nn.Parameter(torch.tensor(self.soft_act_temp_init, dtype=torch.float32, device=self.device))
            self.temp_adversary_optimizer = optim.Adam([self.soft_temp_adversary] + 
                                                       ([self.soft_act_temp_adversary] if self.act_penalty else []),
                                                       lr=self.temp_lr)

            self.temp_adversary_vars = [self.soft_temp_adversary]
            if self.act_penalty:
                self.temp_adversary_vars.append(self.soft_act_temp_adversary)

            if self.kl_separate:
                self.soft_dual_mean_adversary = torch.nn.Parameter(torch.tensor(self.soft_dual_mean_init, dtype=torch.float32, device=self.device))
                self.soft_dual_std_adversary = torch.nn.Parameter(torch.tensor(self.soft_dual_std_init, dtype=torch.float32, device=self.device))
                self.dual_adversary_vars = [self.soft_dual_mean_adversary, self.soft_dual_std_adversary]
            else:
                self.soft_dual_adversary = torch.nn.Parameter(torch.tensor(self.soft_dual_init, dtype=torch.float32, device=self.device))
                self.dual_adversary_vars = [self.soft_dual_adversary]
            
            self.dual_adversary_optimizer = optim.Adam(self.dual_adversary_vars, lr=self.dual_lr)


    def _get_actor_log(self):
        """Returns stats for logging."""
        # Convert soft_temp and duals to numpy for logging
        log_actor = {
            'mpo_temp':     F.softplus(self.soft_temp).detach().cpu().numpy(),
            'mpo_act_temp': F.softplus(self.soft_act_temp).detach().cpu().numpy(),
        }

        # Log dual variables depending on configuration
        if self.kl_separate:
            log_actor['mpo_dual_mean'] = F.softplus(self.soft_dual_mean).detach().cpu().numpy()
            log_actor['mpo_dual_std'] = F.softplus(self.soft_dual_std).detach().cpu().numpy()
        else:
            log_actor['mpo_dual'] = F.softplus(self.soft_dual).detach().cpu().numpy()
        
        # If adversary is used, also log adversary duals and temps
        if self.use_adversary:
            log_actor['mpo_temp_adversary'] = F.softplus(self.soft_temp_adversary).detach().cpu().numpy()
            log_actor['mpo_act_temp_adversary'] = F.softplus(self.soft_act_temp_adversary).detach().cpu().numpy()

            if self.kl_separate:
                log_actor['mpo_dual_mean_adversary'] = F.softplus(self.soft_dual_mean_adversary).detach().cpu().numpy()
                log_actor['mpo_dual_std_adversary'] = F.softplus(self.soft_dual_std_adversary).detach().cpu().numpy()
            else:
                log_actor['mpo_dual_adversary'] = F.softplus(self.soft_dual_adversary).detach().cpu().numpy()
        
        return log_actor


    def _update_temp_closed(self, Q_batch_active, soft_temp, target_kl):
        """Full solve of temperature variables."""

        def temp_dual(temp):
            # Compute log-sum-exp to solve for temp
            Q_temp_batch = Q_batch_active.detach().cpu().numpy() / temp
            Q_logsumexp_batch = sp.special.logsumexp(Q_temp_batch, axis=0) - np.log(self.mpo_num_actions)
            Q_logsumexp = np.mean(Q_logsumexp_batch)
            temp_loss = temp * (Q_logsumexp + target_kl)
            return temp_loss

        # Current temperature
        temp_cur = F.softplus(soft_temp).detach().cpu().item()
        start = np.array([temp_cur])
        try:
            # Minimize using SciPy
            res = sp.optimize.minimize(temp_dual,
                                        start,
                                        method=self.temp_opt_type,
                                        bounds=self.temp_bounds * len(start),
                                        tol=self.temp_tol,
                                        options=self.temp_options)
            temp_star = res.x[0]
        except:
            print('Error in temperature optimization')
            temp_star = start[0]

        # Update soft_temp parameter
        soft_temp_star = soft_value(temp_star)
        with torch.no_grad():
            soft_temp.copy_(torch.tensor(soft_temp_star, dtype=torch.float32, device=self.device))

    def _calc_temp_loss(self, Q_batch_active, soft_temp, target_kl):
        """Calculates temperature variable loss for gradient-based updates."""
        temp = F.softplus(soft_temp)  # Convert soft param to actual temperature
        Q_temp_batch = Q_batch_active / temp
        # Compute mean log-sum-exp across actions
        Q_logsumexp_batch = torch.logsumexp(Q_temp_batch, dim=0) - torch.log(torch.tensor(self.mpo_num_actions, dtype=torch.float32, device=self.device))
        Q_logsumexp = torch.mean(Q_logsumexp_batch)
        
        # Temperature loss encouraging the right KL constraint
        temp_loss = temp * (Q_logsumexp + target_kl)
        return temp_loss

    def _calc_policy_grad(self, s_batch_flat, a_batch_flat, weights_batch_flat,
                     policy_vars, dual_vars, adversary=False):
        """Calculates policy gradient."""
        # Calculate negative log probabilities
        neglogp_batch_flat = self.actor.neglogp(
            s_batch_flat, a_batch_flat, adversary=adversary)

        # Calculate weighted loss
        pg_loss_batch = weights_batch_flat * neglogp_batch_flat
        pg_loss = torch.mean(pg_loss_batch)
        
        # Calculate KL divergence
        kl_targ_batch = self.actor.kl_targ(s_batch_flat,
            self.kl_separate, self.kl_per_dim, adversary=adversary)
        
        if self.kl_separate:
            soft_dual_mean, soft_dual_std = dual_vars
            dual_mean = F.softplus(soft_dual_mean)
            dual_std = F.softplus(soft_dual_std)
            
            kl_targ_mean_batch, kl_targ_std_batch = kl_targ_batch
            kl_targ_mean = torch.mean(kl_targ_mean_batch, dim=0)
            kl_targ_std = torch.mean(kl_targ_std_batch, dim=0)
            kl_targ = [kl_targ_mean, kl_targ_std]

            pg_loss = (pg_loss 
                + torch.sum(dual_mean * kl_targ_mean)
                + torch.sum(dual_std * kl_targ_std)
            )
        else:
            soft_dual = dual_vars[0]
            dual = F.softplus(soft_dual)
            kl_targ = torch.mean(kl_targ_batch, dim=0)
            pg_loss = pg_loss + torch.sum(dual * kl_targ)

        # Calculate gradients
        pol_grad = torch.autograd.grad(pg_loss, policy_vars, retain_graph=True)

        return pol_grad, kl_targ


    def _calc_dual_grad(self, kl_targ, dual_vars):
        """Calculates dual variable gradients for KL constraints."""
        # Zero gradients
        for p in dual_vars:
            if p.grad is not None:
                p.grad.zero_()

        # Compute dual loss
        if self.kl_separate:
            kl_targ_mean, kl_targ_std = kl_targ
            soft_dual_mean, soft_dual_std = dual_vars
            dual_mean = F.softplus(soft_dual_mean)
            dual_std = F.softplus(soft_dual_std)
            dual_loss_mean = torch.sum(dual_mean * (self.mpo_delta_M_mean - kl_targ_mean))
            dual_loss_std = torch.sum(dual_std * (self.mpo_delta_M_std - kl_targ_std))
            dual_loss = dual_loss_mean + dual_loss_std
        else:
            soft_dual = dual_vars[0]
            dual = F.softplus(soft_dual)
            dual_loss = torch.sum(dual * (self.mpo_delta_M - kl_targ))

        # Backprop to get dual gradients
        dual_loss.backward()
        soft_dual_grad = [p.grad for p in dual_vars]
        return soft_dual_grad

    def _apply_actor_grads(self, s_batch, adversary=False):
        """Performs single actor update step for MPO."""
        # If s_batch is a single state, make it a batch of size 1
        torch.cuda.empty_cache()
             
        if len(np.shape(s_batch)) == 1:
            s_batch = np.expand_dims(s_batch, axis=0)

        # Convert s_batch to torch tensor
        s_batch_t = torch.tensor(s_batch, dtype=torch.float32, device=self.device)

        # If adversary update is required, do that first
        if adversary:
            self._apply_adversary_grads(s_batch_t)

        batch_size = s_batch_t.shape[0]
        # Tile states to create [num_actions * batch_size, state_dim]
        s_batch_flat = s_batch_t.repeat(self.mpo_num_actions, 1)

        # Sample actions according to current or target policy
        if self.use_adversary:
            a_batch_flat = self.actor.sample_separate(s_batch_flat, targ=self.use_targ, adversary=False)
        else:
            a_batch_flat = self.actor.sample(s_batch_flat, targ=self.use_targ)

        # Apply action penalty if enabled
        if self.act_penalty:
            a_diff_batch_flat = a_batch_flat - torch.clamp(a_batch_flat, -1., 1.)
            a_penalty_batch_flat = torch.norm(a_diff_batch_flat, dim=-1) * -1
            a_penalty_batch = a_penalty_batch_flat.view(self.mpo_num_actions, batch_size)

        # Compute Q-values (and cost Q-values if safe)
        if self.use_targ:
            Q_batch_flat = self.critic.value_targ((s_batch_flat, a_batch_flat))
        else:
            Q_batch_flat = self.critic.value((s_batch_flat, a_batch_flat))
            
        # Handle multi-objective case
        if self.multiobj:
            if self.critic.use_dynamic_weights:
                # Compute influence weights for each objective
                weights = self.critic.compute_influence_metrics(s_batch_flat, a_batch_flat)
            else:
                # Use equal weights if dynamic weights are disabled
                num_objectives = Q_batch_flat.shape[-1]  # Get number of objectives from Q-values shape
                weights = torch.ones(num_objectives, device=self.device) / num_objectives
            
            # Apply weights to get composite Q-values
            Q_batch_flat = torch.sum(Q_batch_flat * weights, dim=-1)
            
        Q_batch = Q_batch_flat.view(self.mpo_num_actions, batch_size)
        # If safe RL, incorporate cost critic
        if self.safe:
            if self.use_targ:
                Q_cost_batch_flat = self.cost_critic.value_targ((s_batch_flat, a_batch_flat))
            else:
                Q_cost_batch_flat = self.cost_critic.value((s_batch_flat, a_batch_flat))
            Q_cost_batch = Q_cost_batch_flat.view(self.mpo_num_actions, batch_size)

            if (self.safe_type == 'crpo'):
                cost_ave = torch.mean(Q_cost_batch) * -1
                if (cost_ave.item() > self.safety_budget):
                    Q_batch_active = Q_cost_batch
                else:
                    Q_batch_active = Q_batch
                print("cost average=", cost_ave.item(), "safety budget=", self.safety_budget)
                 

            else:
                # Use Lagrange multiplier for cost constraints
                Q_batch_active = Q_batch + self.safety_lagrange * Q_cost_batch
        else:
            Q_batch_active = Q_batch

        # If temp_closed, solve temperature analytically
        if self.temp_closed:
            self._update_temp_closed(Q_batch_active, self.soft_temp, self.mpo_delta_E)
            if self.act_penalty:
                self._update_temp_closed(a_penalty_batch, self.soft_act_temp, self.mpo_delta_E_penalty)

        # Compute weights from Q-values using current temperatures
        temp = F.softplus(self.soft_temp)
        Q_temp_batch = Q_batch_active / temp
        weights_batch = F.softmax(Q_temp_batch, dim=0) * self.mpo_num_actions
        weights_batch_flat = weights_batch.view(-1)

        if self.act_penalty:
            act_temp = F.softplus(self.soft_act_temp)
            a_penalty_temp_batch = a_penalty_batch / act_temp
            weights_act_batch = F.softmax(a_penalty_temp_batch, dim=0) * self.mpo_num_actions
            weights_act_batch_flat = weights_act_batch.view(-1)
            # Average the two sets of weights
            weights_batch_flat = torch.mean(torch.stack([weights_batch_flat, weights_act_batch_flat]), dim=0)

        # Calculate policy gradient and KL targets
        pol_grad, kl_targ = self._calc_policy_grad(s_batch_flat, a_batch_flat,
                                                weights_batch_flat, self.actor.trainable,
                                                self.dual_vars, adversary=False)

        # If not closed form, update temperature by gradient descent
        if not self.temp_closed:
            self.temp_optimizer.zero_grad()
            
            temp_loss = self._calc_temp_loss(Q_batch_active, self.soft_temp, self.mpo_delta_E)
            
            if self.act_penalty:
                act_temp_loss = self._calc_temp_loss(a_penalty_batch, self.soft_act_temp, self.mpo_delta_E_penalty)
                temp_loss = temp_loss + act_temp_loss
                
            temp_loss.backward()
            self.temp_optimizer.step()

        # Update dual variables for KL constraints
        soft_dual_grad = self._calc_dual_grad(kl_targ, self.dual_vars)
        for grad, var in zip(soft_dual_grad, self.dual_vars):
            if grad is not None:
                var.grad = grad
        self.dual_optimizer.step()
        self.dual_optimizer.zero_grad()

        # Apply policy update
        if self.max_grad_norm is not None:
            # Calculate gradient norm before clipping
            grad_norm_pre = torch.norm(torch.stack([torch.norm(g) for g in pol_grad]))
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([p for p in pol_grad], self.max_grad_norm)
            
            # Calculate gradient norm after clipping
            grad_norm_post = torch.norm(torch.stack([torch.norm(g) for g in pol_grad]))
        else:
            # If no clipping, pre and post norms are the same
            grad_norm_pre = torch.norm(torch.stack([torch.norm(g) for g in pol_grad]))
            grad_norm_post = grad_norm_pre

        # Apply gradients to actor
        for param, grad in zip(self.actor.trainable, pol_grad):
            if grad is not None:
                param.grad = grad
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        # If using safety Lagrange multipliers, update them
        if self.safe and (self.safe_type == 'lagrange'):
            cost_all = Q_cost_batch_flat * -1
            cost_ave = torch.mean(cost_all)
            self._update_safety_lagrange(cost_ave)

        return grad_norm_pre, grad_norm_post

    def _apply_adversary_grads(self, s_batch):
        """Performs single actor adversary update step."""
        # Convert s_batch to tensor
        batch_size = s_batch.shape[0]
        s_batch_flat = s_batch.repeat(self.mpo_num_actions, 1)
        # Sample adversarial actions
        a_batch_flat = self.actor.sample_separate(s_batch_flat, targ=self.use_targ, adversary=True)

        if self.act_penalty:
            a_diff_batch_flat = a_batch_flat - torch.clamp(a_batch_flat, -1., 1.)
            a_penalty_batch_flat = torch.norm(a_diff_batch_flat, dim=-1) * -1
            a_penalty_batch = a_penalty_batch_flat.view(self.mpo_num_actions, batch_size)
        
        # Depending on safety, compute Q-values (reward or cost) and flip sign for adversary
        if self.safe:
            if self.use_targ:
                Q_batch_active_flat = self.cost_critic.value_targ((s_batch_flat, a_batch_flat))
            else:
                Q_batch_active_flat = self.cost_critic.value((s_batch_flat, a_batch_flat))
        else:
            if self.use_targ:
                Q_batch_active_flat = self.critic.value_targ((s_batch_flat, a_batch_flat))
            else:
                Q_batch_active_flat = self.critic.value((s_batch_flat, a_batch_flat))
        
        # Adversary tries to maximize cost or minimize reward, hence * -1
        Q_batch_active_flat = Q_batch_active_flat * -1
        Q_batch_active = Q_batch_active_flat.view(self.mpo_num_actions, batch_size)

        # If closed-form temps, solve them
        if self.temp_closed:
            self._update_temp_closed(Q_batch_active, self.soft_temp_adversary, self.mpo_delta_E)
            if self.act_penalty:
                self._update_temp_closed(a_penalty_batch, self.soft_act_temp_adversary, self.mpo_delta_E_penalty)
        
        temp = F.softplus(self.soft_temp_adversary)
        Q_temp_batch = Q_batch_active / temp
        weights_batch = F.softmax(Q_temp_batch, dim=0) * self.mpo_num_actions
        weights_batch_flat = weights_batch.view(-1)

        if self.act_penalty:
            act_temp = F.softplus(self.soft_act_temp_adversary)
            a_penalty_temp_batch = a_penalty_batch / act_temp
            weights_act_batch = F.softmax(a_penalty_temp_batch, dim=0) * self.mpo_num_actions
            weights_act_batch_flat = weights_act_batch.view(-1)
            # Average weights
            weights_batch_flat = torch.mean(torch.stack([weights_batch_flat, weights_act_batch_flat]), dim=0)

        # Compute policy gradient for adversary
        pol_grad, kl_targ = self._calc_policy_grad(s_batch_flat, a_batch_flat,
                                                   weights_batch_flat, self.actor.adversary_trainable,
                                                   self.dual_adversary_vars, adversary=True)

        # Update adversary temperature if not closed-form
        if not self.temp_closed:
            self.temp_adversary_optimizer.zero_grad()
            temp_loss = self._calc_temp_loss(Q_batch_active, self.soft_temp_adversary, self.mpo_delta_E)
            if self.act_penalty:
                act_temp_loss = self._calc_temp_loss(a_penalty_batch, self.soft_act_temp_adversary, self.mpo_delta_E_penalty)
                temp_loss = temp_loss + act_temp_loss
            temp_loss.backward()
            self.temp_adversary_optimizer.step()

        # Update adversary dual variables
        self.dual_adversary_optimizer.zero_grad()
        soft_dual_grad = self._calc_dual_grad(kl_targ, self.dual_adversary_vars)
        for p, g in zip(self.dual_adversary_vars, soft_dual_grad):
            p.grad = g
        self.dual_adversary_optimizer.step()

        # Clip gradients if needed for adversary
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.adversary_trainable, self.max_grad_norm)

        # Update adversary parameters
        self.adversary_optimizer.step()