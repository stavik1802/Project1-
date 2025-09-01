import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from MPAC.common.nn_utils import soft_value
from MPAC.critics.critics import MultiPerspectiveCritic
class BaseUpdate:
    """Base class for RL updates."""

    def __init__(self, actor, critic, cost_critic, rob_net, update_kwargs, safety_kwargs):
        """Initializes RL update class.
        
        Args:
            actor (object): policy
            critic (object): value function
            cost_critic (object): cost value function
            rob_net (object): robustness perturbation network
            update_kwargs (dict): update algorithm parameters
            safety_kwargs (dict): safety parameters
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = actor
        self.critic = critic
        self.cost_critic = cost_critic
        self.rob_net = rob_net
        
        self._setup(update_kwargs, safety_kwargs)

    def _setup(self, update_kwargs, safety_kwargs):
        """Sets up hyperparameters as class attributes."""

        # Extract critic learning rate
        self.critic_lr = update_kwargs['critic_lr']

        # Initialize PyTorch optimizers for the critics 
        self.critic_optimizer = optim.Adam(self.critic.trainable, lr=self.critic_lr)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.trainable, lr=self.critic_lr)

        # Use target networks for actor and critics or not
        self.use_targ = update_kwargs['use_targ']
        self.tau = update_kwargs['tau']

        # Safe RL parameters
        self.safe = safety_kwargs['safe']
        self.safe_type = safety_kwargs['safe_type']
        self.gamma = safety_kwargs['gamma']
        self.safety_budget_tot = safety_kwargs['safety_budget']
        self.env_horizon = safety_kwargs['env_horizon']

        # Compute effective safety_budget depending on gamma
        if self.gamma < 1.0:
            self.safety_budget = (self.safety_budget_tot / self.env_horizon) / (1 - self.gamma)
        else:
            self.safety_budget = self.safety_budget_tot

        # Initialize the safety Lagrange multiplier
        self.safety_lagrange_init = safety_kwargs['safety_lagrange_init']
        self.soft_safety_lagrange_init = soft_value(self.safety_lagrange_init)

        # Convert to PyTorch parameter for optimization
        self.soft_safety_lagrange = nn.Parameter(torch.tensor(self.soft_safety_lagrange_init, dtype=torch.float32))
        self.safety_lagrange = F.softplus(self.soft_safety_lagrange)

        # Safety Lagrange optimizer
        self.safety_lagrange_lr = safety_kwargs['safety_lagrange_lr']
        self.safety_lagrange_optimizer = optim.Adam([self.soft_safety_lagrange], lr=self.safety_lagrange_lr)

        # Adversary parameters
        self.adversary_freq = update_kwargs['actor_adversary_freq']
        self.use_adversary = (self.adversary_freq > 0)
        self.adversary_counter = 0

        # Batch sizes and update frequency
        self.update_batch_size = update_kwargs['update_batch_size']
        self.updates_per_step = update_kwargs['updates_per_step']
        self.eval_batch_size = update_kwargs['eval_batch_size']

        # Robustness flag
        self.robust = update_kwargs['robust']

        # Multi-Objective flag
        self.multiobj = isinstance(self.critic, MultiPerspectiveCritic)

    def _update_safety_lagrange(self, cost_ave):
        """
        Updates safety Lagrange variable.
        cost_ave: average cost to compare against safety budget
        """
        # Zero grad in PyTorch
        self.safety_lagrange_optimizer.zero_grad()

        # Calculate current lagrange multiplier via softplus
        safety_lagrange = F.softplus(self.soft_safety_lagrange)

        # Compute loss for safety Lagrange update: -lambda * (cost_ave - safety_budget)
        safety_lagrange_loss = - safety_lagrange * (cost_ave - self.safety_budget)

        # Backprop
        safety_lagrange_loss.backward()
        # Apply gradients
        self.safety_lagrange_optimizer.step()

        # Update cached value
        self.safety_lagrange = F.softplus(self.soft_safety_lagrange)


    def _apply_actor_grads(self, s_batch, adversary=False):
        """Performs single actor update.

        Args:
            s_batch (np.ndarray): states
            adversary (bool): if True, update actor and adversary
        """
        raise NotImplementedError

    def _get_actor_log(self):
        """Returns stats for logging."""
        raise NotImplementedError

    def _apply_rob_grads(self, s_batch, a_batch, sp_batch, disc_batch):
        """Performs single robustness update and returns critic next values."""
        
        # Convert inputs to torch tensors if needed
        s_batch_t = torch.tensor(s_batch, dtype=torch.float32, device=self.device)
        a_batch_t = torch.tensor(a_batch, dtype=torch.float32, device=self.device)
        sp_batch_t = torch.tensor(sp_batch, dtype=torch.float32, device=self.device)
        disc_batch_t = torch.tensor(disc_batch, dtype=torch.float32, device=self.device)

        # Sample robust perturbed next states
        sp_final, sp_final_cost, delta_out, delta_out_cost = self.rob_net.sample(s_batch_t, a_batch_t, sp_batch_t)

        # Get next values from critics given these perturbed states
        rtg_next_all, ctg_next_all = self._get_critic_next_values(sp_final, sp_final_cost)

                
        if self.multiobj:
            # Handle each objective separately
            num_objectives = rtg_next_all.shape[-1]
            rtg_next_values_list = []
    
            # Process each objective
            for i in range(num_objectives):
                # Extract values for current objective
                rtg_next_i = rtg_next_all[..., i]  # Shape: [1280]
                
                # For cost critic, only use actual cost values for first objective
                ctg_next_i = ctg_next_all if i == 0 else torch.zeros_like(rtg_next_i)
                # Apply risk measure to this objective
                _, rtg_next_values_i, ctg_next_values_i  = self.rob_net.loss_and_targets(
                    disc_batch_t, 
                    rtg_next_i, 
                    ctg_next_i,  # Dummy ctg since we handle one objective
                    delta_out,
                    delta_out_cost
                )
                rtg_next_values_list.append(rtg_next_values_i)
                if i==0:
                    ctg_next_values = ctg_next_values_i
            # Combine results back into tensor
            rtg_next_values = torch.stack(rtg_next_values_list, dim=-1)
        
        else:
            # Compute robust loss and adjusted next values
            rob_loss, rtg_next_values, ctg_next_values = self.rob_net.loss_and_targets(disc_batch_t, rtg_next_all, ctg_next_all, delta_out, delta_out_cost)


        return rtg_next_values, ctg_next_values

    def _get_critic_next_values(self, sp_active, sp_cost_active=None):
        """
        Calculates critic next values for flattened next state batch.

        If safe RL is used and sp_cost_active is provided, we use that for cost critic.
        Otherwise, just sp_active.
        """

        # Sample next actions from the actor (target policy if use_targ is True)
        ap_active = self.actor.sample(sp_active, targ=self.use_targ)

        # Get Q-value predictions from critic for next states
        rtg_next_values = self.critic.value_targ((sp_active, ap_active))
    
        # If safe RL is enabled, compute cost critic next values
        if self.safe:
            if sp_cost_active is not None:
                ap_cost_active = self.actor.sample(sp_cost_active, targ=self.use_targ)
                ctg_next_values = self.cost_critic.value_targ((sp_cost_active, ap_cost_active))
            else:
                ctg_next_values = self.cost_critic.value_targ((sp_active, ap_active))
                
                
                
        else:
            # If not safe, cost-to-go is zero
            ctg_next_values = torch.zeros(rtg_next_values.shape[0])

        return rtg_next_values, ctg_next_values

    def _get_critic_targets(self, r_active, c_active, disc_active, rtg_next_values, ctg_next_values):
        """
        Calculates target values for critic loss.
        r_active, c_active: immediate rewards and costs
        disc_active: discount factor batch
        rtg_next_values: next Q-values for reward critic
        ctg_next_values: next Q-values for cost critic
        """
        # Convert inputs to tensors if not already
        r_active_t = torch.tensor(r_active, dtype=torch.float32, device=self.device) if not torch.is_tensor(r_active) else r_active
        c_active_t = torch.tensor(c_active, dtype=torch.float32, device=self.device) if not torch.is_tensor(c_active) else c_active
        disc_active_t = torch.tensor(disc_active, dtype=torch.float32, device=self.device) if not torch.is_tensor(disc_active) else disc_active
        # If disc_active is [batch_size, 1], reshape to [batch_size]
        if len(disc_active_t.shape) > 1:
            disc_active_tn = disc_active_t.squeeze(-1)
        else:
            disc_active_tn = disc_active_t
    
        # Now expand discount factor to match number of objectives
        # From [batch_size] to [batch_size, num_objectives]
        disc_active_expanded = disc_active_tn.unsqueeze(-1).expand(-1, rtg_next_values.shape[-1])

        
        
        # Compute Q-value targets for rewards
        rtg_active = r_active_t + disc_active_expanded * rtg_next_values
        
        if self.safe:
            # If safe RL, compute Q-value targets for costs
            ctg_active = c_active_t + disc_active_t * ctg_next_values
        else:
            # Otherwise, cost Q-value targets are zero
            ctg_active = torch.zeros_like(rtg_active)

        return rtg_active, ctg_active

    def _apply_critic_grads(self, s_active, a_active, rtg_active, critic, critic_optimizer, reward_flag=True):
        """
        Applies critic gradients.
        Handles both single-objective and multi-objective cases, as well as reward and cost critics.
        
        Args:
            s_active, a_active: batches of states and actions
            rtg_active: target Q-values (single value or vector for multi-objective)
            critic: the critic network
            critic_optimizer: the PyTorch optimizer for the critic
            reward_flag: if True, this is reward critic, if False, cost critic
        """
        # Convert inputs to tensors if needed
        rtg_active = rtg_active.to(self.device)
        inputs_active = (s_active, a_active)

        # Zero gradients
        critic_optimizer.zero_grad()
        
        # Clear cache before major operations
        torch.cuda.empty_cache()

        critic_loss = critic.get_loss(inputs_active, rtg_active)

        
        if not critic.use_cagrad:
            critic_loss.backward(retain_graph=reward_flag)
        
        # Apply gradients
        critic_optimizer.step()

        # Update target networks
        critic.update_targs(self.tau)
        
        # Clear cache after update
        torch.cuda.empty_cache()

    def update_actor_critic(self, buffer, steps_new):
        """
        Updates actor and critic networks using the collected data from buffer.

        buffer: replay buffer object with get_offpolicy_info method
        steps_new: number of new steps collected since last update
        """
        # Get evaluation batch (for logging)
        rollout_data_eval = buffer.get_offpolicy_info(batch_size=self.eval_batch_size)
        s_eval, a_eval, sp_eval, disc_eval, r_eval, c_eval = rollout_data_eval

        # Get reference KL info to measure policy change
        kl_info_ref = self.actor.get_kl_info(s_eval)
        if self.use_adversary:
            kl_info_ref_adversary = self.actor.get_kl_info(s_eval, adversary=True)

        # Determine how many updates to perform
        num_updates = int(steps_new * self.updates_per_step)

        grad_norm_pre_all = 0.0
        grad_norm_post_all = 0.0

        # Perform multiple gradient update steps
        for i in range(num_updates):
            rollout_data = buffer.get_offpolicy_info(batch_size=self.update_batch_size)
            s_batch, a_batch, sp_batch, disc_batch, r_batch, c_batch = rollout_data
                
            if self.robust:
                # If using robustness, update robust net and get next values
                rtg_next_values, ctg_next_values = self._apply_rob_grads(s_batch, a_batch, sp_batch, disc_batch)
            else:
                # Otherwise, just compute next values from target critics
                rtg_next_values, ctg_next_values = self._get_critic_next_values(torch.tensor(sp_batch, dtype=torch.float32, device=self.device))
                        # After rtg_next_values, ctg_next_values are defined

            # Compute targets for critics
            rtg_batch, ctg_batch = self._get_critic_targets(r_batch, c_batch, disc_batch, rtg_next_values, ctg_next_values)

            # Update reward critic
            self._apply_critic_grads(s_batch, a_batch, rtg_batch, self.critic, self.critic_optimizer)

            # If safe RL, update cost critic
            if self.safe:
                self._apply_critic_grads(s_batch, a_batch, ctg_batch, self.cost_critic, self.cost_critic_optimizer, reward_flag=False)

            # Update actor (and adversary if needed)
            if self.use_adversary and (self.adversary_counter == 0):
                grad_norm_pre, grad_norm_post = self._apply_actor_grads(s_batch, adversary=True)
                self.actor.update_adversary_targ(self.tau)
            else:
                grad_norm_pre, grad_norm_post = self._apply_actor_grads(s_batch)

            # Update actor target network
            self.actor.update_targ(self.tau)

            # Handle adversary update frequency
            if self.use_adversary:
                self.adversary_counter += 1
                self.adversary_counter = self.adversary_counter % self.adversary_freq

            grad_norm_pre_all += grad_norm_pre
            grad_norm_post_all += grad_norm_post

        # After updates, compute averages
        grad_norm_pre_ave = grad_norm_pre_all / num_updates
        grad_norm_post_ave = grad_norm_post_all / num_updates

        # Compute various actor metrics for logging
        ent = torch.mean(self.actor.entropy(torch.tensor(s_eval, dtype=torch.float32)))
        kl = torch.mean(self.actor.kl(torch.tensor(s_eval, dtype=torch.float32), kl_info_ref))
        kl_targ_all = self.actor.kl_targ(torch.tensor(s_eval, dtype=torch.float32), separate=False, per_dim=False)
        kl_targ = torch.mean(kl_targ_all)

        kl_targ_mean_all, kl_targ_std_all = self.actor.kl_targ(torch.tensor(s_eval, dtype=torch.float32), separate=True, per_dim=False)
        kl_targ_mean = torch.mean(kl_targ_mean_all)
        kl_targ_std = torch.mean(kl_targ_std_all)

        # Log actor stats
        log_actor = {
            'ent':                  ent.item(),
            'kl':                   kl.item(),
            'kl_targ':              kl_targ.item(),
            'kl_targ_mean':         kl_targ_mean.item(),
            'kl_targ_std':          kl_targ_std.item(),
            'actor_grad_norm_pre':  grad_norm_pre_ave,
            'actor_grad_norm':      grad_norm_post_ave,
            'safety_lagrange':      self.safety_lagrange.item(),
            'safety_budget':        self.safety_budget,
        }
        log_actor_alg = self._get_actor_log()
        log_actor.update(log_actor_alg)

        # If adversary is used, compute adversary metrics
        if self.use_adversary:
            s_eval_t2 = torch.tensor(s_eval, dtype=torch.float32, device=self.device)
            ent_adversary = torch.mean(self.actor.entropy(s_eval_t2, adversary=True))
            kl_adversary = torch.mean(self.actor.kl(s_eval_t2, kl_info_ref_adversary, adversary=True))
            kl_targ_all_adversary = self.actor.kl_targ(s_eval_t2, separate=False, per_dim=False, adversary=True)
            kl_targ_adversary = torch.mean(kl_targ_all_adversary)

            kl_targ_mean_all_adversary, kl_targ_std_all_adversary = self.actor.kl_targ(
                s_eval_t2, separate=True, per_dim=False, adversary=True)
            kl_targ_mean_adversary = torch.mean(kl_targ_mean_all_adversary)
            kl_targ_std_adversary = torch.mean(kl_targ_std_all_adversary)

            log_adversary = {
                'ent_adversary':            ent_adversary.item(),
                'kl_adversary':             kl_adversary.item(),
                'kl_targ_adversary':        kl_targ_adversary.item(),
                'kl_targ_mean_adversary':   kl_targ_mean_adversary.item(),
                'kl_targ_std_adversary':    kl_targ_std_adversary.item(),
            }
            log_actor.update(log_adversary)

        # Evaluate critics on evaluation data in mini-batches
        idx = np.arange(self.eval_batch_size)
        sections = np.arange(0, self.eval_batch_size, self.update_batch_size)[1:]
        batches = np.array_split(idx, sections)

        critic_loss_all = 0.0
        cost_critic_loss_all = 0.0
        if self.multiobj:
            per_obj_loss_all = [0.0] * self.critic.num_objectives
        rob_mag_all = 0.0
        rob_cost_mag_all = 0.0

        # Loop over evaluation sub-batches
        for batch_idx in batches:
            s_eval_active = s_eval[batch_idx]
            a_eval_active = a_eval[batch_idx]
            sp_eval_active = sp_eval[batch_idx]
            disc_eval_active = disc_eval[batch_idx]
            r_eval_active = r_eval[batch_idx]
            c_eval_active = c_eval[batch_idx]

            # Convert to tensors
            s_eval_active_t = torch.tensor(s_eval_active, dtype=torch.float32, device=self.device)
            a_eval_active_t = torch.tensor(a_eval_active, dtype=torch.float32, device=self.device)
            sp_eval_active_t = torch.tensor(sp_eval_active, dtype=torch.float32, device=self.device)
            disc_eval_active_t = torch.tensor(disc_eval_active, dtype=torch.float32, device=self.device)
            r_eval_active_t = torch.tensor(r_eval_active, dtype=torch.float32, device=self.device)
            c_eval_active_t = torch.tensor(c_eval_active, dtype=torch.float32, device=self.device)

            if self.robust:
                # If robust, sample robust next states
                sp_final_eval, sp_final_cost_eval, delta_out_eval, delta_out_cost_eval = self.rob_net.sample(
                    s_eval_active_t, a_eval_active_t, sp_eval_active_t
                )

                next_values_all = self._get_critic_next_values(sp_final_eval, sp_final_cost_eval)
                rtg_next_eval_all, ctg_next_eval_all = next_values_all
                
                if self.multiobj:
                    # Handle each objective separately
                    num_objectives = rtg_next_eval_all.shape[-1]
                    rtg_next_values_list = []
            
                    # Process each objective
                    for i in range(num_objectives):
                        # Extract values for current objective
                        rtg_next_i = rtg_next_eval_all[..., i]  # Shape: [1280]
                        
                        # For cost critic, only use actual cost values for first objective
                        ctg_next_i = ctg_next_eval_all if i == 0 else torch.zeros_like(rtg_next_i)
                        # Apply risk measure to this objective
                        _, rtg_next_values_i, ctg_next_values_i  = self.rob_net.loss_and_targets(
                            disc_eval_active_t, 
                            rtg_next_i, 
                            ctg_next_i,  # Dummy ctg since we handle one objective
                            delta_out_eval,
                            delta_out_cost_eval
                        )
                        rtg_next_values_list.append(rtg_next_values_i)
                        if i==0:
                            ctg_next_eval_active = ctg_next_values_i
                    # Combine results back into tensor
                    rtg_next_eval_active = torch.stack(rtg_next_values_list, dim=-1)
                
                else:
                    # Compute robust loss and adjusted next values
                    _, rtg_next_eval_active, ctg_next_eval_active = self.rob_net.loss_and_targets(
                    disc_eval_active_t, rtg_next_eval_all, ctg_next_eval_all, delta_out_eval, delta_out_cost_eval
                )

                # Compute robustness magnitudes
                rob_mag_eval = self.rob_net.get_rob_magnitude(delta_out_eval)
                rob_cost_mag_eval = self.rob_net.get_rob_magnitude(delta_out_cost_eval)

                rob_mag_all += torch.mean(rob_mag_eval)
                rob_cost_mag_all += torch.mean(rob_cost_mag_eval)
            else:
                next_values_active = self._get_critic_next_values(sp_eval_active_t)
                rtg_next_eval_active, ctg_next_eval_active = next_values_active

            # Compute critic targets for evaluation batch
            rtg_eval_active, ctg_eval_active = self._get_critic_targets(
                r_eval_active_t, c_eval_active_t, disc_eval_active_t,
                rtg_next_eval_active, ctg_next_eval_active
            )

            inputs_eval_active = (s_eval_active_t, a_eval_active_t)
            
            if self.multiobj:
                # Get both combined loss and per-objective losses
                critic_loss_active = self.critic.get_loss(inputs_eval_active, rtg_eval_active)
                critic_loss_all += critic_loss_active[0].item()
                # Track individual losses
                for i, loss in enumerate(critic_loss_active):
                    per_obj_loss_all[i] += loss.item()
            else:
                critic_loss_active = self.critic.get_loss(inputs_eval_active, rtg_eval_active)
                critic_loss_all += critic_loss_active.item()

            if self.safe:
                cost_critic_loss_active = self.cost_critic.get_loss(inputs_eval_active, ctg_eval_active)
                cost_critic_loss_all += cost_critic_loss_active.item()

        # Average critic losses over evaluation batches
        critic_loss = critic_loss_all / len(batches)
        cost_critic_loss = cost_critic_loss_all / len(batches)

        log_critic = {
            'critic_loss': critic_loss,
        }
        if self.multiobj:
            # Add per-objective losses to logging
            for i in range(self.critic.num_objectives):
                log_critic[f'objective_{i}_loss'] = per_obj_loss_all[i] / len(batches)
            
            # Add influence weights to logging
            weights = self.critic.influence_weights.detach().cpu().numpy()
            for i in range(self.critic.num_objectives):
                log_critic[f'objective_{i}_weight'] = weights[i]
                
        if self.safe:
            log_cost_critic = {
                'critic_loss': cost_critic_loss,
            }
        else:
            log_cost_critic = None
        # If robust, log robustness stats
        if self.robust:
            rob_magnitude = (rob_mag_all / len(batches)).item()
            rob_cost_magnitude = (rob_cost_mag_all / len(batches)).item()

            
            log_rob = {
                'rob_magnitude': rob_magnitude,
                'rob_cost_magnitude': rob_cost_magnitude,
            }
            log_critic.update(log_rob)

        return log_actor, log_critic, log_cost_critic