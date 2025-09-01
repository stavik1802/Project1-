import gym
import torch


from MPAC.common.nn_utils import transform_features, SimpleNN


class BaseCritic:
    def __init__(self, env):
        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        self.s_rms = None
        self.ret_rms = None
        self.c_ret_rms = None

    def set_rms(self, normalizer):
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, _, self.ret_rms, self.c_ret_rms = all_rms

class QCritic(BaseCritic):
    def __init__(self, env, layers, activations, init_type, gain, layer_norm, safety=False):
        super(QCritic, self).__init__(env)
        self.safety = safety
        in_dim = self.s_dim + self.a_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #added cpu
        #stav added 
        # self.device = "cpu"
        self._nn = SimpleNN(in_dim, 1, layers, activations, init_type, gain, layer_norm)
        self._nn_targ = SimpleNN(in_dim, 1, layers, activations, init_type, gain, layer_norm)
        self._nn_targ.load_state_dict(self._nn.state_dict())

        self._nn2 = SimpleNN(in_dim, 1, layers, activations, init_type, gain, layer_norm)
        self._nn2_targ = SimpleNN(in_dim, 1, layers, activations, init_type, gain, layer_norm)
        self._nn2_targ.load_state_dict(self._nn2.state_dict())
        
        self._nn.to(self.device)
        self._nn_targ.to(self.device)
        self._nn2.to(self.device)
        self._nn2_targ.to(self.device)

        # Trainable parameters are those of both networks
        self.trainable = list(self._nn.parameters()) + list(self._nn2.parameters())
        self.use_cagrad = False
    def _forward(self, data, nn_choice='base1'):
        s, a = data
        
        if not torch.is_tensor(a):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
            
        # Normalize state via s_rms
        s_norm = self.s_rms.normalize(s)
        a_norm = torch.clamp(a, min=self.act_low[0], max=self.act_high[0])

        # Convert back to torch after normalization
        s_feat = transform_features(s_norm).to(self.device) 
        a_feat = transform_features(a_norm).to(self.device) 

        sa_feat = torch.cat([s_feat, a_feat], dim=-1)

        if nn_choice == 'base1':
            return self._nn(sa_feat)
        elif nn_choice == 'base2':
            return self._nn2(sa_feat)
        elif nn_choice == 'targ1':
            return self._nn_targ(sa_feat)
        elif nn_choice == 'targ2':
            return self._nn2_targ(sa_feat)

    def _nn_value(self, in_data, nn_choice='base1'):
        out = self._forward(in_data, nn_choice)
        value = out.squeeze(-1)
        # Denormalize value
        if self.safety:
            value = self.c_ret_rms.denormalize(value, center=False)
        else:
            value = self.ret_rms.denormalize(value, center=False)
        # Convert back to torch for consistent usage
        return value

    def value(self, in_data):
        value_base1 = self._nn_value(in_data, nn_choice='base1')
        value_base2 = self._nn_value(in_data, nn_choice='base2')
        return torch.mean(torch.stack([value_base1, value_base2], dim=0), dim=0)

    def value_targ(self, in_data):
        value_targ1 = self._nn_value(in_data, nn_choice='targ1')
        value_targ2 = self._nn_value(in_data, nn_choice='targ2')
        # Mitigate Overestimation Bias
        return torch.minimum(value_targ1, value_targ2)

    def _get_nn_loss(self, in_data, target, nn_choice='base1'):
        value = self._nn_value(in_data, nn_choice)

        if self.safety:
            value_norm = self.c_ret_rms.normalize(value, center=False)
            target_norm = self.c_ret_rms.normalize(target, center=False)
        else:
            value_norm = self.ret_rms.normalize(value, center=False)
            target_norm = self.ret_rms.normalize(target, center=False)
        # Mean Squared Error
        return 0.5 * torch.mean((target_norm - value_norm)**2)

    def get_loss(self, in_data, target):
        loss_base1 = self._get_nn_loss(in_data, target, nn_choice='base1')
        loss_base2 = self._get_nn_loss(in_data, target, nn_choice='base2')
        
        # Twin critics - Summing their losses ensures both networks are trained simultaneously
        return loss_base1 + loss_base2

    def update_targs(self, tau):
        with torch.no_grad():
            # Update _nn_targ
            for p_targ, p in zip(self._nn_targ.parameters(), self._nn.parameters()):
                p_targ.copy_((1 - tau)*p_targ + tau*p)
            # Update _nn2_targ
            for p_targ, p in zip(self._nn2_targ.parameters(), self._nn2.parameters()):
                p_targ.copy_((1 - tau)*p_targ + tau*p)

    def get_weights(self):
        # Get weights as lists of numpy arrays for each network
        def get_weights_list(model):
            return [p.detach().cpu().numpy() for p in model.parameters()]

        weights_base1 = get_weights_list(self._nn)
        weights_base2 = get_weights_list(self._nn2)
        weights_targ1 = get_weights_list(self._nn_targ)
        weights_targ2 = get_weights_list(self._nn2_targ)

        return weights_base1, weights_base2, weights_targ1, weights_targ2

    def set_weights(self, weights):
        weights_base1, weights_base2, weights_targ1, weights_targ2 = weights

        def set_weights_list(model, w_list):
            with torch.no_grad():
                for p, w in zip(model.parameters(), w_list):
                    p.copy_(torch.tensor(w, device=self.device))

        set_weights_list(self._nn, weights_base1)
        set_weights_list(self._nn2, weights_base2)
        set_weights_list(self._nn_targ, weights_targ1)
        set_weights_list(self._nn2_targ, weights_targ2)


class MultiPerspectiveCritic(BaseCritic):
    def __init__(self, env, layers, activations, init_type, gain, layer_norm, num_objectives=1, safety=False, **kwargs):
        super(MultiPerspectiveCritic, self).__init__(env)
        self.safety = safety
        self.num_objectives = num_objectives
        in_dim = self.s_dim + self.a_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("self.device:", self.device)
        
        self._nn = SimpleNN(in_dim, num_objectives, layers, activations, init_type, gain, layer_norm)
        self._nn_targ = SimpleNN(in_dim, num_objectives, layers, activations, init_type, gain, layer_norm)
        self._nn_targ.load_state_dict(self._nn.state_dict())

        self._nn2 = SimpleNN(in_dim, num_objectives, layers, activations, init_type, gain, layer_norm)
        self._nn2_targ = SimpleNN(in_dim, num_objectives, layers, activations, init_type, gain, layer_norm)
        self._nn2_targ.load_state_dict(self._nn2.state_dict())
        
        self._nn.to(self.device)
        self._nn_targ.to(self.device)
        self._nn2.to(self.device)
        self._nn2_targ.to(self.device)

        # Trainable parameters are those of both networks
        self.trainable = list(self._nn.parameters()) + list(self._nn2.parameters())
        
        
        # Add influence tracking
        self.influence_weights = torch.ones(num_objectives, device=self.device) / num_objectives
        self.influence_smoothing = 0.95  # For temporal smoothing
        self.use_dynamic_weights = kwargs.get('use_dynamic_weights', False)
        self.use_cagrad = kwargs.get('use_cagrad', False)
        self.cagrad_c = kwargs.get('cagrad_c', 0.5)  
        
        
    def _forward(self, data, nn_choice='base1'):
        s, a = data
        
        if not torch.is_tensor(a):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
            
        # Normalize state via s_rms
        s_norm = self.s_rms.normalize(s)
        a_norm = torch.clamp(a, min=self.act_low[0], max=self.act_high[0])

        # Convert back to torch after normalization
        s_feat = transform_features(s_norm).to(self.device) 
        a_feat = transform_features(a_norm).to(self.device) 

        sa_feat = torch.cat([s_feat, a_feat], dim=-1)

        if nn_choice == 'base1':
            return self._nn(sa_feat)
        elif nn_choice == 'base2':
            return self._nn2(sa_feat)
        elif nn_choice == 'targ1':
            return self._nn_targ(sa_feat)
        elif nn_choice == 'targ2':
            return self._nn2_targ(sa_feat)

    def _nn_value(self, in_data, nn_choice='base1'):
        """
        Get Q-values from network.
        For multi-head case returns values for all objectives.
        For single case returns single value.
        
        Args:
            in_data: tuple of (states, actions)
            nn_choice: which network to use ('base1', 'base2', 'targ1', 'targ2')
        Returns:
            value: tensor of shape [batch_size] or [batch_size, num_objectives]
        """
        
        out = self._forward(in_data, nn_choice)
        
        # Denormalize value
        if self.safety:
            value = self.c_ret_rms.denormalize(out, center=False)
        else:
            value = self.ret_rms.denormalize(out, center=False)
        # Convert back to torch for consistent usage
        return value

    def value(self, in_data):
        value_base1 = self._nn_value(in_data, nn_choice='base1')
        value_base2 = self._nn_value(in_data, nn_choice='base2')
        return torch.mean(torch.stack([value_base1, value_base2], dim=0), dim=0)

    def value_targ(self, in_data):
        value_targ1 = self._nn_value(in_data, nn_choice='targ1')
        value_targ2 = self._nn_value(in_data, nn_choice='targ2')
        # Mitigate Overestimation Bias
        return torch.minimum(value_targ1, value_targ2)

    def _get_nn_loss(self, in_data, target, nn_choice='base1'):
        """
        Compute loss for one network.
        
        Args:
            in_data: tuple of (states, actions)
            target: tensor of shape [batch_size] or [batch_size, num_objectives]
            nn_choice: which network to use
        Returns:
            loss: scalar tensor for single-head or vector tensor for multi-head
        """
        value = self._nn_value(in_data, nn_choice)

        if self.safety:
            value_norm = self.c_ret_rms.normalize(value, center=False)
            target_norm = self.c_ret_rms.normalize(target, center=False)
        else:
            value_norm = self.ret_rms.normalize(value, center=False)
            target_norm = self.ret_rms.normalize(target, center=False)
        
        # If multi-head, compute loss per objective
        if len(value_norm.shape) > 1:  # multi-head case
            # Mean over batch dimension, keeping objective dimension
            loss = 0.5 * torch.mean((target_norm - value_norm)**2, dim=0)
        else:  # single-head case - Mean Squared Error
            loss = 0.5 * torch.mean((target_norm - value_norm)**2)
            
        return loss
        
    def get_loss(self, in_data, target):
        """
        Get combined loss from both networks.
        
        Args:
            in_data: tuple of (states, actions)
            target: tensor of shape [batch_size] or [batch_size, num_objectives]
        Returns:
            loss: scalar tensor for single-head or vector tensor for multi-head
        """
        loss_base1 = self._get_nn_loss(in_data, target, nn_choice='base1')
        loss_base2 = self._get_nn_loss(in_data, target, nn_choice='base2')
            
        # Combine losses from both networks
        losses  = loss_base1 + loss_base2
        
        if self.use_cagrad and isinstance(losses, (list, tuple)) or (isinstance(losses, torch.Tensor) and len(losses.shape) > 0):
            # Multi-objective case
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
                
            # Compute gradients for each objective
            gradients = []
            for loss_i in losses:
                grad_i = torch.autograd.grad(loss_i, self.trainable, retain_graph=True)
                grad_vec_i = torch.cat([g.reshape(-1) for g in grad_i])
                gradients.append(grad_vec_i)

            # Stack gradients and apply CAGrad
            grad_tensor = torch.stack(gradients)
            final_grad = self.cagrad(grad_tensor)

            # Reshape and apply gradients
            pointer = 0
            for param in self.trainable:
                num_param = param.numel()
                param.grad = final_grad[pointer:pointer + num_param].view(param.shape)
                pointer += num_param

        return losses  # Return original vector of losses for logging

    def update_targs(self, tau):
        """
        Soft update of target networks parameters.
        θ_target = (1 - τ) * θ_target + τ * θ_current
        
        Args:
            tau: float in [0,1], update weight (typically small, e.g., 0.005)
        """
        with torch.no_grad():
            # Update _nn_targ
            for p_targ, p in zip(self._nn_targ.parameters(), self._nn.parameters()):
                p_targ.copy_((1 - tau)*p_targ + tau*p)
            # Update _nn2_targ
            for p_targ, p in zip(self._nn2_targ.parameters(), self._nn2.parameters()):
                p_targ.copy_((1 - tau)*p_targ + tau*p)

    def get_weights(self):
        # Get weights as lists of numpy arrays for each network
        def get_weights_list(model):
            return [p.detach().cpu().numpy() for p in model.parameters()]

        weights_base1 = get_weights_list(self._nn)
        weights_base2 = get_weights_list(self._nn2)
        weights_targ1 = get_weights_list(self._nn_targ)
        weights_targ2 = get_weights_list(self._nn2_targ)

        return weights_base1, weights_base2, weights_targ1, weights_targ2

    def set_weights(self, weights):
        weights_base1, weights_base2, weights_targ1, weights_targ2 = weights

        def set_weights_list(model, w_list):
            with torch.no_grad():
                for p, w in zip(model.parameters(), w_list):
                    p.copy_(torch.tensor(w, device=self.device))

        set_weights_list(self._nn, weights_base1)
        set_weights_list(self._nn2, weights_base2)
        set_weights_list(self._nn_targ, weights_targ1)
        set_weights_list(self._nn2_targ, weights_targ2)

    def compute_influence_metrics(self, s, a):
            """
            Compute influence metric for each head and derive importance weights.
            Based on equation: I_i(s;θ) = λ||∇aQi(s,a;θ) - ∇aQ¬i(s,a;θ)||₂
            
            Args:
                s: states batch [batch_size, state_dim]
                a: actions batch [batch_size, action_dim]
            Returns:
                influence_weights: normalized weights for each head [num_objectives]
            """
            # Convert inputs to tensors if they aren't already
            s = torch.tensor(s, dtype=torch.float32, device=self.device) if not torch.is_tensor(s) else s
            a = torch.tensor(a, dtype=torch.float32, device=self.device) if not torch.is_tensor(a) else a
            
            # We need to enable gradient computation for actions
            a = a.detach().clone()  # Create a new tensor detached from its computation history
            a.requires_grad = True  # Enable gradient computation for this tensor
            
            # Get gradients for each head separately
            head_gradients = []
            for i in range(self.num_objectives):
                # Get Q-value predictions for head i only
                value_i = self.value((s, a))[..., i] # Shape: [batch_size]
                
                # Compute gradient of Q-value with respect to actions
                # retain_graph=True because we'll need to compute more gradients
                grad_i = torch.autograd.grad(value_i.sum(), a, retain_graph=True)[0]
                head_gradients.append(grad_i)
                
            # Compute influence metrics
            influences = []
            for i in range(self.num_objectives):
                # Calculate Q¬i by averaging gradients of all other heads except i
                # This is how we approximate the composite value function excluding component i
                other_grads = torch.stack([g for j, g in enumerate(head_gradients) if j != i]).mean(0)
                # Compute L2 norm of difference between head i's gradient and average of others - how different this head's preferred actions are from others
                influence_i = torch.norm(head_gradients[i] - other_grads, p=2)
                influences.append(influence_i)
                
            # Normalize influences
            influences = torch.stack(influences) # Shape: [num_objectives]
            # This ensures weights sum to 1 while preserving relative importance
            new_weights = influences / influences.sum()
            
            # Temporal smoothing to avoid rapid weight changes
            self.influence_weights = (self.influence_smoothing * self.influence_weights + 
                                    (1 - self.influence_smoothing) * new_weights)
            
            return self.influence_weights
        
        
    def cagrad(self, grad_vec, s=None, a=None):
        """
        Implementation of CAGrad with optional influence-based dynamic weights
        
        Args:
            grad_vec: [num_tasks, dim] gradient vectors
            s: Optional state batch for influence computation
            a: Optional action batch for influence computation
        """
        grads = grad_vec
        num_tasks = grads.shape[0]
        
        # If using dynamic weights based on influence
        if self.use_dynamic_weights and s is not None and a is not None:
            # Compute influence-based weights
            influence_weights = self.influence_weights
        else:
            # Use uniform weights
            influence_weights = torch.ones(num_tasks, device=self.device) / num_tasks

        
        # Move computation to CPU for memory efficiency
        grads_cpu = grads.cpu()
        GG = grads_cpu.mm(grads_cpu.t())
        del grads_cpu
        torch.cuda.empty_cache()

        # Compute Gram matrix and scaling
        # GG = grads.mm(grads.t()).cpu()
        # Scale for numerical stability
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)

        # Compute average gradients
        # Gg = [⟨gi,g0⟩] where g0 is average gradient
        Gg = (GG * influence_weights.cpu().view(-1,1)).sum(1, keepdims=True)  # [num_tasks, 1]
        # gg = ⟨g0,g0⟩ 
        gg = Gg.mean(0, keepdims=True)   # [1, 1]

        # Initialize weights and optimizer
        w = torch.zeros(num_tasks, 1, requires_grad=True)
        # Use SGD to solve the optimization problem
        w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        # Compute scaling factor - c||g0|| term from constraint
        c = (gg + 1e-4).sqrt() * self.cagrad_c

        # Optimize weights
        w_best = None
        obj_best = float('inf')
        for i in range(21):  # 20 optimization steps
            w_opt.zero_grad()
            # Convert to probability simplex using softmax
            ww = torch.softmax(w, 0)
            # Objective: ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww)).sqrt()
            # This corresponds to finding weights that maximize
            # the minimum improvement across tasks while staying
            # close to average gradient
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
                
            if i < 20:
                obj.backward()
                w_opt.step()

        # Compute final gradient using optimal weights
        ww = torch.softmax(w_best, 0)
        # ||gw|| where gw is weighted sum of gradients
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        # Lagrange multiplier λ = c||g0||/||gw||
        lmbda = c.view(-1) / (gw_norm + 1e-4)
        # Final update direction: d* = g0 + λgw/(1 + c²)
        # where g0 is average gradient (1/num_tasks term)
        # and gw is weighted sum of gradients (ww term)
        g = ((1/num_tasks + ww.cuda() * lmbda.cuda()).view(-1, 1) * grads).sum(0) / (1 + self.cagrad_c**2)
        
        return g