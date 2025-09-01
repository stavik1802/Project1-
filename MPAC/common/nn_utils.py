import numpy as np
import torch
import torch.nn as nn
import math

def create_activations(layers, activations_in):
    """Creates list of PyTorch activation modules from string inputs."""
    if len(layers) > 1 and len(activations_in) == 1:
        activations_in = [activations_in[0] for _ in layers]
    activation_modules = []
    for act_str in activations_in:
        if act_str == 'tanh':
            act = nn.Tanh()
        elif act_str == 'relu':
            act = nn.ReLU()
        elif act_str == 'elu':
            act = nn.ELU()
        else:
            raise ValueError('activations must be tanh, relu or elu')
        activation_modules.append(act)

    return activation_modules

def init_layer_orthogonal(layer, gain=None):
    if gain is None:
        gain = math.sqrt(2.0)
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

def init_layer_variance_scaling(layer, gain=None):
    with torch.no_grad():
        if gain is None:
            scale = 0.333
        else:
            scale = gain
        # fan_in, fan_out for a linear layer: weight shape [out_dim, in_dim]
        fan_out = layer.weight.shape[0]
        limit = math.sqrt(3.0 * scale / fan_out)  # uniform distribution
        layer.weight.uniform_(-limit, limit)
        if layer.bias is not None:
            layer.bias.zero_()

def init_layer_glorot_uniform(layer):
    # Glorot uniform is the same as xavier_uniform_
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

def create_initializer_function(init_type, gain=None):
    """Returns a function that, given a layer, applies the chosen initialization."""
    def init_func(layer):
        if init_type == 'orthogonal':
            init_layer_orthogonal(layer, gain=gain)
        elif init_type == 'var':
            init_layer_variance_scaling(layer, gain=gain)
        elif init_type == 'uniform':
            init_layer_glorot_uniform(layer)
        else:
            raise ValueError('init_type must be orthogonal, var or uniform')
    return init_func

def create_initializations(layers, init_type, gain=0.01):
    """
    Creates initialization functions for hidden layers and a final layer.
    Instead of returning initializers, we return callables that
    can be applied to layers after creation.
    """
    init_fns = [create_initializer_function(init_type) for _ in layers]
    init_final = create_initializer_function(init_type, gain=gain)
    return init_fns, init_final

def create_layer(in_dim, out_dim, initialization=None, activation=None):
    """Creates a single PyTorch linear layer and applies initialization."""
    layer = nn.Linear(in_dim, out_dim)
    if initialization is not None:
        initialization(layer)
    # activation is handled outside this function in PyTorch;
    # we typically return just the layer here.
    return layer

def transform_features(s):
    """Updates dtype and shape of inputs to neural networks."""
    if not torch.is_tensor(s):
        s = torch.tensor(s)
    s = s.float()
    if s.dim() == 1:
        s = s.unsqueeze(0)  # Add batch dimension
    return s

class SimpleNN(nn.Module):
    """
    A feedforward neural network with optional layer norm on the first layer.
    This class encapsulates the logic from create_nn in a PyTorch style.
    """
    def __init__(self, in_dim, out_dim, layers, activations, init_type, gain, layer_norm=False):
        self.dtype = torch.float32  # Match TF's default float32
        super(SimpleNN, self).__init__()
        # Convert activation strings to modules
        activation_modules = create_activations(layers, activations)
        initializations, init_final = create_initializations(layers, init_type, gain)

        # We'll build a list of modules and then wrap them in nn.Sequential
        modules = []
        prev_dim = in_dim

        # Build hidden layers
        for i, layer_size in enumerate(layers):
            if i == 0:
                if layer_norm:
                    linear_layer = create_layer(prev_dim, layer_size, initialization=initializations[i])
                    modules.append(linear_layer)
                    modules.append(nn.LayerNorm(layer_size,  eps=1e-3))
                    modules.append(nn.Tanh())
            
                else:
                    linear_layer = create_layer(prev_dim, layer_size, initialization=initializations[i])
                    modules.append(linear_layer)
                    modules.append(activation_modules[i])
            else:
                linear_layer = create_layer(prev_dim, layer_size, initialization=initializations[i])
                modules.append(linear_layer)
                modules.append(activation_modules[i])
            prev_dim = layer_size

        # Final output layer
        final_layer = nn.Linear(prev_dim, out_dim)
        init_final(final_layer)
        modules.append(final_layer)

        # Create a sequential model
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = transform_features(x)
        return self.model(x)

def flat_to_list(trainable_tensors, weights):
    """Converts flattened array back into list of tensors with original shapes."""
    # trainable_tensors is a list of parameters (nn.Parameter)
    # weights is a numpy array of flattened parameters
    shapes = [p.shape for p in trainable_tensors]
    sizes = [p.numel() for p in trainable_tensors]
    idxs = np.cumsum([0]+sizes)

    weights_list = []
    for i, shape in enumerate(shapes):
        elem_flat = weights[idxs[i]:idxs[i+1]]
        elem = elem_flat.reshape(shape)
        weights_list.append(elem)
    return weights_list

def list_to_flat(weights):
    """Flattens list of arrays/tensors into a single array."""
    flat_list = [w.reshape(-1) for w in weights]
    return np.concatenate(flat_list, axis=0)

def soft_value(value):
    """Converts value into soft value (log(exp(value)-1)) if value < 100."""
    if value < 100:
        return np.log(np.exp(value)-1)
    else:
        return value
