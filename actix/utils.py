# actix/utils.py

import numpy as np

__all__ = ['plot_activation', 'plot_derivative']

def plot_activation(name: str, framework: str = 'tensorflow', x_range=(-5, 5), num_points=400):
    """
    Plots an activation function over a specified range.

    Args:
        name (str): The name of the activation function.
        framework (str, optional): The framework to use ('tensorflow' or 'pytorch'). Defaults to 'tensorflow'.
        x_range (tuple, optional): The (min, max) range for the x-axis. Defaults to (-5, 5).
        num_points (int, optional): The number of points to plot. Defaults to 400.
    """
    from actix import get_activation, _TF_AVAILABLE, _TORCH_AVAILABLE

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Please install it to use plotting utilities: pip install matplotlib")
        return

    act_func = get_activation(name, framework=framework)
    x = np.linspace(x_range[0], x_range[1], num_points).astype(np.float32)
    
    if framework.lower() in ('tensorflow', 'tf'):
        if not _TF_AVAILABLE: raise ImportError("TensorFlow not found.")
        import tensorflow as tf
        x_tensor = tf.constant(x)
        y_tensor = act_func(x_tensor)
        y = y_tensor.numpy()
    elif framework.lower() in ('pytorch', 'torch'):
        if not _TORCH_AVAILABLE: raise ImportError("PyTorch not found.")
        import torch
        x_tensor = torch.from_numpy(x)
        y_tensor = act_func(x_tensor)
        y = y_tensor.detach().numpy()
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'f(x) = {name}')
    plt.title(f'Activation Function: {name} ({framework.capitalize()})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()

def plot_derivative(name: str, framework: str = 'tensorflow', x_range=(-5, 5), num_points=400):
    """
    Plots the derivative of an activation function over a specified range.

    Args:
        name (str): The name of the activation function.
        framework (str, optional): The framework to use ('tensorflow' or 'pytorch'). Defaults to 'tensorflow'.
        x_range (tuple, optional): The (min, max) range for the x-axis. Defaults to (-5, 5).
        num_points (int, optional): The number of points to plot. Defaults to 400.
    """
    from actix import get_activation, _TF_AVAILABLE, _TORCH_AVAILABLE

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Please install it to use plotting utilities: pip install matplotlib")
        return

    act_func = get_activation(name, framework=framework)
    x = np.linspace(x_range[0], x_range[1], num_points).astype(np.float32)
    
    if framework.lower() in ('tensorflow', 'tf'):
        if not _TF_AVAILABLE: raise ImportError("TensorFlow not found.")
        import tensorflow as tf
        x_tensor = tf.constant(x)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            y_tensor = act_func(x_tensor)
        dy_dx_tensor = tape.gradient(y_tensor, x_tensor)
        dy_dx = dy_dx_tensor.numpy()
    elif framework.lower() in ('pytorch', 'torch'):
        if not _TORCH_AVAILABLE: raise ImportError("PyTorch not found.")
        import torch
        x_tensor = torch.from_numpy(x).requires_grad_(True)
        y_tensor = act_func(x_tensor)
        dy_dx_tensor, = torch.autograd.grad(y_tensor.sum(), x_tensor, create_graph=False)
        dy_dx = dy_dx_tensor.detach().numpy()
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, dy_dx, label=f"f'(x) of {name}", color='r')
    plt.title(f'Derivative of Activation: {name} ({framework.capitalize()})')
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()
