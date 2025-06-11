# Actix Functions

[![PyPI version](https://badge.fury.io/py/actix.svg)](https://badge.fury.io/py/actix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges if needed (CI/CD, test coverage, etc.) -->

`actix` is a Python package providing a collection of novel and experimental activation functions for deep learning models, implemented for both TensorFlow/Keras and PyTorch. Many of these are custom-designed and aim to offer improved performance or interesting properties compared to standard activations.

## Features

*   **Dual Framework Support:** Seamless integration with TensorFlow (Keras) and PyTorch.
*   **Parametric Activations:** Functions with trainable parameters that adapt during training.
*   **Static Activations:** Novel non-parametric functions.
*   **Easy to Use:** Simple API to get and use activation functions.

## Choosing Activation Function

Based on extensive testing on classification and regression tasks, it's recommended to start with the following activation functions. They have consistently shown superior performance, stability, and versatility compared to standard activations.

For a complete list of all 40+ implemented functions and their mathematical formulas, please refer to the source code: [`actix/activations_tf.py`](actix/activations_tf.py)

### Universal Performers (Best All-Rounders)

If you need a single function that works exceptionally well across most tasks, choose one of these.

*   **`OptimA`**: **The Go-To Choice.** This function was the clear winner in regression tasks and a top-tier performer in classification. Its combination of `tanh` and gated `softplus` provides a powerful and flexible non-linearity.
*   **`A_ELuC`**: **A Powerful & Versatile Alternative.** Often the best performer in classification and very strong in regression. It combines the benefits of `ELU` with a gating mechanism, making it highly effective.

### Top Choices for Classification

If your primary task is classification (e.g., image recognition), the following options are recommended in order of preference:

1.  **`A_ELuC`**: Showed the highest accuracy in several tests, outperforming even modern standards like `gelu` and `swish`.
2.  **`ParametricLogish`**: A simple but highly effective parametric version of Swish (`x * sigmoid(x)`). A great, lightweight replacement for `swish` or `gelu` that often performs better.
3.  **`OptimA`**: While it shines brightest in regression, its performance in classification is also excellent and highly reliable.

### Top Choices for Regression

For regression tasks (e.g., predicting house prices or other continuous values):

1.  **`OptimA`**: The undisputed champion in our regression benchmarks. Consistently delivered the lowest error rates and should be your first choice.
2.  **`OptimXTemporal`**: A slightly simpler version of `OptimA` that also achieves top-tier performance. A great, efficient choice for regression.
3.  **`A_STReLU`** / **`ATanSigU`**: Both functions demonstrated excellent and stable results on tabular regression datasets, making them strong candidates.

### Experimental & Specialized Functions

These functions can provide exceptional results on specific tasks but may be less stable or require more tuning. Use them when standard approaches are not sufficient.

*   **`ComplexHarmonicActivation`**: A "secret weapon" for specific regression datasets but can be highly unstable on others. High risk, high reward.
*   **`WeibullSoftplusActivation`**: A consistent and reliable performer that, while not always #1, is a very safe and solid choice across different tasks.
*   **`GeneralizedAlphaSigmoid`**: Showed strong potential in specific regression scenarios, making it worth a try if the top contenders don't yield desired results.

## Performance Highlights

The `actix` activation functions have been benchmarked on standard datasets against common activations. Below are the highlights from the provided notebook runs. Note that performance can vary based on model architecture, hyperparameters, and random seeds. These results are intended to showcase the potential of `actix` functions.

The benchmarks were run using a CNN architecture for CIFAR-10 and a simple MLP for the regression tasks (Diabetes, California Housing).

**Key:**
*   **Acc (↑):** Mean Test Accuracy (higher is better)
*   **MSE (↓):** Mean Test Loss (lower is better)

| Dataset | Metric | LR | Top `actix` Activation(s) | `actix` Mean Score | Best Standard Activation | Standard Mean Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | Acc (↑) | `1e-3` | `ParametricLogish` | 0.7730 | `gelu` | 0.7703 |
| | | | `OptimA` | 0.7684 | `swish` | 0.7697 |
| **California Housing** | MSE (↓) | `1e-2` | `OptimA` | 0.2143 | `sigmoid` | 0.2235 |
| | | | `OptimXTemporal` | 0.2183 | `relu` | 0.2399 |
| **Diabetes** | MSE (↓) | `1e-2` | `A_ELuC` | 0.4345 | `swish` | 0.4408 |
| | | | `ParametricBetaSoftsign` | 0.4349 | `mish` | 0.4523 |

**Observations:**

*   On **CIFAR-10**, `ParametricLogish` slightly outperformed the best standard activation (`gelu`). Several other `actix` functions like `OptimA` and `A_ELuC` were also highly competitive, performing on par with or better than `relu`.
*   On the **California Housing** regression task, `actix` functions showed a notable advantage. `OptimA` and `OptimXTemporal` achieved a significantly lower (better) MSE than the best-performing standard activation (`sigmoid`).
*   On the **Diabetes** regression task, the top `actix` functions (`A_ELuC`, `ParametricBetaSoftsign`) again edged out the best standard activation (`swish`), demonstrating their potential in tabular regression settings.

It's important to note that some `actix` functions (e.g., `ParametricBetaSoftsign` on CIFAR-10, or `ParametricGeneralizedGompertzActivation` on California Housing) did not converge well or resulted in errors (NaNs) under these specific test conditions. Many other experimental activations also showed poor performance, highlighting their sensitivity and the importance of hyperparameter tuning (especially the learning rate) when exploring new activation functions.

For more detailed results, please refer to the benchmark notebooks: [`cifar.ipynb`](/benchmark/cifar.ipynb) and [`regression.ipynb`](/benchmark/regression.ipynb).

## Installation

You can install `actix` via pip:

```bash
pip install actix
```

The package will automatically detect if TensorFlow or PyTorch (or both) are installed. The corresponding activation functions will be made available. To install with specific framework support:
```bash
pip install actix[tf]    # For TensorFlow only
pip install actix[torch] # For PyTorch only
pip install actix[tf,torch] # For both
```


## Usage

### TensorFlow / Keras

```python
import tensorflow as tf
# Import directly (recommended for custom layers)
from actix import OptimA, ATanSigmoid
# Or use the getter function
from actix import get_activation

# Option 1: Direct class instantiation
model_tf_direct = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    OptimA(), # Custom Keras Layer
    tf.keras.layers.Dense(64),
    ATanSigmoid(), # Custom Keras Layer for static activation
    tf.keras.layers.Dense(10, activation='softmax')
])
model_tf_direct.summary()

# Option 2: Using the get_activation function
optima_layer_tf = get_activation('OptimA', framework='tensorflow')
atan_sigmoid_layer_tf = get_activation('ATanSigmoid', framework='tf')
# For standard Keras activations through the getter:
# relu_tf = get_activation('relu', framework='tf')

model_tf_getter = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    optima_layer_tf,
    tf.keras.layers.Dense(64),
    atan_sigmoid_layer_tf,
    tf.keras.layers.Dense(10, activation='softmax')
])
model_tf_getter.summary()

# Compile and train (example)
# (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# model_tf_direct.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model_tf_direct.fit(x_train[:100], y_train[:100], epochs=1)
```

### PyTorch

```python
import torch
import torch.nn as nn
# Import directly (recommended for custom modules)
# Note: PyTorch versions might have 'Torch' suffix if TensorFlow version with same name exists
from actix import OptimATorch, ATanSigmoidTorch
# Or use the getter function
from actix import get_activation

# Option 1: Direct class instantiation
class MyModelTorchDirect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = OptimATorch() # Custom nn.Module
        self.fc2 = nn.Linear(128, 64)
        self.act2 = ATanSigmoidTorch() # Custom nn.Module for static activation
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.act1(self.fc1(x.view(-1, 784)))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

model_torch_direct = MyModelTorchDirect()
print(model_torch_direct)

# Option 2: Using the get_activation function
optima_module_torch = get_activation('OptimA', framework='pytorch')
atan_sigmoid_module_torch = get_activation('ATanSigmoid', framework='torch')
# For standard PyTorch activations through the getter:
# relu_torch = get_activation('ReLU', framework='pytorch') # or 'relu'

class MyModelTorchGetter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = optima_module_torch
        self.fc2 = nn.Linear(128, 64)
        self.act2 = atan_sigmoid_module_torch
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.act1(self.fc1(x.view(-1, 784)))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

model_torch_getter = MyModelTorchGetter()
print(model_torch_getter)

# Compile and train (example)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_torch_direct.parameters())
# dummy_input = torch.randn(2, 784)
# labels = torch.randint(0, 10, (2,))
# output = model_torch_direct(dummy_input)
# loss = criterion(output, labels)
# loss.backward()
# optimizer.step()
```

### Utility Functions (Plotting)

`actix` also includes helper utilities to quickly visualize any activation function and its derivative. This is useful for understanding the behavior of a function before using it in a model. The `plot_activation` and `plot_derivative` functions work for both custom and standard activations from either framework.

```python
import actix

print("--- Plotting multiple functions ---")

# Example 1: A new function for TensorFlow
if actix._TF_AVAILABLE:
    print("\nPlot for GeneralizedAlphaSigmoid (TensorFlow)")
    actix.plot_activation('GeneralizedAlphaSigmoid', framework='tf')
    actix.plot_derivative('GeneralizedAlphaSigmoid', framework='tf')

# Example 2: A new function for PyTorch
if actix._TORCH_AVAILABLE:
    print("\nPlot for GeneralizedAlphaSigmoid (PyTorch)")
    actix.plot_activation('GeneralizedAlphaSigmoid', framework='torch')
    actix.plot_derivative('GeneralizedAlphaSigmoid', framework='torch')

# Example 3: A standard function for comparison
# Note: The plotting utility can also plot standard activations
if actix._TORCH_AVAILABLE:
    print("\nPlot for standard 'sigmoid' (PyTorch)")
    actix.plot_activation('sigmoid', framework='torch')
    actix.plot_derivative('sigmoid', framework='torch')
```

## Dependencies
*   Python 3.7+
*   NumPy (>=1.19)
*   Matplotlib (>=3.3)
*   TensorFlow (>=2.4, optional, for TF activations)
*   PyTorch (>=1.8, optional, for PyTorch activations)

## Contributing
Contributions are welcome! If you have an idea for a new activation function, find a bug, or want to improve documentation, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Add tests for your changes.
5.  Ensure all tests pass (`pytest`).
6.  Commit your changes (`git commit -m 'Add some feature'`).
7.  Push to the branch (`git push origin feature/your-feature-name`).
8.  Open a Pull Request.

Please ensure your code adheres to common Python coding standards (e.g., PEP8/Black, Flake8).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
