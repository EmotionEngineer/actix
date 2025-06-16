# Actix Functions

[![PyPI version](https://badge.fury.io/py/actix.svg)](https://badge.fury.io/py/actix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of novel and experimental activation functions for deep learning, supporting both TensorFlow/Keras and PyTorch.

## Key Features

- **50+ activation functions** including custom parametric and static variants
- **Dual framework support** for TensorFlow/Keras and PyTorch
- **Proven performance gains** over standard activations in benchmarks
- **Simple API** with direct imports and getter functions

## Installation

```bash
pip install actix                # Auto-detects installed frameworks
pip install actix[tf]            # TensorFlow only
pip install actix[torch]         # PyTorch only
pip install actix[tf,torch]      # Both frameworks
```

## Quick Start

### TensorFlow/Keras

```python
from actix import OptimA, ATanSigmoid, get_activation
import tensorflow as tf

# Direct usage
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    OptimA(),
    tf.keras.layers.Dense(64),
    ATanSigmoid(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Using getter function
activation = get_activation('OptimA', framework='tensorflow')
```

### PyTorch

```python
from actix import OptimATorch, ATanSigmoidTorch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = OptimATorch()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = ATanSigmoidTorch()
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)
```

## Recommended Activations

### Universal Top Performers

| Function | Key Strength | Best Use Case |
|----------|--------------|---------------|
| **`ATanSigU`** | Best overall performance | General-purpose, especially classification (77.63% on CIFAR-10) |
| **`A_ELuC`** | Consistent across tasks | When you need reliable performance everywhere |

### Classification Leaders

| Rank | Activation Function | CIFAR-10 Accuracy |
|------|-------------------|-------------------|
| 1 | **`ATanSigU`** | 77.63% |
| 2 | **`AdaptiveSinusoidalSoftgate`** | 77.52% |
| 3 | **`ParametricLogish`** | 77.48% |
| 4 | **`SmoothedAbsoluteGatedUnit`** | 77.43% |
| 5 | **`A_ELuC`** | 77.21% |

### Regression Champions

#### California Housing Dataset

| Rank | Activation Function | MSE | Improvement vs Baseline |
|------|-------------------|-----|------------------------|
| 1 | **`SymmetricParametricRationalSigmoid`** | 0.2183 | -17.75% |
| 2 | **`OptimA`** | 0.2211 | -16.70% |
| 3 | **`OptimXTemporal`** | 0.2213 | -16.62% |

#### Diabetes Dataset

| Rank | Activation Function | MSE | Improvement vs Baseline |
|------|-------------------|-----|------------------------|
| 1 | **`ParametricBetaSoftsign`** | 0.4564 | -2.37% |
| 2 | **`A_ELuC`** | 0.4567 | -2.31% |
| 3 | **`SmoothedAbsoluteGatedUnit`** | 0.4590 | -1.82% |

### Experimental Functions

| Function | Characteristics | Application Domain |
|----------|----------------|-----------------|
| **`ComplexHarmonicActivation`** | Demonstrates significant potential, requires precise hyperparameter optimization | Advanced research applications with computational flexibility |
| **`WeibullSoftplusActivation`** | Exhibits robust performance with predictable convergence properties | Production systems requiring high stability and reliability |
| **`GeneralizedAlphaSigmoid`** | Provides domain-specific adaptability through parameterization | Specialized tasks requiring function customization |
| **`OptimQ`** | Integrates three activation mechanisms with extensive configurability | Complex architectures demanding high adaptability |
| **`StabilizedHarmonic`** | Captures periodic patterns while maintaining numerical stability | Temporal modeling, signal processing, and cyclic phenomena |
| **`AdaptiveArcTanSwish`** | Achieves optimal balance between computational efficiency and expressiveness | Regression tasks requiring enhanced performance |

## Visualization Tools

```python
import actix

# Visualize activation function and its derivative
actix.plot_activation('GeneralizedAlphaSigmoid', framework='tf')
actix.plot_derivative('GeneralizedAlphaSigmoid', framework='tf')
```

## Requirements

- Python 3.7+
- NumPy ≥1.19
- Matplotlib ≥3.3
- TensorFlow ≥2.4 (optional)
- PyTorch ≥1.8 (optional)

## Documentation

For complete function list and mathematical formulas, see [`actix/activations_tf.py`](actix/activations_tf.py)

For detailed benchmarks, check:
- [`cifar.ipynb`](/benchmark/cifar.ipynb)
- [`regression.ipynb`](/benchmark/regression.ipynb)

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functions
4. Ensure code follows PEP8 standards
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.
