# actix/__init__.py

# Determine which frameworks are available
_TF_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass

if not _TF_AVAILABLE and not _TORCH_AVAILABLE:
    print("Warning: Neither TensorFlow nor PyTorch are installed. "
          "The 'actix' activation functions will not be available.")

# --- TensorFlow Activations ---
if _TF_AVAILABLE:
    from .activations_tf import (
        OptimA, ParametricPolyTanh, AdaptiveRationalSoftsign, OptimXTemporal,
        ParametricGaussianActivation, LearnableFourierActivation, A_ELuC,
        ParametricSmoothStep, AdaptiveBiHyperbolic, ParametricLogish, AdaptSigmoidReLU,
        SinhGate, SoftRBF, ATanSigmoid, ExpoSoft, HarmonicTanh, RationalSoftplus,
        UnifiedSineExp, SigmoidErf, LogCoshGate, TanhArc,
        ParametricLambertWActivation, AdaptiveHyperbolicLogarithm,
        ParametricGeneralizedGompertzActivation, ComplexHarmonicActivation,
        WeibullSoftplusActivation, AdaptiveErfSwish, ParametricBetaSoftsign,
        ParametricArcSinhGate, GeneralizedAlphaSigmoid, RiemannianSoftsignActivation,
        QuantumTanhActivation, LogExponentialActivation, BipolarGaussianArctanActivation,
        EllipticGaussianActivation, ExpArcTanHarmonicActivation, LogisticWActivation,
        ParametricTanhSwish, GeneralizedHarmonicSwish, A_STReLU, ETU, PMGLU,
        GPOSoft, SHLU, GaussSwish, ATanSigU, PAPG,
        SwishLogTanh, ArcGaLU, ParametricHyperbolicQuadraticActivation, RootSoftplus,
        AdaptiveSinusoidalSoftgate, ExpTanhGatedActivation, HybridSinExpUnit,
        ParametricLogarithmicSwish, AdaptiveCubicSigmoid, SmoothedAbsoluteGatedUnit,
        GaussianTanhHarmonicUnit, SymmetricParametricRationalSigmoid, AdaptivePolynomialSwish,
        LogSigmoidGatedElu, AdaptiveBipolarExponentialUnit, ParametricHyperGaussianGate,
        TanhGatedArcsinhLinearUnit, ParametricOddPowerSwish, AdaptiveLinearLogTanh,
        AdaptiveArcTanSwish, StabilizedHarmonic, RationalSwish, AdaptiveGatedUnit,
        ExponentialArcTan, OptimQ
    )
    # Dictionary for convenient access by string name in TensorFlow
    tf_activations_map = {
        'OptimA': OptimA, 'ParametricPolyTanh': ParametricPolyTanh,
        'AdaptiveRationalSoftsign': AdaptiveRationalSoftsign, 'OptimXTemporal': OptimXTemporal,
        'ParametricGaussianActivation': ParametricGaussianActivation,
        'LearnableFourierActivation': LearnableFourierActivation, 'A_ELuC': A_ELuC,
        'ParametricSmoothStep': ParametricSmoothStep, 'AdaptiveBiHyperbolic': AdaptiveBiHyperbolic,
        'ParametricLogish': ParametricLogish, 'AdaptSigmoidReLU': AdaptSigmoidReLU,
        'SinhGate': SinhGate, 'SoftRBF': SoftRBF, 'ATanSigmoid': ATanSigmoid,
        'ExpoSoft': ExpoSoft, 'HarmonicTanh': HarmonicTanh, 'RationalSoftplus': RationalSoftplus,
        'UnifiedSineExp': UnifiedSineExp, 'SigmoidErf': SigmoidErf, 'LogCoshGate': LogCoshGate,
        'TanhArc': TanhArc,
        'ParametricLambertWActivation': ParametricLambertWActivation,
        'AdaptiveHyperbolicLogarithm': AdaptiveHyperbolicLogarithm,
        'ParametricGeneralizedGompertzActivation': ParametricGeneralizedGompertzActivation,
        'ComplexHarmonicActivation': ComplexHarmonicActivation,
        'WeibullSoftplusActivation': WeibullSoftplusActivation,
        'AdaptiveErfSwish': AdaptiveErfSwish,
        'ParametricBetaSoftsign': ParametricBetaSoftsign,
        'ParametricArcSinhGate': ParametricArcSinhGate,
        'GeneralizedAlphaSigmoid': GeneralizedAlphaSigmoid,
        'RiemannianSoftsignActivation': RiemannianSoftsignActivation,
        'QuantumTanhActivation': QuantumTanhActivation,
        'LogExponentialActivation': LogExponentialActivation,
        'BipolarGaussianArctanActivation': BipolarGaussianArctanActivation,
        'EllipticGaussianActivation': EllipticGaussianActivation,
        'ExpArcTanHarmonicActivation': ExpArcTanHarmonicActivation,
        'LogisticWActivation': LogisticWActivation,
        'ParametricTanhSwish': ParametricTanhSwish,
        'GeneralizedHarmonicSwish': GeneralizedHarmonicSwish,
        'A_STReLU': A_STReLU,
        'ETU': ETU,
        'PMGLU': PMGLU,
        'GPOSoft': GPOSoft,
        'SHLU': SHLU,
        'GaussSwish': GaussSwish,
        'ATanSigU': ATanSigU,
        'PAPG': PAPG,
        'SwishLogTanh': SwishLogTanh,
        'ArcGaLU': ArcGaLU,
        'ParametricHyperbolicQuadraticActivation': ParametricHyperbolicQuadraticActivation,
        'RootSoftplus': RootSoftplus,
        'AdaptiveSinusoidalSoftgate': AdaptiveSinusoidalSoftgate,
        'ExpTanhGatedActivation': ExpTanhGatedActivation,
        'HybridSinExpUnit': HybridSinExpUnit,
        'ParametricLogarithmicSwish': ParametricLogarithmicSwish,
        'AdaptiveCubicSigmoid': AdaptiveCubicSigmoid,
        'SmoothedAbsoluteGatedUnit': SmoothedAbsoluteGatedUnit,
        'GaussianTanhHarmonicUnit': GaussianTanhHarmonicUnit,
        'SymmetricParametricRationalSigmoid': SymmetricParametricRationalSigmoid,
        'AdaptivePolynomialSwish': AdaptivePolynomialSwish,
        'LogSigmoidGatedElu': LogSigmoidGatedElu,
        'AdaptiveBipolarExponentialUnit': AdaptiveBipolarExponentialUnit,
        'ParametricHyperGaussianGate': ParametricHyperGaussianGate,
        'TanhGatedArcsinhLinearUnit': TanhGatedArcsinhLinearUnit,
        'ParametricOddPowerSwish': ParametricOddPowerSwish,
        'AdaptiveLinearLogTanh': AdaptiveLinearLogTanh,
        'AdaptiveArcTanSwish': AdaptiveArcTanSwish,
        'StabilizedHarmonic': StabilizedHarmonic,
        'RationalSwish': RationalSwish,
        'AdaptiveGatedUnit': AdaptiveGatedUnit,
        'ExponentialArcTan': ExponentialArcTan,
        'OptimQ': OptimQ,
    }
else:
    tf_activations_map = {}

# --- PyTorch Activations ---
if _TORCH_AVAILABLE:
    from .activations_torch import (
        OptimATorch, ParametricPolyTanhTorch, AdaptiveRationalSoftsignTorch, OptimXTemporalTorch,
        ParametricGaussianActivationTorch, LearnableFourierActivationTorch, A_ELuCTorch,
        ParametricSmoothStepTorch, AdaptiveBiHyperbolicTorch, ParametricLogishTorch,
        AdaptSigmoidReLUTorch, SinhGateTorch, SoftRBFTorch, ATanSigmoidTorch, ExpoSoftTorch,
        HarmonicTanhTorch, RationalSoftplusTorch, UnifiedSineExpTorch, SigmoidErfTorch,
        LogCoshGateTorch, TanhArcTorch,
        ParametricLambertWActivationTorch, AdaptiveHyperbolicLogarithmTorch,
        ParametricGeneralizedGompertzActivationTorch, ComplexHarmonicActivationTorch,
        WeibullSoftplusActivationTorch, AdaptiveErfSwishTorch, ParametricBetaSoftsignTorch,
        ParametricArcSinhGateTorch, GeneralizedAlphaSigmoidTorch, RiemannianSoftsignActivationTorch,
        QuantumTanhActivationTorch, LogExponentialActivationTorch, BipolarGaussianArctanActivationTorch,
        EllipticGaussianActivationTorch, ExpArcTanHarmonicActivationTorch, LogisticWActivationTorch,
        ParametricTanhSwishTorch, GeneralizedHarmonicSwishTorch, A_STReLUTorch, ETUTorch, PMGLUTorch,
        GPOSoftTorch, SHLUTorch, GaussSwishTorch, ATanSigUTorch, PAPGTorch,
        SwishLogTanhTorch, ArcGaLUTorch, ParametricHyperbolicQuadraticActivationTorch, RootSoftplusTorch,
        AdaptiveSinusoidalSoftgateTorch, ExpTanhGatedActivationTorch, HybridSinExpUnitTorch,
        ParametricLogarithmicSwishTorch, AdaptiveCubicSigmoidTorch, SmoothedAbsoluteGatedUnitTorch,
        GaussianTanhHarmonicUnitTorch, SymmetricParametricRationalSigmoidTorch, AdaptivePolynomialSwishTorch,
        LogSigmoidGatedEluTorch, AdaptiveBipolarExponentialUnitTorch, ParametricHyperGaussianGateTorch,
        TanhGatedArcsinhLinearUnitTorch, ParametricOddPowerSwishTorch, AdaptiveLinearLogTanhTorch,
        AdaptiveArcTanSwishTorch, StabilizedHarmonicTorch, RationalSwishTorch,
        AdaptiveGatedUnitTorch, ExponentialArcTanTorch, OptimQTorch
    )
    # Dictionary for convenient access by string name in PyTorch
    torch_activations_map = {
        'OptimA': OptimATorch, 'ParametricPolyTanh': ParametricPolyTanhTorch,
        'AdaptiveRationalSoftsign': AdaptiveRationalSoftsignTorch, 'OptimXTemporal': OptimXTemporalTorch,
        'ParametricGaussianActivation': ParametricGaussianActivationTorch,
        'LearnableFourierActivation': LearnableFourierActivationTorch, 'A_ELuC': A_ELuCTorch,
        'ParametricSmoothStep': ParametricSmoothStepTorch, 'AdaptiveBiHyperbolic': AdaptiveBiHyperbolicTorch,
        'ParametricLogish': ParametricLogishTorch, 'AdaptSigmoidReLU': AdaptSigmoidReLUTorch,
        'SinhGate': SinhGateTorch, 'SoftRBF': SoftRBFTorch, 'ATanSigmoid': ATanSigmoidTorch,
        'ExpoSoft': ExpoSoftTorch, 'HarmonicTanh': HarmonicTanhTorch, 'RationalSoftplus': RationalSoftplusTorch,
        'UnifiedSineExp': UnifiedSineExpTorch, 'SigmoidErf': SigmoidErfTorch, 'LogCoshGate': LogCoshGateTorch,
        'TanhArc': TanhArcTorch,
        'ParametricLambertWActivation': ParametricLambertWActivationTorch,
        'AdaptiveHyperbolicLogarithm': AdaptiveHyperbolicLogarithmTorch,
        'ParametricGeneralizedGompertzActivation': ParametricGeneralizedGompertzActivationTorch,
        'ComplexHarmonicActivation': ComplexHarmonicActivationTorch,
        'WeibullSoftplusActivation': WeibullSoftplusActivationTorch,
        'AdaptiveErfSwish': AdaptiveErfSwishTorch,
        'ParametricBetaSoftsign': ParametricBetaSoftsignTorch,
        'ParametricArcSinhGate': ParametricArcSinhGateTorch,
        'GeneralizedAlphaSigmoid': GeneralizedAlphaSigmoidTorch,
        'RiemannianSoftsignActivation': RiemannianSoftsignActivationTorch,
        'QuantumTanhActivation': QuantumTanhActivationTorch,
        'LogExponentialActivation': LogExponentialActivationTorch,
        'BipolarGaussianArctanActivation': BipolarGaussianArctanActivationTorch,
        'EllipticGaussianActivation': EllipticGaussianActivationTorch,
        'ExpArcTanHarmonicActivation': ExpArcTanHarmonicActivationTorch,
        'LogisticWActivation': LogisticWActivationTorch,
        'ParametricTanhSwish': ParametricTanhSwishTorch,
        'GeneralizedHarmonicSwish': GeneralizedHarmonicSwishTorch,
        'A_STReLU': A_STReLUTorch,
        'ETU': ETUTorch,
        'PMGLU': PMGLUTorch,
        'GPOSoft': GPOSoftTorch,
        'SHLU': SHLUTorch,
        'GaussSwish': GaussSwishTorch,
        'ATanSigU': ATanSigUTorch,
        'PAPG': PAPGTorch,
        'SwishLogTanh': SwishLogTanhTorch,
        'ArcGaLU': ArcGaLUTorch,
        'ParametricHyperbolicQuadraticActivation': ParametricHyperbolicQuadraticActivationTorch,
        'RootSoftplus': RootSoftplusTorch,
        'AdaptiveSinusoidalSoftgate': AdaptiveSinusoidalSoftgateTorch,
        'ExpTanhGatedActivation': ExpTanhGatedActivationTorch,
        'HybridSinExpUnit': HybridSinExpUnitTorch,
        'ParametricLogarithmicSwish': ParametricLogarithmicSwishTorch,
        'AdaptiveCubicSigmoid': AdaptiveCubicSigmoidTorch,
        'SmoothedAbsoluteGatedUnit': SmoothedAbsoluteGatedUnitTorch,
        'GaussianTanhHarmonicUnit': GaussianTanhHarmonicUnitTorch,
        'SymmetricParametricRationalSigmoid': SymmetricParametricRationalSigmoidTorch,
        'AdaptivePolynomialSwish': AdaptivePolynomialSwishTorch,
        'LogSigmoidGatedElu': LogSigmoidGatedEluTorch,
        'AdaptiveBipolarExponentialUnit': AdaptiveBipolarExponentialUnitTorch,
        'ParametricHyperGaussianGate': ParametricHyperGaussianGateTorch,
        'TanhGatedArcsinhLinearUnit': TanhGatedArcsinhLinearUnitTorch,
        'ParametricOddPowerSwish': ParametricOddPowerSwishTorch,
        'AdaptiveLinearLogTanh': AdaptiveLinearLogTanhTorch,
        'AdaptiveArcTanSwish': AdaptiveArcTanSwishTorch,
        'StabilizedHarmonic': StabilizedHarmonicTorch,
        'RationalSwish': RationalSwishTorch,
        'AdaptiveGatedUnit': AdaptiveGatedUnitTorch,
        'ExponentialArcTan': ExponentialArcTanTorch,
        'OptimQ': OptimQTorch,
    }
else:
    torch_activations_map = {}


def get_activation(name: str, framework: str = 'tensorflow'):
    """
    Retrieves an activation function/layer by its name for the specified framework.

    Args:
        name (str): The name of the activation function (case-sensitive for custom ones).
        framework (str, optional): The deep learning framework.
                                   Options: 'tensorflow', 'tf', 'pytorch', 'torch'.
                                   Defaults to 'tensorflow'.

    Returns:
        An instance of the activation layer/module.

    Raises:
        ImportError: If the requested framework is not installed.
        ValueError: If the activation function is not found or the framework is unsupported.
    """
    framework = framework.lower()
    if framework in ('tensorflow', 'tf'):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed, but was requested for 'actix' activation.")
        if name in tf_activations_map:
            return tf_activations_map[name]() # Return an instance for Keras Layers
        else:
            # Try to get standard Keras activations
            try:
                return tf.keras.activations.get(name)
            except ValueError:
                raise ValueError(
                    f"TensorFlow activation '{name}' not found in 'actix' package or Keras standard activations."
                )
    elif framework in ('pytorch', 'torch'):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed, but was requested for 'actix' activation.")
        if name in torch_activations_map:
            return torch_activations_map[name]() # Return an instance for PyTorch Modules
        else:
            # Try to get standard PyTorch activations (nn.Module versions if they exist)
            if hasattr(torch.nn, name) and isinstance(getattr(torch.nn, name), type) and issubclass(getattr(torch.nn, name), torch.nn.Module):
                 return getattr(torch.nn, name)() # e.g. torch.nn.ReLU()
            # Try functional versions, wrapped for consistency if needed
            elif hasattr(F, name.lower()):
                class FunctionalWrapper(torch.nn.Module):
                    def __init__(self, func):
                        super().__init__()
                        self.activation_func = func
                    def forward(self, x):
                        return self.activation_func(x)
                return FunctionalWrapper(getattr(F, name.lower()))
            else:
                raise ValueError(
                    f"PyTorch activation '{name}' not found in 'actix' package or PyTorch standard activations (nn or nn.functional)."
                )
    else:
        raise ValueError(f"Unsupported framework: {framework}. Choose 'tensorflow' or 'pytorch'.")


# --- Exports ---
from .utils import plot_activation, plot_derivative

__all__ = ['get_activation', 'plot_activation', 'plot_derivative']
__version__ = "0.5.0"

if _TF_AVAILABLE:
    for act_name, act_class in tf_activations_map.items():
        globals()[act_name] = act_class
        if act_name not in __all__:
            __all__.append(act_name)

if _TORCH_AVAILABLE:
    for act_name, act_class in torch_activations_map.items():
        # Suffix PyTorch classes to avoid name collision if a TF class with the same name exists
        export_name = act_name + "Torch"
        globals()[export_name] = act_class
        if export_name not in __all__:
            __all__.append(export_name)

# Ensure unique list if any manual additions caused duplicates
__all__ = sorted(list(set(__all__)))
