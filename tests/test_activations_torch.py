# tests/test_activations_torch.py

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    from actix.activations_torch import (
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
    from actix import get_activation
    torch_available = True
except ImportError:
    torch_available = False

ALL_TORCH_ACTIVATION_CLASSES = [
    OptimATorch, ParametricPolyTanhTorch, AdaptiveRationalSoftsignTorch, OptimXTemporalTorch,
    ParametricGaussianActivationTorch, LearnableFourierActivationTorch, A_ELuCTorch,
    ParametricSmoothStepTorch, AdaptiveBiHyperbolicTorch, ParametricLogishTorch, AdaptSigmoidReLUTorch,
    SinhGateTorch, SoftRBFTorch, ATanSigmoidTorch, ExpoSoftTorch, HarmonicTanhTorch, RationalSoftplusTorch,
    UnifiedSineExpTorch, SigmoidErfTorch, LogCoshGateTorch, TanhArcTorch,
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
]

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
@pytest.mark.parametrize("activation_class", ALL_TORCH_ACTIVATION_CLASSES)
def test_torch_activation_output_properties(activation_class):
    """Tests that the PyTorch activation returns a tensor of correct shape and dtype."""
    module = activation_class()
    test_input = torch.randn(10, 5, dtype=torch.float32)
    
    try:
        output = module(test_input)
        assert output.shape == test_input.shape, f"{activation_class.__name__} changed input shape."
        assert output.dtype == test_input.dtype, f"{activation_class.__name__} changed data type."
        assert not torch.isnan(output).any(), f"{activation_class.__name__} produced NaN output."
    except Exception as e:
        pytest.fail(f"Test for {activation_class.__name__} failed during forward pass: {e}")

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
def test_get_activation_torch_custom():
    """Tests the get_activation function for a custom Torch activation."""
    act = get_activation('OptimQ', framework='torch')
    assert isinstance(act, OptimQTorch)

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
def test_get_activation_torch_standard():
    """Tests the get_activation function for a standard PyTorch activation."""
    act = get_activation('ReLU', framework='torch')
    assert isinstance(act, nn.ReLU)