# tests/test_activations_tf.py
import pytest
import numpy as np

try:
    import tensorflow as tf
    from actix.activations_tf import (
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
        EllipticGaussianActivation, ExpArcTanHarmonicActivation, LogisticWActivation
    )
    from actix import get_activation
    tf_available = True
except ImportError:
    tf_available = False

ALL_TF_ACTIVATION_CLASSES = [
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
    EllipticGaussianActivation, ExpArcTanHarmonicActivation, LogisticWActivation
]

@pytest.mark.skipif(not tf_available, reason="TensorFlow not installed")
@pytest.mark.parametrize("activation_class", ALL_TF_ACTIVATION_CLASSES)
def test_tf_activation_output_properties(activation_class):
    """Tests that the TF activation returns a tensor of correct shape and dtype."""
    layer = activation_class()
    test_input_np = np.random.rand(10, 5).astype(np.float32)
    test_input_tf = tf.constant(test_input_np)
    
    try:
        output = layer(test_input_tf)
        assert output.shape == test_input_tf.shape, f"{activation_class.__name__} changed input shape."
        assert output.dtype == test_input_tf.dtype, f"{activation_class.__name__} changed data type."
        assert not tf.reduce_any(tf.math.is_nan(output)).numpy(), f"{activation_class.__name__} produced NaN output."
    except Exception as e:
        pytest.fail(f"Test for {activation_class.__name__} failed during forward pass: {e}")

@pytest.mark.skipif(not tf_available, reason="TensorFlow not installed")
def test_get_activation_tf_custom():
    """Tests the get_activation function for a custom TF activation."""
    act = get_activation('ParametricLambertWActivation', framework='tf')
    assert isinstance(act, ParametricLambertWActivation)

@pytest.mark.skipif(not tf_available, reason="TensorFlow not installed")
def test_get_activation_tf_standard():
    """Tests the get_activation function for a standard Keras activation."""
    act = get_activation('relu', framework='tf')
    # Keras returns the function itself, not a layer instance
    assert act == tf.keras.activations.relu
