# actix/activations_torch.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Functions for Torch Activations ---

def torch_lambertw_principal(z, iterations=8):
    """Computes the principal branch of the Lambert W function for PyTorch."""
    w = torch.where(z < 1.0, z, torch.log(z + 1e-38))
    w = torch.clamp(w, min=0.0)
    for _ in range(iterations):
        ew = torch.exp(w)
        w_ew_minus_z = w * ew - z
        denominator = ew * (w + 1.0) + 1e-20
        delta_w = w_ew_minus_z / denominator
        w = w - delta_w
        w = torch.clamp(w, min=0.0)
    return w

def torch_ellipj_cn(u, m, num_terms=4):
    """Computes the Jacobi elliptic function cn(u,m) for PyTorch."""
    u_sq = torch.square(u)
    cn_val = torch.ones_like(u)
    if num_terms > 1:
        term1_val = -u_sq / 2.0
        cn_val = cn_val + term1_val
    if num_terms > 2:
        u_4 = u_sq * u_sq
        term2_val = (u_4 / 24.0) * (1.0 + 4.0 * m)
        cn_val = cn_val + term2_val
    if num_terms > 3:
        u_6 = u_4 * u_sq
        term3_val = -(u_6 / 720.0) * (1.0 + 44.0 * m + 16.0 * torch.square(m))
        cn_val = cn_val + term3_val
    cn_val = torch.clamp(cn_val, -1.0, 1.0)
    return cn_val

# --- Parametric Activation Functions (PyTorch Modules) ---

class OptimATorch(nn.Module):
    """
    OptimA: An 'Optimal Activation' function with trainable parameters for PyTorch.
    f(x) = alpha * tanh(beta * x) + gamma * softplus(delta * x) * sigmoid(lambda_ * x)
    """
    def __init__(self):
        super(OptimATorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.full((1,), 0.5))
        self.gamma_param = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.full((1,), 0.5))
        self.lambda_param = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * torch.tanh(self.beta * x)
        term2 = self.gamma_param * F.softplus(self.delta * x) * torch.sigmoid(self.lambda_param * x)
        return term1 + term2

class ParametricPolyTanhTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x^2 + gamma * x + delta)"""
    def __init__(self):
        super(ParametricPolyTanhTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta_param = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(self.beta * torch.square(x) + self.gamma * x + self.delta_param)

class AdaptiveRationalSoftsignTorch(nn.Module):
    """f(x) = (alpha * x) / (1 + |beta * x|^gamma)"""
    def __init__(self):
        super(AdaptiveRationalSoftsignTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.full((1,), 2.0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.alpha * x) / (1.0 + torch.pow(torch.abs(self.beta * x), self.gamma_param))

class OptimXTemporalTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x) + gamma * sigmoid(delta * x)"""
    def __init__(self):
        super(OptimXTemporalTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.full((1,), 0.5))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(self.beta * x) + self.gamma_param * torch.sigmoid(self.delta * x)

class ParametricGaussianActivationTorch(nn.Module):
    """f(x) = alpha * x * exp(-beta * x^2)"""
    def __init__(self):
        super(ParametricGaussianActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1)) # Beta should be > 0; consider constraints if necessary
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x * torch.exp(-self.beta * torch.square(x))

class LearnableFourierActivationTorch(nn.Module):
    """f(x) = alpha * sin(beta * x + gamma_shift) + delta * cos(lambda_param * x + phi)"""
    def __init__(self):
        super(LearnableFourierActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_shift = nn.Parameter(torch.zeros(1))
        self.delta_param = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.ones(1))
        self.phi = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * torch.sin(self.beta * x + self.gamma_shift)
        term2 = self.delta_param * torch.cos(self.lambda_param * x + self.phi)
        return term1 + term2

class A_ELuCTorch(nn.Module):
    """f(x) = alpha * ELU(beta * x) + gamma * x * sigmoid(delta * x)"""
    def __init__(self):
        super(A_ELuCTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.full((1,), 0.5))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * F.elu(self.beta * x)
        term2 = self.gamma_param * x * torch.sigmoid(self.delta * x)
        return term1 + term2

class ParametricSmoothStepTorch(nn.Module):
    """f(x) = alpha * sigmoid(beta_slope*(x - gamma_shift)) - alpha * sigmoid(delta_slope_param*(x + mu_shift))"""
    def __init__(self):
        super(ParametricSmoothStepTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta_slope = nn.Parameter(torch.ones(1))
        self.gamma_shift = nn.Parameter(torch.zeros(1))
        self.delta_slope_param = nn.Parameter(torch.ones(1))
        self.mu_shift = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * torch.sigmoid(self.beta_slope * (x - self.gamma_shift))
        term2 = self.alpha * torch.sigmoid(self.delta_slope_param * (x + self.mu_shift))
        return term1 - term2

class AdaptiveBiHyperbolicTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x) + (1-alpha) * tanh^3(gamma_param * x)"""
    def __init__(self):
        super(AdaptiveBiHyperbolicTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * torch.tanh(self.beta * x)
        term2 = (1.0 - self.alpha) * torch.pow(torch.tanh(self.gamma_param * x), 3)
        return term1 + term2

class ParametricLogishTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x)"""
    def __init__(self):
        super(ParametricLogishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x * torch.sigmoid(self.beta * x)

class AdaptSigmoidReLUTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x) + gamma_param * ReLU(delta * x)"""
    def __init__(self):
        super(AdaptSigmoidReLUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.full((1,), 0.5))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * x * torch.sigmoid(self.beta * x)
        term2 = self.gamma_param * F.relu(self.delta * x)
        return term1 + term2

class ParametricLambertWActivationTorch(nn.Module):
    """f(x) = alpha * x * W(|beta| * exp(gamma * x)) where W is the Lambert W function."""
    def __init__(self):
        super(ParametricLambertWActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg_lambertw = torch.abs(self.beta) * torch.exp(self.gamma * x)
        lambertw_val = torch_lambertw_principal(arg_lambertw)
        return self.alpha * x * lambertw_val

class AdaptiveHyperbolicLogarithmTorch(nn.Module):
    """f(x) = alpha * asinh(beta * x) + gamma * log(|delta| + x^2)"""
    def __init__(self):
        super(AdaptiveHyperbolicLogarithmTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.full((1,), 0.5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * torch.asinh(self.beta * x)
        term2 = self.gamma * torch.log(torch.abs(self.delta) + torch.square(x) + 1e-7)
        return term1 + term2

class ParametricGeneralizedGompertzActivationTorch(nn.Module):
    """f(x) = alpha * exp(-beta * exp(-gamma * x)) - delta"""
    def __init__(self):
        super(ParametricGeneralizedGompertzActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.exp(-self.beta * torch.exp(-self.gamma * x)) - self.delta

class ComplexHarmonicActivationTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x) + gamma * sin(delta * x^2 + lambda)"""
    def __init__(self):
        super(ComplexHarmonicActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * torch.tanh(self.beta * x)
        term2 = self.gamma * torch.sin(self.delta * torch.square(x) + self.lambda_param)
        return term1 + term2

class WeibullSoftplusActivationTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * (x - gamma)) + delta * (1 - exp(-|lambda| * |x|^|mu|))"""
    def __init__(self):
        super(WeibullSoftplusActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.ones(1))
        self.mu = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = self.alpha * x * torch.sigmoid(self.beta * (x - self.gamma))
        weibull_exponent = torch.abs(self.lambda_param) * torch.pow(torch.abs(x) + 1e-7, torch.abs(self.mu))
        term2 = self.delta * (1.0 - torch.exp(-weibull_exponent))
        return term1 + term2

class AdaptiveErfSwishTorch(nn.Module):
    """f(x) = alpha * x * erf(beta * x) * sigmoid(gamma * x)"""
    def __init__(self):
        super(AdaptiveErfSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x * torch.erf(self.beta * x) * torch.sigmoid(self.gamma * x)

class ParametricBetaSoftsignTorch(nn.Module):
    """f(x) = alpha * sign(x) * (|x|^|beta|) / (1 + |x|^|gamma|)"""
    def __init__(self):
        super(ParametricBetaSoftsignTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        pow_beta = torch.pow(abs_x, torch.abs(self.beta))
        pow_gamma = torch.pow(abs_x, torch.abs(self.gamma))
        return self.alpha * (pow_beta / (1.0 + pow_gamma + 1e-7)) * torch.sign(x)

class ParametricArcSinhGateTorch(nn.Module):
    """f(x) = alpha * x * asinh(beta * x)"""
    def __init__(self):
        super(ParametricArcSinhGateTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x * torch.asinh(self.beta * x)

class GeneralizedAlphaSigmoidTorch(nn.Module):
    """f(x) = (alpha * x) / (1 + |beta * x|^|gamma|)^(1/|delta|)"""
    def __init__(self):
        super(GeneralizedAlphaSigmoidTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_beta_x = torch.abs(self.beta * x)
        pow_gamma = torch.pow(abs_beta_x, torch.abs(self.gamma))
        denominator_base = 1.0 + pow_gamma
        inv_delta = 1.0 / (torch.abs(self.delta) + 1e-7)
        denominator = torch.pow(denominator_base, inv_delta)
        return (self.alpha * x) / (denominator + 1e-7)

class EllipticGaussianActivationTorch(nn.Module):
    """f(x) = x * exp(-cn(x, m)) where cn is the Jacobi elliptic function."""
    def __init__(self):
        super(EllipticGaussianActivationTorch, self).__init__()
        self.m_param = nn.Parameter(torch.full((1,), 0.5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_clamped = torch.clamp(self.m_param, 0.0, 1.0)
        cn_val = torch_ellipj_cn(x, m_clamped)
        return x * torch.exp(-cn_val)

class ParametricTanhSwishTorch(nn.Module):
    """f(x) = alpha * x * tanh(beta * x) * sigmoid(gamma * x)"""
    def __init__(self):
        super(ParametricTanhSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x * torch.tanh(self.beta * x) * torch.sigmoid(self.gamma * x)

class GeneralizedHarmonicSwishTorch(nn.Module):
    """f(x) = alpha * x * sin(beta * x^2 + gamma) * sigmoid(delta * x)"""
    def __init__(self):
        super(GeneralizedHarmonicSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        harmonic_part = torch.sin(self.beta * torch.square(x) + self.gamma)
        swish_gate = x * torch.sigmoid(self.delta * x)
        return self.alpha * swish_gate * harmonic_part

class A_STReLUTorch(nn.Module):
    """f(x) = alpha * ReLU(x) + beta * x * sigmoid(gamma * x) + delta * tanh(lambda * x)"""
    def __init__(self):
        super(A_STReLUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.33))
        self.beta = nn.Parameter(torch.full((1,), 0.33))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.full((1,), 0.33))
        self.lambda_param = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_part = self.alpha * F.relu(x)
        swish_part = self.beta * x * torch.sigmoid(self.gamma * x)
        tanh_part = self.delta * torch.tanh(self.lambda_param * x)
        return relu_part + swish_part + tanh_part

class ETUTorch(nn.Module):
    """ExponentialTanhUnit: f(x) = alpha * tanh(beta * x) * exp(-gamma * x^2)"""
    def __init__(self):
        super(ETUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(self.beta * x) * torch.exp(-self.gamma * torch.square(x))

class PMGLUTorch(nn.Module):
    """Parametric Multi-Gated Linear Unit: f(x) = (alpha * x + beta) * sigmoid(gamma * x + delta) * tanh(lambda * x)"""
    def __init__(self):
        super(PMGLUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.zeros(1))
        self.lambda_param = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.alpha * x + self.beta
        sigmoid_gate = torch.sigmoid(self.gamma * x + self.delta)
        tanh_gate = torch.tanh(self.lambda_param * x)
        return linear_part * sigmoid_gate * tanh_gate

class GPOSoftTorch(nn.Module):
    """Generalized Parametric Oscillatory Softplus: f(x) = alpha * softplus(beta * x) + gamma * sin(delta * x + lambda)"""
    def __init__(self):
        super(GPOSoftTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softplus_part = self.alpha * F.softplus(self.beta * x)
        oscillatory_part = self.gamma * torch.sin(self.delta * x + self.lambda_param)
        return softplus_part + oscillatory_part

class SHLUTorch(nn.Module):
    """Sigmoid-Harmonic Linear Unit: f(x)= (alpha * x) * sigmoid(beta * x) + gamma * cos(delta * x^2 + lambda)"""
    def __init__(self):
        super(SHLUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        harmonic_part = self.gamma * torch.cos(self.delta * torch.square(x) + self.lambda_param)
        return swish_part + harmonic_part

class GaussSwishTorch(nn.Module):
    """Gaussian Parametric Swish: f(x) = (alpha * x) * sigmoid(beta * x) * exp(-gamma * x^2)"""
    def __init__(self):
        super(GaussSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        gaussian_part = torch.exp(-self.gamma * torch.square(x))
        return swish_part * gaussian_part

class ATanSigUTorch(nn.Module):
    """Adaptive ArcTanSigmoid Unit: f(x) = alpha * arctan(beta * x) + gamma * x * sigmoid(delta * x)"""
    def __init__(self):
        super(ATanSigUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arctan_part = self.alpha * torch.atan(self.beta * x)
        swish_part = self.gamma * x * torch.sigmoid(self.delta * x)
        return arctan_part + swish_part

class PAPGTorch(nn.Module):
    """Parametric Adaptive Polynomial Gate: f(x) = (alpha * x + beta * x^3) / (1 + |gamma * x|^delta) + lambda * x * sigmoid(mu * x)"""
    def __init__(self):
        super(PAPGTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.ones(1))
        self.mu = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = self.alpha * x + self.beta * torch.pow(x, 3)
        denominator = 1.0 + torch.pow(torch.abs(self.gamma * x), self.delta)
        poly_gate_part = numerator / (denominator + 1e-7)
        swish_part = self.lambda_param * x * torch.sigmoid(self.mu * x)
        return poly_gate_part + swish_part

class SwishLogTanhTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x) + gamma * tanh(softplus(delta * x))"""
    def __init__(self):
        super(SwishLogTanhTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        log_tanh_part = self.gamma * torch.tanh(F.softplus(self.delta * x))
        return swish_part + log_tanh_part

class ArcGaLUTorch(nn.Module):
    """f(x) = alpha * arctan(beta * x) + gamma * x * sigmoid(delta * x + lambda)"""
    def __init__(self):
        super(ArcGaLUTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arctan_part = self.alpha * torch.atan(self.beta * x)
        gated_linear_part = self.gamma * x * torch.sigmoid(self.delta * x + self.lambda_param)
        return arctan_part + gated_linear_part

class ParametricHyperbolicQuadraticActivationTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x^2 + gamma * x) + delta * x * sigmoid(lambda * x)"""
    def __init__(self):
        super(ParametricHyperbolicQuadraticActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_part = self.alpha * torch.tanh(self.beta * torch.square(x) + self.gamma * x)
        swish_part = self.delta * x * torch.sigmoid(self.lambda_param * x)
        return tanh_part + swish_part

class RootSoftplusTorch(nn.Module):
    """f(x) = alpha * x + beta * sqrt(softplus(gamma * x))"""
    def __init__(self):
        super(RootSoftplusTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.alpha * x
        root_softplus_part = self.beta * torch.sqrt(F.softplus(self.gamma * x) + 1e-7)
        return linear_part + root_softplus_part

class AdaptiveSinusoidalSoftgateTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x) + gamma * sin(delta * x)"""
    def __init__(self):
        super(AdaptiveSinusoidalSoftgateTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        sin_part = self.gamma * torch.sin(self.delta * x)
        return swish_part + sin_part

class ExpTanhGatedActivationTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x) * sigmoid(gamma * exp(delta * x))"""
    def __init__(self):
        super(ExpTanhGatedActivationTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_part = self.alpha * torch.tanh(self.beta * x)
        exp_gate = torch.sigmoid(self.gamma * torch.exp(self.delta * x))
        return tanh_part * exp_gate

class HybridSinExpUnitTorch(nn.Module):
    """f(x) = alpha * sin(beta * x) + gamma * x * exp(-delta * x^2)"""
    def __init__(self):
        super(HybridSinExpUnitTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sin_part = self.alpha * torch.sin(self.beta * x)
        exp_part = self.gamma * x * torch.exp(-self.delta * torch.square(x))
        return sin_part + exp_part

class ParametricLogarithmicSwishTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x) + gamma * log(1 + |delta| * |x|)"""
    def __init__(self):
        super(ParametricLogarithmicSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        log_part = self.gamma * torch.log(1.0 + torch.abs(self.delta) * torch.abs(x) + 1e-7)
        return swish_part + log_part

class AdaptiveCubicSigmoidTorch(nn.Module):
    """f(x) = alpha * x + beta * x^3 * sigmoid(gamma * x)"""
    def __init__(self):
        super(AdaptiveCubicSigmoidTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.alpha * x
        cubic_part = self.beta * torch.pow(x, 3) * torch.sigmoid(self.gamma * x)
        return linear_part + cubic_part

class SmoothedAbsoluteGatedUnitTorch(nn.Module):
    """f(x) = alpha * |x| * sigmoid(beta * x) + gamma * x"""
    def __init__(self):
        super(SmoothedAbsoluteGatedUnitTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_abs_part = self.alpha * torch.abs(x) * torch.sigmoid(self.beta * x)
        linear_part = self.gamma * x
        return gated_abs_part + linear_part

class GaussianTanhHarmonicUnitTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x) + gamma * exp(-delta * (x - lambda)^2)"""
    def __init__(self):
        super(GaussianTanhHarmonicUnitTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
        self.lambda_param = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_part = self.alpha * torch.tanh(self.beta * x)
        gaussian_part = self.gamma * torch.exp(-self.delta * torch.square(x - self.lambda_param))
        return tanh_part + gaussian_part

class SymmetricParametricRationalSigmoidTorch(nn.Module):
    """f(x) = (alpha * x) / (1 + |beta * x^2|^gamma)"""
    def __init__(self):
        super(SymmetricParametricRationalSigmoidTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denominator = 1.0 + torch.pow(torch.abs(self.beta * torch.square(x)), self.gamma)
        return (self.alpha * x) / (denominator + 1e-7)

class AdaptivePolynomialSwishTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x) + gamma * x^2 * sigmoid(delta * x)"""
    def __init__(self):
        super(AdaptivePolynomialSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        poly_swish_part = self.gamma * torch.square(x) * torch.sigmoid(self.delta * x)
        return swish_part + poly_swish_part

class LogSigmoidGatedEluTorch(nn.Module):
    """f(x) = alpha * ELU(beta * x) * softplus(gamma * x) * sigmoid(delta * x)"""
    def __init__(self):
        super(LogSigmoidGatedEluTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        elu_part = F.elu(self.beta * x)
        log_gate = F.softplus(self.gamma * x)
        sig_gate = torch.sigmoid(self.delta * x)
        return self.alpha * elu_part * log_gate * sig_gate

class AdaptiveBipolarExponentialUnitTorch(nn.Module):
    """f(x) = alpha * x + beta * x * exp(-gamma * x^2)"""
    def __init__(self):
        super(AdaptiveBipolarExponentialUnitTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.alpha * x
        exp_part = self.beta * x * torch.exp(-self.gamma * torch.square(x))
        return linear_part + exp_part

class ParametricHyperGaussianGateTorch(nn.Module):
    """f(x) = alpha * x * exp(-beta * |x|^gamma)"""
    def __init__(self):
        super(ParametricHyperGaussianGateTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.exp(-torch.abs(self.beta) * torch.pow(torch.abs(x), torch.abs(self.gamma)))
        return self.alpha * x * gate

class TanhGatedArcsinhLinearUnitTorch(nn.Module):
    """f(x) = alpha * x * tanh(beta * asinh(gamma * x))"""
    def __init__(self):
        super(TanhGatedArcsinhLinearUnitTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.tanh(self.beta * torch.asinh(self.gamma * x))
        return self.alpha * x * gate

class ParametricOddPowerSwishTorch(nn.Module):
    """f(x) = alpha * x * sigmoid(beta * x) + gamma * x^3"""
    def __init__(self):
        super(ParametricOddPowerSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = self.alpha * x * torch.sigmoid(self.beta * x)
        power_part = self.gamma * torch.pow(x, 3)
        return swish_part + power_part

class AdaptiveLinearLogTanhTorch(nn.Module):
    """f(x) = alpha * x + beta * tanh(gamma * softplus(delta * x))"""
    def __init__(self):
        super(AdaptiveLinearLogTanhTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.alpha * x
        log_tanh_part = self.beta * torch.tanh(self.gamma * F.softplus(self.delta * x))
        return linear_part + log_tanh_part

class AdaptiveArcTanSwishTorch(nn.Module):
    """f(x) = α·arctan(β·x) + γ·x·sigmoid(δ·x) + ε·tanh(ζ·x)"""
    def __init__(self):
        super(AdaptiveArcTanSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.5))
        self.delta = nn.Parameter(torch.ones(1))
        self.epsilon = nn.Parameter(torch.full((1,), 0.3))
        self.zeta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arctan_part = self.alpha * torch.atan(self.beta * x)
        swish_part = self.gamma * x * torch.sigmoid(self.delta * x)
        tanh_part = self.epsilon * torch.tanh(self.zeta * x)
        return arctan_part + swish_part + tanh_part

class StabilizedHarmonicTorch(nn.Module):
    """f(x) = α·tanh(β·x) + γ·sin(δ·tanh(ε·x)) + ζ·x·sigmoid(η·x)"""
    def __init__(self):
        super(StabilizedHarmonicTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.2))
        self.delta = nn.Parameter(torch.ones(1))
        self.epsilon = nn.Parameter(torch.ones(1))
        self.zeta = nn.Parameter(torch.full((1,), 0.5))
        self.eta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_part = self.alpha * torch.tanh(self.beta * x)
        sin_part = self.gamma * torch.sin(self.delta * torch.tanh(self.epsilon * x))
        swish_part = self.zeta * x * torch.sigmoid(self.eta * x)
        return tanh_part + sin_part + swish_part

class RationalSwishTorch(nn.Module):
    """f(x) = (α·x·sigmoid(β·x)) / (1 + |γ·x|^δ) + ε·arctan(ζ·x)"""
    def __init__(self):
        super(RationalSwishTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.full((1,), 1.5))
        self.epsilon = nn.Parameter(torch.full((1,), 0.3))
        self.zeta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_numerator = self.alpha * x * torch.sigmoid(self.beta * x)
        rational_denominator = 1.0 + torch.pow(torch.abs(self.gamma * x) + 1e-7, torch.abs(self.delta))
        rational_part = swish_numerator / (rational_denominator + 1e-7)
        arctan_part = self.epsilon * torch.atan(self.zeta * x)
        return rational_part + arctan_part

class AdaptiveGatedUnitTorch(nn.Module):
    """f(x) = α·x·sigmoid(β·x) + γ·x·tanh(δ·x) + ε·arctan(ζ·x)"""
    def __init__(self):
        super(AdaptiveGatedUnitTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.4))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.4))
        self.delta = nn.Parameter(torch.ones(1))
        self.epsilon = nn.Parameter(torch.full((1,), 0.2))
        self.zeta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_gate = self.alpha * x * torch.sigmoid(self.beta * x)
        tanh_gate = self.gamma * x * torch.tanh(self.delta * x)
        arctan_part = self.epsilon * torch.atan(self.zeta * x)
        return sigmoid_gate + tanh_gate + arctan_part

class ExponentialArcTanTorch(nn.Module):
    """f(x) = α·arctan(β·x) + γ·x·exp(-δ·|x|) + ε·sigmoid(ζ·x)"""
    def __init__(self):
        super(ExponentialArcTanTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.5))
        self.delta = nn.Parameter(torch.full((1,), 0.1))
        self.epsilon = nn.Parameter(torch.full((1,), 0.3))
        self.zeta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arctan_part = self.alpha * torch.atan(self.beta * x)
        exp_decay = self.gamma * x * torch.exp(-torch.abs(self.delta * x))
        sigmoid_part = self.epsilon * torch.sigmoid(self.zeta * x)
        return arctan_part + exp_decay + sigmoid_part

class OptimQTorch(nn.Module):
    """OptimQ: f(x) = α·arctan(β·x) + γ·x·sigmoid(δ·x) + ε·softplus(ζ·x)·tanh(η·x)"""
    def __init__(self):
        super(OptimQTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.full((1,), 0.6))
        self.delta = nn.Parameter(torch.ones(1))
        self.epsilon = nn.Parameter(torch.full((1,), 0.4))
        self.zeta = nn.Parameter(torch.full((1,), 0.5))
        self.eta = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arctan_part = self.alpha * torch.atan(self.beta * x)
        swish_part = self.gamma * x * torch.sigmoid(self.delta * x)
        gated_softplus = self.epsilon * F.softplus(self.zeta * x) * torch.tanh(self.eta * x)
        return arctan_part + swish_part + gated_softplus

# --- Static Activation Functions (PyTorch Modules for consistency) ---

class SinhGateTorch(nn.Module):
    """f(x) = x * sinh(x)"""
    def __init__(self): super(SinhGateTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x * torch.sinh(x)

class SoftRBFTorch(nn.Module):
    """f(x) = x * exp(-x^2)"""
    def __init__(self): super(SoftRBFTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x * torch.exp(-torch.square(x))

class ATanSigmoidTorch(nn.Module):
    """f(x) = arctan(x) * sigmoid(x)"""
    def __init__(self): super(ATanSigmoidTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.atan(x) * torch.sigmoid(x)

class ExpoSoftTorch(nn.Module):
    """f(x) = softsign(x) * exp(-|x|)"""
    def __init__(self): super(ExpoSoftTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return F.softsign(x) * torch.exp(-torch.abs(x))

class HarmonicTanhTorch(nn.Module):
    """f(x) = tanh(x) + sin(x)"""
    def __init__(self): super(HarmonicTanhTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.tanh(x) + torch.sin(x)

class RationalSoftplusTorch(nn.Module):
    """f(x) = (x * sigmoid(x)) / (0.5 + x * sigmoid(x))"""
    def __init__(self): super(RationalSoftplusTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_x = x * torch.sigmoid(x)
        return swish_x / (0.5 + swish_x + 1e-7) # Added epsilon for numerical stability

class UnifiedSineExpTorch(nn.Module):
    """f(x) = x * sin(exp(-x^2))"""
    def __init__(self): super(UnifiedSineExpTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x * torch.sin(torch.exp(-torch.square(x)))

class SigmoidErfTorch(nn.Module):
    """f(x) = sigmoid(x) * erf(x)"""
    def __init__(self): super(SigmoidErfTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.sigmoid(x) * torch.erf(x)

class LogCoshGateTorch(nn.Module):
    """f(x) = x * log(cosh(x))"""
    def __init__(self): super(LogCoshGateTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add epsilon for numerical stability
        return x * torch.log(torch.cosh(x) + 1e-7)

class TanhArcTorch(nn.Module):
    """f(x) = tanh(x) * arctan(x)"""
    def __init__(self): super(TanhArcTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.tanh(x) * torch.atan(x)

class RiemannianSoftsignActivationTorch(nn.Module):
    """f(x) = (arctan(x) * erf(x)) / (1 + |x|)"""
    def __init__(self): super(RiemannianSoftsignActivationTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.atan(x) * torch.erf(x)
        denominator = 1.0 + torch.abs(x)
        return numerator / (denominator + 1e-7)

class QuantumTanhActivationTorch(nn.Module):
    """f(x) = tanh(x) * exp(-tan(x)^2)"""
    def __init__(self): super(QuantumTanhActivationTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tan_x_squared = torch.square(torch.tan(x))
        return torch.tanh(x) * torch.exp(-tan_x_squared)

class LogExponentialActivationTorch(nn.Module):
    """f(x) = sign(x) * log(1 + exp(|x| - 1/|x|))"""
    def __init__(self): super(LogExponentialActivationTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        abs_x_safe = abs_x + 1e-7
        exponent = abs_x - torch.pow(abs_x_safe, -1.0)
        return torch.sign(x) * torch.log(1.0 + torch.exp(exponent) + 1e-7)

class BipolarGaussianArctanActivationTorch(nn.Module):
    """f(x) = arctan(x) * exp(-x^2)"""
    def __init__(self): super(BipolarGaussianArctanActivationTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x) * torch.exp(-torch.square(x))

class ExpArcTanHarmonicActivationTorch(nn.Module):
    """f(x) = exp(-x^2) * arctan(x) * sin(x)"""
    def __init__(self): super(ExpArcTanHarmonicActivationTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.square(x)) * torch.atan(x) * torch.sin(x)

class LogisticWActivationTorch(nn.Module):
    """f(x) = x / (1 + exp(-x * W(exp(x)))) where W is the Lambert W function."""
    def __init__(self): super(LogisticWActivationTorch, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_x = torch.exp(x)
        lambertw_exp_x = torch_lambertw_principal(exp_x)
        denominator_arg = -x * lambertw_exp_x
        return x / (1.0 + torch.exp(denominator_arg) + 1e-7)