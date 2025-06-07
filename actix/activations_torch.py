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
        self.gamma_param = nn.Parameter(torch.ones(1)) # Renamed to avoid confusion with other gamma
        self.delta = nn.Parameter(torch.full((1,), 0.5))
        self.lambda_param = nn.Parameter(torch.ones(1)) # lambda is a keyword

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
        self.delta_param = nn.Parameter(torch.zeros(1)) # Renamed to avoid confusion
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(self.beta * torch.square(x) + self.gamma * x + self.delta_param)

class AdaptiveRationalSoftsignTorch(nn.Module):
    """f(x) = (alpha * x) / (1 + |beta * x|^gamma)"""
    def __init__(self):
        super(AdaptiveRationalSoftsignTorch, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.full((1,), 2.0)) # Renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.alpha * x) / (1.0 + torch.pow(torch.abs(self.beta * x), self.gamma_param))

class OptimXTemporalTorch(nn.Module):
    """f(x) = alpha * tanh(beta * x) + gamma * sigmoid(delta * x)"""
    def __init__(self):
        super(OptimXTemporalTorch, self).__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma_param = nn.Parameter(torch.full((1,), 0.5)) # Renamed
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
        self.delta_param = nn.Parameter(torch.ones(1)) # Renamed
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
        self.gamma_param = nn.Parameter(torch.full((1,), 0.5)) # Renamed
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
        # Add epsilon for numerical stability as log(1) (cosh(0)) is 0, and log can be sensitive near 0.
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