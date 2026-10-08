"""Differentiable geometric MTF estimate from traced ray intercepts.

Frequency, field and wavelengths are configuration controlled. This model
does not include diffraction or DMD pixel response; it is not measured MTF.
"""

import math

import torch


def ray_fourier_mtf(lens, settings):
    """直接对有效光线的像面落点求傅里叶幅值，保留几何参数梯度。

    对非相干几何 PSF，频率 f 处 OTF 是落点 exp(-2πifx) 的平均。
    多波长先合成复数 OTF 再取模，避免分波长取模掩盖色差。
    返回各视场、子午(T=y)和弧矢(S=x)的代理 MTF。
    """
    frequency = float(settings["frequency_lp_per_mm"])
    diagonal = float(settings["screen_diagonal_inch"])
    aspect_w, aspect_h = settings["screen_aspect_ratio"]
    screen_width = diagonal * 25.4 * aspect_w / math.hypot(aspect_w, aspect_h)
    distance = screen_width * float(settings["throw_ratio"])
    field_fractions = settings["field_fractions"]
    waves = settings["wavelengths_nm"]
    weights = settings["wavelength_weights"]
    if len(waves) != len(weights) or not waves:
        raise ValueError("MTF wavelengths and weights must have equal nonzero length")
    spp = int(settings["rays_per_wavelength"])
    if spp < 2:
        raise ValueError("MTF rays_per_wavelength must be at least 2")
    points = torch.tensor([[0.0, -float(f), -distance] for f in field_fractions],
                          device=lens.device, dtype=lens.dtype)
    scale = lens.calc_scale(points[:, 2])
    sensor_w, sensor_h = lens.sensor_size
    object_points = torch.stack([points[:, 0] * scale * sensor_w / 2,
                                 points[:, 1] * scale * sensor_h / 2,
                                 points[:, 2]], dim=-1)
    total_t = total_s = total_valid = None
    for wavelength, weight in zip(waves, weights):
        ray = lens.sample_from_points(object_points, num_rays=spp, wvln=float(wavelength) / 1000)
        ray = lens.trace2sensor(ray)
        valid = ray.valid.to(lens.dtype) * float(weight)
        # 使用全波段共同原点；复数相位保留不同波长的像移。
        phase_t = 2 * math.pi * frequency * ray.o[..., 1]
        phase_s = 2 * math.pi * frequency * ray.o[..., 0]
        t = torch.complex((valid * torch.cos(phase_t)).sum(-1),
                          (valid * torch.sin(phase_t)).sum(-1))
        s = torch.complex((valid * torch.cos(phase_s)).sum(-1),
                          (valid * torch.sin(phase_s)).sum(-1))
        total_t = t if total_t is None else total_t + t
        total_s = s if total_s is None else total_s + s
        count = valid.sum(-1)
        total_valid = count if total_valid is None else total_valid + count
    denominator = total_valid.clamp_min(1e-6)
    # 当完全没有有效光线时显式给出零，并由外层正则项抑制退化结构。
    return torch.stack([torch.abs(total_t) / denominator,
                        torch.abs(total_s) / denominator], dim=-1)


def mtf_shortfall(mtf, minimum, sharpness=30.0):
    """平滑近似最差视场约束；低于目标时惩罚更大。"""
    return torch.nn.functional.softplus((float(minimum) - mtf) * sharpness).mean() / sharpness
