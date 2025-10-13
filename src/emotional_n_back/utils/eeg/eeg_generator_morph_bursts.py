
import numpy as np
from numpy.random import default_rng

def gaussian_peak(f, mu, sigma, gain):
    return gain * np.exp(-0.5 * ((f - mu) / sigma)**2)

def target_amplitude_spectrum(f, pink_exponent=1.0, peaks=None, noise_floor=1e-6):
    A = 1.0 / np.maximum(f, noise_floor)**pink_exponent
    if peaks:
        for p in peaks:
            A += gaussian_peak(f, p["mu"], p["sigma"], p["gain"])
    A[0] = A[1]
    return A

def synthesize_from_spectrum(A_pos, rng, phases=None):
    n_pos = len(A_pos)
    if n_pos < 2:
        raise ValueError("A_pos must include at least DC and one positive frequency bin.")
    if phases is None:
        phases = rng.uniform(0, 2*np.pi, size=n_pos-2)
    complex_pos = np.zeros(n_pos, dtype=np.complex128)
    complex_pos[0] = A_pos[0] + 0j
    complex_pos[1:-1] = A_pos[1:-1] * np.exp(1j * phases)
    complex_pos[-1] = A_pos[-1] + 0j
    complex_full = np.concatenate([complex_pos, np.conj(complex_pos[-2:0:-1])])
    x = np.fft.ifft(complex_full).real
    return x, phases

def normalize_time_std(x):
    s = np.std(x)
    return x/s if s > 0 else x

def make_variant_spectrum(f_pos, pink_exponent=1.0, theta=False, alpha=True,
                          theta_gain=0.2, alpha_gain=1.0,
                          theta_mu=5.5, theta_sigma=1.0,
                          alpha_mu=10.0, alpha_sigma=1.5):
    peaks = []
    if theta:
        peaks.append({"mu": theta_mu, "sigma": theta_sigma, "gain": theta_gain})
    if alpha:
        peaks.append({"mu": alpha_mu, "sigma": alpha_sigma, "gain": alpha_gain})
    return target_amplitude_spectrum(f_pos, pink_exponent=pink_exponent, peaks=peaks)

def interpolate_weights(t, waypoints):
    times = np.array([wp[0] for wp in waypoints], dtype=float)
    weights = np.stack([np.array(wp[1], dtype=float) for wp in waypoints], axis=0)
    K = weights.shape[1]
    weights = weights / weights.sum(axis=1, keepdims=True)
    W = np.zeros((K, len(t)))
    for k in range(K):
        W[k] = np.interp(t, times, weights[:, k])
    W /= W.sum(axis=0, keepdims=True)
    return W

def generate_eeg_variants(fs=256, duration=20.0, seed=0, variant_specs=None, pink_exponent=1.0):
    rng = default_rng(seed)
    n = int(np.round(fs * duration))
    if n % 2 == 1:
        n += 1
    f_pos = np.fft.rfftfreq(n, d=1/fs)
    if variant_specs is None:
        variant_specs = [
            dict(pink_exponent=pink_exponent, theta=False, alpha=True, alpha_gain=1.4),
            dict(pink_exponent=pink_exponent, theta=True, alpha=True, theta_gain=0.35, alpha_gain=0.9),
            dict(pink_exponent=pink_exponent, theta=False, alpha=False),
        ]
    shared_phases = rng.uniform(0, 2*np.pi, size=len(f_pos)-2)
    A_list, X = [], []
    for spec in variant_specs:
        A = make_variant_spectrum(f_pos, **spec)
        x_tmp, _ = synthesize_from_spectrum(A, rng, phases=shared_phases)
        scale = np.std(x_tmp)
        if scale > 0:
            A = A / scale
        x, _ = synthesize_from_spectrum(A, rng, phases=shared_phases)
        A_list.append(A)
        X.append(normalize_time_std(x))
    X = np.stack(X, axis=0)
    t = np.arange(int(np.round(fs * duration))) / fs
    return t, X, f_pos, A_list, shared_phases

def morph_variants_over_time(t, X, waypoints):
    W = interpolate_weights(t, waypoints)
    x = np.sum(W * X, axis=0)
    return x, W

def add_random_bursts(
    x, fs, seed=None,
    rate_per_sec=0.4,
    bands=(("theta", 4, 7), ("alpha", 8, 12)),
    dur_range=(0.2, 1.0),
    amp_range=(0.2, 1.0),
    phase_random=True,
):
    rng = default_rng(seed)
    n = len(x)
    T = n / fs
    n_bursts = rng.poisson(rate_per_sec * T)
    t = np.arange(n) / fs
    y = np.zeros_like(x)
    x_std = np.std(x) if np.std(x) > 0 else 1.0
    band_freqs = [(b[1], b[2]) for b in bands]
    for _ in range(n_bursts):
        band_idx = rng.integers(0, len(bands))
        f0 = rng.uniform(*band_freqs[band_idx])
        center = rng.uniform(0.0, T)
        dur = rng.uniform(*dur_range)
        sigma = dur / 3.0
        A = rng.uniform(*amp_range) * x_std
        phi = rng.uniform(0, 2*np.pi) if phase_random else 0.0
        env = np.exp(-0.5 * ((t - center)/sigma)**2)
        carrier = np.cos(2*np.pi*f0*(t - center) + phi)
        y += A * env * carrier
    return x + y
