from __future__ import annotations

import torch
from isaaclab.utils import configclass
from .geodetic_utils import *

@configclass
class ImuNoiseCfg:
    """Configuration for Noise of an Inertial Measurement Unit (IMU) sensor."""

    add_noise: bool = False
    """Add noise to the IMU sensor."""

    # use_lat: bool = False
    # """Use latitude for gravity and coriolis generation."""

    # lat: float = 0.0
    # """Latitude (radians)."""

    g_std: tuple[float, float, float] = (0., 0., 0.)
    """1x3 gyros standard deviations (radians/s)."""

    a_std: tuple[float, float, float] = (0., 0., 0.)
    """1x3 accrs standard deviations (m/s^2)."""

    gb_sta: tuple[float, float, float] = (0., 0., 0.)
    """1x3 gyros static biases or turn-on biases (radians/s)."""

    ab_sta: tuple[float, float, float] = (0., 0., 0.)
    """1x3 accrs static biases or turn-on biases (m/s^2)."""

    gb_dyn: tuple[float, float, float] = (0., 0., 0.)
    """1x3 gyros dynamic biases or bias instabilities (radians/s)."""

    ab_dyn: tuple[float, float, float] = (0., 0., 0.)
    """1x3 accrs dynamic biases or bias instabilities (m/s^2)."""

    gb_corr: tuple[float, float, float] = (0., 0., 0.)
    """1x3 gyros correlation times (seconds)."""

    ab_corr: tuple[float, float, float] = (0., 0., 0.)
    """1x3 accrs correlation times (seconds)."""

    # gb_psd: tuple[float, float, float] = (0., 0., 0.)
    """1x3 gyros dynamic biases root-PSD (rad/s/root-Hz)."""

    # ab_psd: tuple[float, float, float] = (0., 0., 0.)
    """1x3 accrs dynamic biases root-PSD (m/s^2/root-Hz)."""

    # arw: tuple[float, float, float] = (0., 0., 0.)
    """1x3 angle random walks (rad/s/root-Hz)."""

    arrw_std: tuple[float, float, float] = (0., 0., 0.)
    """1x3 angle rate random walks (rad/s^2)."""

    # vrw: tuple[float, float, float] = (0., 0., 0.)
    """1x3 velocity random walks (m/s^2/root-Hz)"""

    vrrw_std: tuple[float, float, float] = (0., 0., 0.)
    """velocity rate random walks (m/s^3)."""

def gravity(lat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Calculates the gravity vector in the navigation frame.

    Args:
        lat (torch.Tensor): Mx1 tensor of latitude in radians.
        h (torch.Tensor): Mx1 tensor of altitude in meters.

    Returns:
        torch.Tensor: Mx3 tensor of gravity vector in the nav-frame [m/s^2, m/s^2, m/s^2].
    """
    # Ensure input tensors are of the same shape
    if lat.shape != h.shape:
        raise ValueError("Latitude and altitude tensors must have the same shape.")
    
    # Initialize output tensor
    M = lat.size(0)
    gn = torch.zeros((M, 3), dtype=lat.dtype, device=lat.device)

    # Parameters
    RN = 6378137.0              # WGS84 Equatorial radius in meters
    RM = 6356752.31425          # WGS84 Polar radius in meters
    e = 0.0818191908425         # WGS84 eccentricity
    f = 1 / 298.257223563       # WGS84 flattening
    mu = 3.986004418e14         # WGS84 Earth gravitational constant (m^3 s^-2)
    omega_ie_n = 7.292115e-5    # Earth rotation rate (rad/s)

    # Calculate surface gravity using the Somigliana model
    sinl2 = torch.sin(lat) ** 2
    g_0 = 9.7803253359 * (1 + 0.001931853 * sinl2) / torch.sqrt(1 - e**2 * sinl2)

    # Calculate north gravity using
    gn[:, 0] = (-8.08e-9 * h * torch.sin(2 * lat)).squeeze()

    # East gravity is zero
    gn[:, 1] = 0

    # Calculate down gravity using
    term1 = (2 / RN) * (1 + f * (1 - 2 * sinl2) + (omega_ie_n**2 * RN**2 * RM / mu))
    term2 = (3 * h**2) / RN**2
    gn[:, 2] = (g_0 * (1 - term1 * h + term2)).squeeze()

    return gn

def coriolis(lat: torch.Tensor, vel: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Calculates Coriolis forces in the navigation frame.

    Args:
        lat (torch.Tensor): Mx1 tensor of latitude in radians.
        vel (torch.Tensor): Mx3 tensor of NED velocities (m/s).
        h (torch.Tensor): Mx1 tensor of altitude in meters.

    Returns:
        torch.Tensor: Mx3 tensor of Coriolis forces in the navigation frame (m/s^2).
    """
    # Ensure input dimensions are consistent
    if lat.shape[0] != vel.shape[0] or lat.shape[0] != h.shape[0]:
        raise ValueError("Input tensors lat, vel, and h must have the same number of rows.")
    
    cor_n = torch.zeros((lat.shape[0], 3), dtype=lat.dtype, device=lat.device)
    
    # Loop through each row
    for i in range(lat.shape[0]):
        # Calculate transport and earth rates for this row
        omega_en_n = transport_rate(lat[i], vel[i, 0], vel[i, 1], h[i])
        omega_ie_n = earth_rate(lat[i])

        # Calculate Coriolis forces for this row
        cor_n[i, :] = torch.mm(omega_en_n + 2 * omega_ie_n, vel[i, :].unsqueeze(-1)).squeeze()
    return cor_n

def omega_in_n(lat: torch.Tensor, vel: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Earth's rotation rate in the navigation frame for each given latitude, velocity, and altitude.

    Args:
        lat (torch.Tensor): Mx1 tensor of latitude in radians.
        vel (torch.Tensor): Mx3 tensor of NED velocities (m/s).
        h (torch.Tensor): Mx1 tensor of altitude in meters.

    Returns:
        torch.Tensor: Mx3 tensor of Earth's rotation rate in the navigation frame (rad/s).
    """
    # Initialize the output tensor
    omega_in_n = torch.zeros((lat.shape[0], 3), dtype=lat.dtype, device=lat.device)

    # Iterate over each data point to compute the Earth's rotation rate
    for i in range(lat.shape[0]):
        # Calculate the Earth's rotation rate in the navigation frame
        omega_ie_n = earth_rate(lat[i])
        # Calculate the transport rate in the navigation frame
        omega_en_n = transport_rate(lat[i], vel[i, 0], vel[i, 1], h[i])
        # Calculate the total rotation rate in the navigation frame as the sum
        omega_in_n[i, :] = skewm_inv(omega_ie_n + omega_en_n).squeeze()

    return omega_in_n

def noise_b_sta(b_sta: torch.Tensor, M: int) -> torch.Tensor:
    """
    Generates both a deterministic and a stochastic (run-to-run) static bias error.

    Args:
        b_sta (torch.Tensor): 3d static bias to define the interval [-sbias sbias]
        M (int): dimension of output vector

    Returns:
        b_sta_n (torch.Tensor): Mx3 matrix with simulated static biases [X Y Z]
    """
    # It is considered that the stochastic static bias is a 10% of the deterministic static bias
    a = -b_sta * 0.1
    b = b_sta * 0.1

    # Stochastic static biases are chosen randomly within the interval [-b_sta b_sta]
    b_sta_stoc = torch.mul(torch.rand(1, 3, dtype=b_sta.dtype, device=b_sta.device), (b - a)) + a
    b_sta_stoc = b_sta_stoc.squeeze()

    # Create an array of ones with shape (M, 1)
    I = torch.ones(M, 1, dtype=b_sta.dtype, device=b_sta.device)

    # Create the output matrix
    b_sta_n = torch.cat((b_sta_stoc[0] * I + b_sta[0],
                         b_sta_stoc[1] * I + b_sta[1],
                         b_sta_stoc[2] * I + b_sta[2]), dim=1)

    return b_sta_n

def white_noise(sigma: torch.Tensor, M: int) -> torch.Tensor:
    """
    Generates white noise.

    Args:
        sigma (torch.Tensor): 3d white noise standard deviation
        M (int): dimension of output vector

    Returns:
        noise (torch.Tensor): Mx3 matrix with white noise [X Y Z]
    """
    # Generate white noise with standard normal distribution
    wn = torch.randn(M, 3, dtype=sigma.dtype, device=sigma.device)

    # Scale the white noise by the standard deviation
    noise = sigma.reshape(1, 3) * wn

    return noise

def noise_b_dyn(dyn_n_last: torch.Tensor, b_corr: torch.Tensor, b_dyn: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Generates a dynamic bias perturbation.

    Args:
        dyn_n_last (torch.Tensor): Mx3 last epoch noise of dynamic bias.
        b_corr (torch.Tensor): 3d correlation times.
        b_dyn (torch.Tensor): 3d level of dynamic biases.
        dt (float): sampling time.

    Returns:
        b_dyn_n (torch.Tensor): Mx3 matrix with simulated dynamic biases [X Y Z].
    """
    # First-order Gauss-Markov process
    M = dyn_n_last.shape[0]

    beta = dt / b_corr
    a1 = torch.exp(-beta)
    sigma_gm = b_dyn * torch.sqrt(1 - a1**2)

    b_wn = sigma_gm.reshape(1, 3) * torch.randn(M, 3, dtype=b_dyn.dtype, device=b_dyn.device)

    b_dyn_n = a1.reshape(1, 3) * dyn_n_last + b_wn

    return b_dyn_n

def noise_rrw(rrw_n_last: torch.Tensor, rrw: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Generates rate random walk noise.

    Args:
        rrw_n_last (torch.Tensor): Mx3 last epoch noise of rate random walk.
        rrw (torch.Tensor): 3d level of rate random walk.
        dt (float): sampling time.

    Returns:
        torch.Tensor: Mx3 matrix with simulated rate random walk noise [X Y Z] (rad/s^2, rad/s^2, rad/s^2).
    """
    b_noise = torch.randn(rrw_n_last.shape, dtype=rrw.dtype, device=rrw.device)

    rrw_n = rrw_n_last + rrw.reshape(1, 3) * dt * b_noise

    return rrw_n
