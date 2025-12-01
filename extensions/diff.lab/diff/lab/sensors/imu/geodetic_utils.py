import torch

def skewm(vector: torch.Tensor) -> torch.Tensor:
    """
    Constructs a skew-symmetric matrix from a 3-element vector.

    Args:
        vector (torch.Tensor): 3x1 tensor.

    Returns:
        torch.Tensor: 3x3 skew-symmetric matrix.
    """
    return torch.tensor([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ], dtype=vector.dtype, device=vector.device)

def skewm_inv(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a skew-symmetric matrix.

    Args:
        matrix (torch.Tensor): 3x3 skew-symmetric matrix.

    Returns:
        torch.Tensor: 3x1 vector.
    """
    return torch.tensor([
        matrix[2, 1],
        -matrix[2, 0],
        matrix[1, 0]
    ], dtype=matrix.dtype, device=matrix.device)

def radius(lat: torch.Tensor) -> tuple:
    """
    Calculates meridian (RM) and normal (RN) radii of curvature.

    Args:
        lat (torch.Tensor): Nx1 tensor of latitude in radians.

    Returns:
        tuple: RM (meridian radius of curvature) and RN (normal radius of curvature).
    """
    # WGS84 constants
    a = 6378137.0  # Equatorial radius in meters
    e = 0.0818191908425  # Eccentricity

    e2 = e ** 2  # Square of eccentricity
    sin_lat = torch.sin(lat)
    den = 1 - e2 * sin_lat**2

    # Meridian radius of curvature (North-South)
    RM = a * (1 - e2) / (den**(3 / 2))

    # Normal radius of curvature (East-West)
    RN = a / torch.sqrt(den)

    return RM, RN

def earth_rate(lat: torch.Tensor) -> torch.Tensor:
    """
    Calculates Earth's rotation rate in the navigation frame as a skew-symmetric matrix.

    Args:
        lat (torch.Tensor): Latitude (radians).

    Returns:
        torch.Tensor: 3x3 skew-symmetric Earth rate matrix (rad/s).
    """
    omega_ie = 7.2921155e-5  # Earth rotation rate (rad/s)

    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)

    # Earth's rotation rate vector
    omega_ie_n = torch.tensor([cos_lat.item(), 0, -sin_lat.item()],
                              dtype=lat.dtype, device=lat.device)

    return skewm(omega_ie_n * omega_ie)


def transport_rate(lat: torch.Tensor, Vn: torch.Tensor, Ve: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Calculates the transport rate in the navigation frame as a skew-symmetric matrix.

    Args:
        lat (torch.Tensor): Latitude (radians).
        Vn (torch.Tensor): North velocity (m/s).
        Ve (torch.Tensor): East velocity (m/s).
        h (torch.Tensor): Altitude (m).

    Returns:
        torch.Tensor: 3x3 skew-symmetric transport rate matrix (rad/s).
    """
    # Ensure altitude is non-negative
    h = torch.abs(h)

    # Compute radii
    RM, RN = radius(lat)

    # Transport rate vector
    om_en_n = torch.tensor([
        Ve / (RN + h),              # North
        -Vn / (RM + h),             # East
        -Ve * torch.tan(lat) / (RN + h)  # Down
    ], dtype=lat.dtype, device=lat.device)

    return skewm(om_en_n)