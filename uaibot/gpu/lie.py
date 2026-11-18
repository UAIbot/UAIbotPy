import torch


def _eye(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)

# ------------------------------------------------------------------
# Smap / invSmap
# ------------------------------------------------------------------
def smap_se3(xi: torch.Tensor) -> torch.Tensor:
    """xi: (..., 6) -> (..., 4, 4)"""
    S = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    S[..., 0, 1] = -xi[..., 5]
    S[..., 0, 2] = xi[..., 4]
    S[..., 1, 2] = -xi[..., 3]
    S = S - S.transpose(-2, -1)
    S[..., 0, 3] = xi[..., 0]
    S[..., 1, 3] = xi[..., 1]
    S[..., 2, 3] = xi[..., 2]
    return S


def inv_smap_se3(A: torch.Tensor) -> torch.Tensor:
    """A: (..., 4, 4) -> (..., 6)"""
    xi = torch.zeros(*A.shape[:-2], 6, device=A.device, dtype=A.dtype)
    xi[..., 0] = A[..., 0, 3]
    xi[..., 1] = A[..., 1, 3]
    xi[..., 2] = A[..., 2, 3]
    xi[..., 3] = -A[..., 1, 2]
    xi[..., 4] = A[..., 0, 2]
    xi[..., 5] = -A[..., 0, 1]
    return xi


# ------------------------------------------------------------------
# expSE3 – batched, branchless
# ------------------------------------------------------------------
def exp_se3(X: torch.Tensor, theta_zero: float = 1e-6) -> torch.Tensor:
    """X: (..., 4, 4) -> exp(X) in SE(3)."""
    A = X[..., :3, :3]
    v = X[..., :3, 3]

    theta = torch.sqrt(A[..., 1, 0] ** 2 + A[..., 0, 2] ** 2 + A[..., 2, 1] ** 2)
    small = theta < theta_zero
    theta_safe = torch.where(small, torch.ones_like(theta), theta)

    sin_t = torch.sin(theta_safe)
    cos_t = torch.cos(theta_safe)

    c1 = torch.where(small, torch.ones_like(theta), sin_t / theta_safe)
    c2 = torch.where(small, torch.zeros_like(theta), (1.0 - cos_t) / theta_safe**2)
    c3 = torch.where(
        small, torch.zeros_like(theta), (theta_safe - sin_t) / theta_safe**3
    )

    I3 = _eye(3, A)
    AA = A @ A
    R = I3 + c1[..., None, None] * A + c2[..., None, None] * AA
    U = I3 + c2[..., None, None] * A + c3[..., None, None] * AA

    t = (U @ v.unsqueeze(-1)).squeeze(-1)

    # small-theta fallback: R = I, t = v
    R = torch.where(small[..., None, None], I3.expand_as(R), R)
    t = torch.where(small[..., None], v, t)

    result = torch.zeros_like(X)
    result[..., :3, :3] = R
    result[..., :3, 3] = t
    result[..., 3, 3] = 1.0
    return result


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from se(3) to SE(3).

    Parameters
    ----------
    xi : torch.Tensor of shape (6,)
        Twist vector (v, omega).

    Returns
    -------
    torch.Tensor of shape (4,4)
        Transformation matrix.
    """
    v = xi[:3]
    omega = xi[3:]
    theta = torch.norm(omega)
    if theta < 1e-8:
        # Pure translation
        R = torch.eye(3, dtype=xi.dtype, device=xi.device)
        t = v
    else:
        omega_hat = torch.tensor(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ],
            dtype=xi.dtype,
            device=xi.device,
        )
        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / (theta**2)
        C = (1.0 - A) / (theta**2)
        R = (
            torch.eye(3, dtype=xi.dtype, device=xi.device)
            + A * omega_hat
            + B * (omega_hat @ omega_hat)
        )
        V = (
            torch.eye(3, dtype=xi.dtype, device=xi.device)
            + B * omega_hat
            + C * (omega_hat @ omega_hat)
        )
        t = V @ v
    H = torch.eye(4, dtype=xi.dtype, device=xi.device)
    H[:3, :3] = R
    H[:3, 3] = t
    return H
