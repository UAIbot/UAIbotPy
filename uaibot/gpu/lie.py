import torch


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
