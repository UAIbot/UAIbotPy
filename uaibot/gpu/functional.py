import torch


def holder_min(
    values: torch.Tensor, gamma: float, dim: int = 1, eps: float = 1e-12
) -> torch.Tensor:
    r"""Hölder minimum: an almost differentiable approximation of the
    minimum.

    The Hölder minimum is differentiable except when the true minimum is
    zero. As :math:`\gamma \to \infty`, it converges to the true
    minimum. It is associative, commutative, and has the same sign as
    the true minimum.

    Parameters
    ----------
    values: torch.Tensor
        Input tensor for which the minimum is computed.
    gamma: float
        Positive parameter controlling how close the approximation is
        to the true minimum. Larger values give a sharper approximation.
    dim: int, optional
        Dimension along which the Hölder minimum is computed. Default
        is 1.
    eps: float, optional
        Small positive value used to avoid numerical issues, especially
        division by zero. Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``values`` except that the
        specified ``dim`` is removed.

    Notes
    -----
        For more information, see Definition 2.1 in
        https://arxiv.org/abs/2608.07707.
    """
    p = gamma + 1.0
    m = values.min(dim=dim, keepdim=True).values

    if torch.all(m > 0):
        y = values / m.clamp_min(eps)
        out = m * y.pow(-p).sum(dim=dim, keepdim=True).pow(-1.0 / p)

    elif torch.all(m < 0):
        M = (-m).clamp_min(eps)  # since M = max(-x) = -min(x)
        y = torch.clamp(-values / M, min=0.0)
        out = -M * y.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)

    else:
        # mixed batch case: falls back to safe elementwise selection
        y_pos = values / m.clamp_min(eps)
        pos = m * y_pos.pow(-p).sum(dim=dim, keepdim=True).pow(-1.0 / p)

        M = (-m).clamp_min(eps)
        y_neg = torch.clamp(-values / M, min=0.0)
        neg = -M * y_neg.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)

        out = torch.where(m > 0, pos, torch.where(m < 0, neg, torch.zeros_like(m)))

    return out.squeeze(dim)


def holder_max(
    values: torch.Tensor, gamma: float, dim: int = 1, eps: float = 1e-12
) -> torch.Tensor:
    r"""Hölder maximum: an almost differentiable approximation of the
    maximum.

    The Hölder maximum is differentiable except when the true maximum is
    zero. As :math:`\gamma \to \infty`, it converges to the true
    maximum. It is associative, commutative, and has the same sign as
    the true maximum.

    This implementation uses the identity
    :math:`\max(x) = -\min(-x)`.

    Parameters
    ----------
    values : torch.Tensor
        Input tensor for which the maximum is computed.
    gamma : float
        Positive parameter controlling how close the approximation is
        to the true maximum.
    dim : int, optional
        Dimension along which the Hölder maximum is computed. Default
        is 1.
    eps : float, optional
        Small positive value used to avoid numerical issues. Default is
        1e-12.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``values`` except that the
        specified ``dim`` is removed.
    """
    return -holder_min(-values, gamma, dim, eps)


def shaping_function(x: torch.Tensor, k: int, eps: float = 1e-3) -> torch.Tensor:
    r"""Smooth k-th order shaping function.

    The shaping function :math:`\phi_{k,\varepsilon}` modifies a function
    :math:`f(x)` to remove its non-differentiability at :math:`f(x)=0`,
    while approximately preserving its behavior away from zero.

    It is defined as:

    .. math::

        \phi_{k,\varepsilon}(x)
        =
        x \frac{|x|^k}{|x|^k + \varepsilon}.

    This function is :math:`k`-times differentiable. As :math:`\varepsilon
    \to 0` or :math:`|x| \gg \varepsilon`, it approaches the identity.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, usually the output of another function that may
        have a non-differentiable point at zero.
    k : int
        Positive integer controlling the order of differentiability.
    eps : float, optional
        Small positive value that avoids division by zero. Smaller values
        make the approximation closer to the identity, but can increase
        the magnitude of the derivative near zero. Default is 1e-3.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``x``.

    Notes
    -----
    For more information, see Eq. 1 and Definition 2.3 in
    https://arxiv.org/abs/2608.07707.
    """
    abs_x_pow = torch.abs(x) ** k
    res = x * abs_x_pow / (abs_x_pow + eps)
    return res
