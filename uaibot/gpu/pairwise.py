import torch
import torch.nn.functional as tfun


def pairwise_concat_objects(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concatenate object components from two batches along a new pair dimension.

    Given two tensors of component vectors from batch A and batch B, this
    builds all combinations ``(n_A, n_B)`` and concatenates the component
    vectors along the object dimension.

    Parameters
    ----------
    x : torch.Tensor
        Components from batch A, shape ``(N_A, C_A, 3)``.
    y : torch.Tensor
        Components from batch B, shape ``(N_B, C_B, 3)``.

    Returns
    -------
    torch.Tensor
        Combined components for every pair, shape ``(N_A, N_B, C_A + C_B, 3)``.
    """
    n1, no1, d1 = x.shape
    n2, no2, d2 = y.shape

    assert d1 == 3 and d2 == 3

    # (n1,1,no1,3) -> (n1,n2,no1,3)
    a_exp = x[:, None, :, :].expand(n1, n2, no1, 3)

    # (1,n2,no2,3) -> (n1,n2,no2,3)
    b_exp = y[None, :, :, :].expand(n1, n2, no2, 3)

    # concatenate along object dimension  (n1,n2,no1+no2,3)
    return torch.cat([a_exp, b_exp], dim=2)


def pairwise_cross(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Compute all pairwise cross products between two sets of edge vectors.

    Given two batches of edge direction vectors, computes the normalized cross
    product between every pair of vectors across the two batches.

    Parameters
    ----------
    x : torch.Tensor
        Edge vectors from batch A, shape ``(O1, E1, 3)``.
    y : torch.Tensor
        Edge vectors from batch B, shape ``(O2, E2, 3)``.
    eps : float, optional
        Small value used by normalization to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized cross products for every pair of edges, shape
        ``(O1, O2, E1 * E2, 3)``.
    """
    O1, E1, _ = x.shape
    O2, E2, _ = y.shape

    result = torch.cross(
        x[:, None, :, None, :],  # (O1,1,E1,1,3)
        y[None, :, None, :, :],  # (1,O2,1,E2,3)
        dim=-1,
    )  # (O1,O2,E1,E2,3)

    # Normalize along the last axis (the 3D vector)
    result = tfun.normalize(result, p=2, dim=-1, eps=eps)

    return result.reshape(O1, O2, E1 * E2, 3)  # (O1,O2,E1*E2,3)


def pairwise_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise differences between two sets of vertices.

    Parameters
    ----------
    x : torch.Tensor
        Vertices from batch A, shape ``(O1, V1, 3)``.
    y : torch.Tensor
        Vertices from batch B, shape ``(O2, V2, 3)``.

    Returns
    -------
    torch.Tensor
        Differences between every pair of vertices, shape
        ``(O1, O2, V1 * V2, 3)``.
    """
    O1, V1, _ = x.shape
    O2, V2, _ = y.shape

    result = (
        x[:, None, :, None, :]  # (O1,1,V1,1,3)
        - y[None, :, None, :, :]  # (1,O2,1,V2,3)
    )  # (O1,O2,V1,V2,3)

    return result.reshape(O1, O2, V1 * V2, 3)  # (O1,O2,V1*V2,3)


def pairwise_direction_vertex_dot(
    direction: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """Compute dot products between direction vectors and vertex differences.

    Parameters
    ----------
    direction : torch.Tensor
        Direction vectors for every object pair, shape ``(O1, O2, N, 3)``.
    vertices : torch.Tensor
        Vertex differences for every object pair, shape ``(O1, O2, V, 3)``.

    Returns
    -------
    torch.Tensor
        Dot products for every direction-vertex combination, shape
        ``(O1, O2, N, V)``.
    """
    # Save original shape, flatten outer dims
    o1, o2, N, _ = direction.shape
    _, _, V, _ = vertices.shape

    # Reshape to (o1*o2, N, 3) and (o1*o2, V, 3)
    dir_flat = direction.reshape(-1, N, 3)
    vert_flat = vertices.reshape(-1, V, 3)

    # Matrix multiply: (o1*o2, N, 3) × (o1*o2, 3, V) -> (o1*o2, N, V)
    result_flat = torch.matmul(dir_flat, vert_flat.transpose(-2, -1))

    # Reshape back to (o1, o2, N, V)
    return result_flat.reshape(o1, o2, N, V)
