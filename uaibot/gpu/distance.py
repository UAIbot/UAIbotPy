import torch
from uaibot.gpu.pairwise import (
    pairwise_difference,
    pairwise_direction_vertex_dot,
    pairwise_concat_objects,
)
from uaibot.gpu.functional import (
    holder_max,
    holder_min,
    shaping_function,
)
from collections import defaultdict
from uaibot.gpu.geometry import extract_VEF


def holder_distance_objects(
    objects_a, objects_b=None, gamma=2.0, eps=1e-3, device=None
):
    """Compute the Hölder distance between two heterogeneous sets of objects.

    Parameters
    ----------
    objects_a : sequence of Box or ConvexPolytope
        First collection of objects.
    objects_b : sequence of Box or ConvexPolytope, optional
        Second collection. If None, objects_a is used for both sides.
    gamma : float
        Hölder distance parameter.
    eps : float
        Numerical smoothing parameter.
    device : torch.device, optional
        Device on which to perform the computation. Defaults to CUDA if available.

    Returns
    -------
    torch.Tensor
        Distance matrix of shape ``(len(objects_a), len(objects_b))``.
    """
    if objects_b is None:
        objects_b = objects_a

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract components for all objects
    comp_a = [extract_VEF(obj) for obj in objects_a]
    comp_b = [extract_VEF(obj) for obj in objects_b]

    # Group indices by component signature (V, E, F)
    def group_indices(components):
        groups = defaultdict(list)
        for i, (v, e, n) in enumerate(components):
            sig = (v.shape[0], e.shape[0], n.shape[0])
            groups[sig].append(i)
        return groups

    groups_a = group_indices(comp_a)
    groups_b = group_indices(comp_b)

    # Initialize output matrix
    dist = torch.empty(
        (len(objects_a), len(objects_b)), dtype=torch.float32, device=device
    )

    # For every pair of homogeneous groups, compute a block
    for sig_a, idx_a in groups_a.items():
        # Stack tensors for group A
        vA = torch.stack([comp_a[i][0] for i in idx_a]).to(device)
        eA = torch.stack([comp_a[i][1] for i in idx_a]).to(device)
        nA = torch.stack([comp_a[i][2] for i in idx_a]).to(device)

        for sig_b, idx_b in groups_b.items():
            # Stack tensors for group B
            vB = torch.stack([comp_b[i][0] for i in idx_b]).to(device)
            eB = torch.stack([comp_b[i][1] for i in idx_b]).to(device)
            nB = torch.stack([comp_b[i][2] for i in idx_b]).to(device)

            # Compute batched distances for this group pair
            block = holder_distance(vA, eA, nA, vB, eB, nB, gamma, eps)

            # Fill the corresponding submatrix
            idx_a_t = torch.tensor(idx_a, device=device)
            idx_b_t = torch.tensor(idx_b, device=device)
            dist[idx_a_t[:, None], idx_b_t[None, :]] = block

    return dist


def _aggregate_holder_distance(values, gamma, eps=1e-3):
    r"""Aggregate distances using Hölder min/max operations.

    This function applies the final aggregation steps of the Hölder
    distance: it applies the Hölder min/max over aggregated values.

    Parameters
    ----------
    values : torch.Tensor
        Distances of shape ``(N_A, N_B, D, V)``, where ``D`` is
        the number of directions (edges and normals) and ``V`` is the number
        of vertex differences.
    gamma : float
        Positive parameter controlling the sharpness and differentiability
        of the aggregation.
    eps : float, optional
        Small positive value used for numerical smoothing. Default is 1e-3.

    Returns
    -------
    torch.Tensor
        Aggregated distance of shape ``(N_A, N_B)``.
    """
    # Group vertices
    x1 = holder_min(values, gamma, dim=3, eps=eps)  # (O1, O2, N)

    # Group normals
    x2 = holder_max(
        shaping_function(x1, gamma, eps=eps), gamma, dim=2, eps=eps
    )  # (O1, O2)

    return shaping_function(x2, gamma, eps=eps)


def holder_distance(
    vertexA: torch.Tensor,
    edgesA: torch.Tensor,
    normalsA: torch.Tensor,
    vertexB: torch.Tensor,
    edgesB: torch.Tensor,
    normalsB: torch.Tensor,
    gamma: float,
    eps: float = 1e-3,
) -> torch.Tensor:
    r"""Compute the Hölder distance between two batches of convex
    polyhedra.

    The Hölder distance is a differentiable signed distance between
    convex polyhedra. This function computes the distance between every
    pair of objects from batch ``A`` and batch ``B``.

    Parameters
    ----------
    vertexA : torch.Tensor
        Vertices of objects in batch A, shape ``(N_A, V_A, 3)``.
    edgesA : torch.Tensor
        Edge direction vectors of objects in batch A, shape ``(N_A, E_A, 3)``.
    normalsA : torch.Tensor
        Face normal vectors of objects in batch A, shape ``(N_A, F_A, 3)``.
    vertexB : torch.Tensor
        Vertices of objects in batch B, shape ``(N_B, V_B, 3)``.
    edgesB : torch.Tensor
        Edge direction vectors of objects in batch B, shape ``(N_B, E_B, 3)``.
    normalsB : torch.Tensor
        Face normal vectors of objects in batch B, shape ``(N_B, F_B, 3)``.
    gamma : float
        Positive parameter controlling the differentiability order and the
        sharpness of the Hölder min/max approximation. For integer values,
        the distance is ``gamma``-times differentiable with respect to the
        object geometry.
    eps : float, optional
        Small positive value used by the shaping function and the Hölder
        min/max aggregations to avoid numerical issues. Smaller values
        approximate the true distance more closely but can increase
        gradient magnitudes. Default is 1e-3.

    Returns
    -------
    torch.Tensor
        Signed distance between each pair of objects from A and B, with
        shape ``(N_A, N_B)``.

    Notes
    -----
    All input tensors must be on the same device and have a floating-point
    dtype. The objects in each batch need to have the same number of
    vertices, edges, or normals.

    For the mathematical definition, see Definition 3.2 in
    https://arxiv.org/abs/2608.07707.
    """
    edges_AB = pairwise_concat_objects(edgesA, edgesB)
    edges_AB = torch.cat([edges_AB, -edges_AB], dim=2)
    normals_AB = pairwise_concat_objects(normalsA, normalsB)
    vertices_AB = pairwise_difference(vertexA, vertexB)

    direction_AB = torch.cat([edges_AB, normals_AB], dim=2)

    pnv = pairwise_direction_vertex_dot(direction_AB, vertices_AB)

    dist = _aggregate_holder_distance(pnv, gamma, eps=eps)

    return dist
