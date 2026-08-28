import torch
import numpy as np
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

# -------------------------------------------------------------------- #
#                            DISTANCE ONLY
# -------------------------------------------------------------------- #


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
    facenormalsA: torch.Tensor,
    vertexB: torch.Tensor,
    edgesB: torch.Tensor,
    facenormalsB: torch.Tensor,
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
    facenormalsA : torch.Tensor
        Face normal vectors of objects in batch A, shape ``(N_A, F_A, 3)``.
    vertexB : torch.Tensor
        Vertices of objects in batch B, shape ``(N_B, V_B, 3)``.
    edgesB : torch.Tensor
        Edge direction vectors of objects in batch B, shape ``(N_B, E_B, 3)``.
    facenormalsB : torch.Tensor
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
    facenormals_AB = pairwise_concat_objects(facenormalsA, facenormalsB)
    vertices_AB = pairwise_difference(vertexA, vertexB)

    normals_AB = torch.cat([edges_AB, facenormals_AB], dim=2)

    pnv = pairwise_direction_vertex_dot(normals_AB, vertices_AB)

    dist = _aggregate_holder_distance(pnv, gamma, eps=eps)

    return dist


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


# -------------------------------------------------------------------- #
#                            GRADIENT
# -------------------------------------------------------------------- #


def holder_distance_with_grad(
    vertexA: torch.Tensor,
    edgesA: torch.Tensor,
    facenormalsA: torch.Tensor,
    vertexB: torch.Tensor,
    edgesB: torch.Tensor,
    facenormalsB: torch.Tensor,
    gamma: float,
    eps: float,
):
    """Compute the Hölder distance between two batches of convex polyhedra,
    and the gradient of the sum of all distances with respect to the
    intermediate projected values (pnv).

    Parameters
    ----------
    vertexA, edgesA, facenormalsA, vertexB, edgesB,
    facenormalsB : torch.Tensor
        Batched geometry tensors, shapes as in `holder_distance`.
    gamma : float
        Hölder parameter.
    eps : float
        Numerical smoothing.

    Returns
    -------
    dist : torch.Tensor
        Distance matrix of shape ``(N_A, N_B)``.
    grad_pnv : torch.Tensor
        Gradient of ``dist.sum()`` with respect to the projected dot
        products ``pnv``. Shape: ``(N_A, N_B, D, V)``, where
        ``D = E_A + E_B + F_A + F_B`` (after concatenation and duplication)
        and ``V = V_A * V_B``.
    """
    # Convert to double for numerical stability
    vertexA = vertexA.double()
    edgesA = edgesA.double()
    facenormalsA = facenormalsA.double()
    vertexB = vertexB.double()
    edgesB = edgesB.double()
    facenormalsB = facenormalsB.double()
    # Forward projection without tracking gradients
    with torch.no_grad():
        edges_AB = pairwise_concat_objects(edgesA, edgesB)
        edges_AB = torch.cat([edges_AB, -edges_AB], dim=2)
        facenormals_AB = pairwise_concat_objects(facenormalsA, facenormalsB)
        vertices_AB = pairwise_difference(vertexA, vertexB)
        normals_AB = torch.cat([edges_AB, facenormals_AB], dim=2)
        pnv = pairwise_direction_vertex_dot(normals_AB, vertices_AB)

    # Detach and create a leaf tensor for gradient computation
    pnv_leaf = pnv.detach().requires_grad_(True)

    # Compute distances
    dist = _aggregate_holder_distance(pnv_leaf, gamma, eps)

    # Backward on the sum of distances
    loss = dist.sum()
    loss.backward()
    # Optionally convert back
    dist = dist.detach().float()
    grad = pnv_leaf.grad.float()
    # grad = torch.nan_to_num(grad, nan=0.0)  # replace NaN with 0
    # Return detached distances and the gradient
    return dist, grad


def se3_generators():
    """Return the six 4x4 SE(3) Lie algebra generators.

    Shape: (6, 4, 4). The ordering is:
      S[0] = rotation about x-axis   (angular velocity ω_x)
      S[1] = rotation about y-axis   (angular velocity ω_y)
      S[2] = rotation about z-axis   (angular velocity ω_z)
      S[3] = translation along x-axis (linear velocity v_x)
      S[4] = translation along y-axis (linear velocity v_y)
      S[5] = translation along z-axis (linear velocity v_z)
    """
    S = torch.zeros(6, 4, 4)
    # Translation generators
    S[0, 0, 3] = 1.0  # x translation
    S[1, 1, 3] = 1.0  # y translation
    S[2, 2, 3] = 1.0  # z translation
    # Rotation generators (skew-symmetric)
    # x-axis rotation
    S[3, 0, 1] = -1.0
    S[3, 1, 0] = 1.0
    # y-axis rotation
    S[4, 0, 2] = 1.0
    S[4, 2, 0] = -1.0
    # z-axis rotation
    S[5, 1, 2] = -1.0
    S[5, 2, 1] = 1.0
    return S


def pnv_grad_SE3(normals_A, vertices_A, normals_B, vertices_B, S=None):
    """Compute the L-operator gradients of pnv = n^T (a - b) with respect to
    SE(3) poses, for all combinations of normals from A/B and vertices
    from A/B.

    The function assumes that under a pose H, vertices and normals transform
    as H * v (homogeneous, last=1) and H * n (last=0), respectively.
    Both A and B are subject to independent poses H_A and H_B.

    Parameters
    ----------
    normals_A : torch.Tensor, shape (N_A, 3)
        Unit normals (face + edges) of object A in world frame.
    vertices_A : torch.Tensor, shape (V_A, 3)
        Vertices of object A in world frame.
    normals_B : torch.Tensor, shape (N_B, 3)
        Unit normals (face + edges) of object B.
    vertices_B : torch.Tensor, shape (V_B, 3)
        Vertices of object B.
    S : torch.Tensor, shape (6,4,4), optional
        SE(3) Lie algebra generators. If None, uses the default basis.

    Returns
    -------
    dict of torch.Tensor:
        'nA_grad_HA': shape (N_A, V_A, V_B, 6)
            Gradient of pnv w.r.t. pose A for normals from A.
        'nA_grad_HB': shape (N_A, V_A, V_B, 6)
            Gradient of pnv w.r.t. pose B for normals from A.
        'nB_grad_HA': shape (N_B, V_A, V_B, 6)
            Gradient of pnv w.r.t. pose A for normals from B.
        'nB_grad_HB': shape (N_B, V_A, V_B, 6)
            Gradient of pnv w.r.t. pose B for normals from B.
    """
    if S is None:
        S = se3_generators().to(normals_A.device)

    # Convert to homogeneous coordinates
    nA_h = torch.cat(
        [normals_A, torch.zeros_like(normals_A[..., :1])], dim=-1
    )  # (F_A,4)
    nB_h = torch.cat(
        [normals_B, torch.zeros_like(normals_B[..., :1])], dim=-1
    )  # (F_B,4)
    vA_h = torch.cat(
        [vertices_A, torch.ones_like(vertices_A[..., :1])], dim=-1
    )  # (V_A,4)
    vB_h = torch.cat(
        [vertices_B, torch.ones_like(vertices_B[..., :1])], dim=-1
    )  # (V_B,4)

    # Precompute transposed generators
    S_t = S.transpose(-1, -2)  # (6,4,4) with indices (g,j,i)

    # Compute fundamental products with generator index g
    # Compute fundamental products using einsum
    # n^T S_i v (i: basis index, f: normal index, v: vertex index)
    prod_nA_S_vA = torch.einsum("fi,gij,vj->fvg", nA_h, S, vA_h)  # (F_A, V_A, 6)
    prod_nA_St_vB = torch.einsum("fi,gji,vj->fvg", nA_h, S_t, vB_h)  # (F_A, V_B, 6)
    prod_nB_S_vA = torch.einsum("fi,gij,vj->fvg", nB_h, S, vA_h)  # (F_B, V_A, 6)
    prod_nB_St_vA = torch.einsum("fi,gji,vj->fvg", nB_h, S_t, vA_h)  # (F_B, V_A, 6)
    prod_nB_S_vB = torch.einsum("fi,gij,vj->fvg", nB_h, S, vB_h)  # (F_B, V_B, 6)

    # Expand dimensions to combine over (V_A, V_B)
    # For n from A:
    #   grad_HA = 2 * nA^T S a_A - nA^T S^T b_B
    grad_HA_nA = (
        2.0 * prod_nA_S_vA[:, :, None, :] - prod_nA_St_vB[:, None, :, :]
    )  # (F_A, V_A, V_B, 6)
    #   grad_HB = - nA^T S b_B
    grad_HB_nA = -prod_nA_St_vB[:, None, :, :]

    # For n from B:
    #   grad_HA = nB^T S a_A
    grad_HA_nB = prod_nB_S_vA[:, :, None, :]  # (F_B, V_A, V_B, 6)
    #   grad_HB = nB^T S^T a_A - nB^T S^T b_B - nB^T S b_B
    grad_HB_nB = (
        prod_nB_St_vA[:, :, None, :]
        # - prod_nB_St_vB[:, None, :, :]
        - 2.0 * prod_nB_S_vB[:, None, :, :]
    )

    return {
        "nA_grad_HA": grad_HA_nA,
        "nA_grad_HB": grad_HB_nA,
        "nB_grad_HA": grad_HA_nB,
        "nB_grad_HB": grad_HB_nB,
    }


def holder_distance_objects_with_grad(
    objects_a,
    objects_b=None,
    gamma=2.0,
    eps=1e-3,
    device=None,
):
    """
    Compute Hölder distances and pose gradients for heterogeneous object sets.

    Distances are computed between all objects in `objects_a` and `objects_b`
    (or between objects in `objects_a` if `objects_b` is `None`). Gradients
    are returned for the sum of all distances with respect to left‑perturbations
    of the SE(3) poses.

    Parameters
    ----------
    objects_a : list
        Objects (e.g., `ub.Box`) in batch A.
    objects_b : list, optional
        Objects in batch B. If `None`, uses `objects_a` (self‑comparison).
    gamma : float, default 2.0
        Hölder parameter.
    eps : float, default 1e-3
        Numerical smoothing.
    device : torch.device, optional
        Device for tensors. If `None`, uses CUDA if available, else CPU.

    Returns
    -------
    dist : torch.Tensor, shape (N_A, N_B)
        Distance matrix.
    grad_dict : dict of torch.Tensor
        'grad_HA' : (N_A, 6) – gradient w.r.t. pose of each object in A.
        'grad_HB' : (N_B, 6) – gradient w.r.t. pose of each object in B.
        Gradient vector order: [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        (linear velocity first, angular velocity second).
        If `objects_b is None`, `grad_HB` equals `grad_HA`.
    """

    if objects_b is None:
        objects_b = objects_a

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract components (vertices, edges, face normals)
    comp_a = [extract_VEF(obj) for obj in objects_a]
    comp_b = [extract_VEF(obj) for obj in objects_b]

    # Group by signature
    def group_indices(components):
        groups = defaultdict(list)
        for i, (v, e, n) in enumerate(components):
            sig = (v.shape[0], e.shape[0], n.shape[0])
            groups[sig].append(i)
        return groups

    groups_a = group_indices(comp_a)
    groups_b = group_indices(comp_b)

    N_A = len(objects_a)
    N_B = len(objects_b)
    dist = torch.empty((N_A, N_B), dtype=torch.float32, device=device)
    grad_HA = torch.zeros(N_A, 6, dtype=torch.float32, device=device)
    grad_HB = torch.zeros(N_B, 6, dtype=torch.float32, device=device)

    for sig_a, idx_a in groups_a.items():
        # Stack geometry for the whole group
        vA = torch.stack([comp_a[i][0] for i in idx_a]).to(device)  # (nA, V_A, 3)
        eA = torch.stack([comp_a[i][1] for i in idx_a]).to(device)  # (nA, E_A, 3)
        nA = torch.stack([comp_a[i][2] for i in idx_a]).to(device)  # (nA, F_A, 3)

        for sig_b, idx_b in groups_b.items():
            vB = torch.stack([comp_b[i][0] for i in idx_b]).to(device)
            eB = torch.stack([comp_b[i][1] for i in idx_b]).to(device)
            nB = torch.stack([comp_b[i][2] for i in idx_b]).to(device)

            # Compute block distances and pnv gradient (w.r.t. pnv)
            block_dist, block_grad = holder_distance_with_grad(
                vA, eA, nA, vB, eB, nB, gamma, eps
            )  # block_dist: (len(idx_a), len(idx_b)), block_grad: (len(idx_a), len(idx_b), D, V)

            # Dimensions
            nA_local, V_A, _ = vA.shape
            nB_local, V_B, _ = vB.shape
            E_A = eA.shape[1]
            E_B = eB.shape[1]
            F_A = nA.shape[1]
            F_B = nB.shape[1]
            D = block_grad.shape[2]  # = 2*E_A + 2*E_B + F_A + F_B
            V = block_grad.shape[3]  # should equal V_A * V_B

            # Fill distance block
            dist[np.ix_(idx_a, idx_b)] = block_dist.detach()

            # Loop over each pair in the block to compute pose gradients
            for i_local, i_global in enumerate(idx_a):
                for j_local, j_global in enumerate(idx_b):
                    # Extract geometry for this pair
                    vertices_i = vA[i_local]  # (V_A, 3)
                    edges_i = eA[i_local]  # (E_A, 3)
                    normals_i = nA[i_local]  # (F_A, 3)

                    vertices_j = vB[j_local]  # (V_B, 3)
                    edges_j = eB[j_local]  # (E_B, 3)
                    normals_j = nB[j_local]  # (F_B, 3)

                    # Directions from A and B (positive edges + normals)
                    dirs_A = torch.cat([edges_i, normals_i], dim=0)  # (E_A + F_A, 3)
                    dirs_B = torch.cat([edges_j, normals_j], dim=0)  # (E_B + F_B, 3)

                    # Compute L-operator gradients for this pair
                    pnv_grads = pnv_grad_SE3(
                        normals_A=dirs_A,
                        vertices_A=vertices_i,
                        normals_B=dirs_B,
                        vertices_B=vertices_j,
                    )

                    # Gradients for positive directions (shape: (num_dirs, V_A, V_B, 6))
                    pos_nA_grad_HA = pnv_grads["nA_grad_HA"]  # (E_A+F_A, V_A, V_B, 6)
                    pos_nA_grad_HB = pnv_grads["nA_grad_HB"]
                    pos_nB_grad_HA = pnv_grads["nB_grad_HA"]  # (E_B+F_B, V_A, V_B, 6)
                    pos_nB_grad_HB = pnv_grads["nB_grad_HB"]

                    # Build full gradient tensors for this pair w.r.t. H_A and H_B
                    pair_grad_HA = torch.zeros(D, V_A, V_B, 6, device=device)
                    pair_grad_HB = torch.zeros(D, V_A, V_B, 6, device=device)

                    # Positive edges of A
                    pair_grad_HA[:E_A] = pos_nA_grad_HA[:E_A]
                    pair_grad_HB[:E_A] = pos_nA_grad_HB[:E_A]

                    # Positive edges of B
                    pair_grad_HA[E_A : E_A + E_B] = pos_nB_grad_HA[:E_B]
                    pair_grad_HB[E_A : E_A + E_B] = pos_nB_grad_HB[:E_B]

                    # Negative edges of A (negated)
                    pair_grad_HA[E_A + E_B : 2 * E_A + E_B] = -pos_nA_grad_HA[:E_A]
                    pair_grad_HB[E_A + E_B : 2 * E_A + E_B] = -pos_nA_grad_HB[:E_A]

                    # Negative edges of B (negated)
                    pair_grad_HA[2 * E_A + E_B : 2 * E_A + 2 * E_B] = -pos_nB_grad_HA[
                        :E_B
                    ]
                    pair_grad_HB[2 * E_A + E_B : 2 * E_A + 2 * E_B] = -pos_nB_grad_HB[
                        :E_B
                    ]

                    # Normals of A
                    pair_grad_HA[2 * E_A + 2 * E_B : 2 * E_A + 2 * E_B + F_A] = (
                        pos_nA_grad_HA[E_A:]
                    )
                    pair_grad_HB[2 * E_A + 2 * E_B : 2 * E_A + 2 * E_B + F_A] = (
                        pos_nA_grad_HB[E_A:]
                    )

                    # Normals of B
                    pair_grad_HA[2 * E_A + 2 * E_B + F_A : D] = pos_nB_grad_HA[E_B:]
                    pair_grad_HB[2 * E_A + 2 * E_B + F_A : D] = pos_nB_grad_HB[E_B:]

                    # Reshape the pnv gradient for this pair to (D, V_A, V_B)
                    grad_pnv_ij = block_grad[i_local, j_local].reshape(D, V_A, V_B)

                    # Compute contribution to pose gradients
                    grad_HA_contrib = torch.einsum(
                        "dab,dabk->k", grad_pnv_ij, pair_grad_HA
                    )
                    grad_HB_contrib = torch.einsum(
                        "dab,dabk->k", grad_pnv_ij, pair_grad_HB
                    )

                    # Accumulate
                    grad_HA[i_global] += grad_HA_contrib
                    grad_HB[j_global] += grad_HB_contrib

    return dist, {"grad_HA": grad_HA, "grad_HB": grad_HB}
