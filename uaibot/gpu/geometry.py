import torch
import numpy as np
from scipy.spatial import HalfspaceIntersection
from uaibot.simobjects import Box, ConvexPolytope


def extract_VEF(obj):
    """Extract vertices, edge directions, and face normals from a supported object.

    Parameters
    ----------
    obj : Box or ConvexPolytope
        UAIbotPy object with a homogeneous transform.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Vertices, edge direction vectors, and face normals.
    """
    if isinstance(obj, Box):
        return get_VEF_from_box(obj.htm, obj.width, obj.depth, obj.height)
    elif isinstance(obj, ConvexPolytope):
        return get_VEF_from_polytope(obj.htm, obj.A, obj.b)
    else:
        raise TypeError(f"Unsupported object type for GPU distance: {type(obj)}")


def get_VEF_from_platonic(htm, A, b, rtol=1e-5):
    """Extract vertices, true edge directions, and face normals from a
    convex polyhedron.

    Parameters
    ----------
    htm : np.ndarray (4,4)
        Homogeneous transformation matrix (world frame).
    A : np.ndarray (F,3)
        Outward unit normals in local frame.
    b : np.ndarray (F,)
        Face offsets (distance from origin to plane) in local frame.
    rtol : float
        Relative tolerance for edge‑length equality.

    Returns
    -------
    vertices : torch.Tensor (V, 3)
        Vertices in world frame.
    edges : torch.Tensor (E, 3)
        Unit direction vectors of each undirected edge in world frame.
    normals : torch.Tensor (F, 3)
        Outward unit normals of each face in world frame.
    """
    # 1. Local vertices from half‑space intersection
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    local_verts = hs.intersections  # (V, 3)
    local_verts = np.unique(local_verts.round(decimals=10), axis=0)

    # 2. Find true edges by minimal positive distance (like C++ logic)
    nv = local_verts.shape[0]
    # all pairwise squared distances
    diff = local_verts[:, None, :] - local_verts[None, :, :]  # (V, V, 3)
    sq_dist = np.sum(diff * diff, axis=-1)  # (V, V)

    # Ignore zero (self) and find minimal positive squared length
    mask = sq_dist > 1e-12
    if not mask.any():
        raise ValueError("No edges found – degenerate vertices?")
    edge_len_sq = np.min(sq_dist[mask])
    tol = edge_len_sq * rtol

    # Select all unordered pairs with that length
    edge_set = set()
    for i in range(nv):
        for j in range(i + 1, nv):
            if abs(sq_dist[i, j] - edge_len_sq) < tol:
                edge_set.add((i, j))

    edges_idx = np.array(list(edge_set))  # (E, 2)
    # Edge direction vectors (unit) – shape (E, 3)
    edge_vectors = local_verts[edges_idx[:, 1]] - local_verts[edges_idx[:, 0]]
    edge_vectors /= np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    local_edges = edge_vectors

    # 3. Face normals are already unit – shape (F, 3)
    local_normals = A

    # 4. Transform to world frame
    R = htm[:3, :3]
    t = htm[:3, 3]
    world_vertices = (R @ local_verts.T).T + t  # (V, 3)
    world_edges = (R @ local_edges.T).T  # (E, 3)
    world_normals = (R @ local_normals.T).T  # (F, 3)

    # 5. Return float32 tensors
    return (
        torch.tensor(world_vertices, dtype=torch.float32),
        torch.tensor(world_edges, dtype=torch.float32),
        torch.tensor(world_normals, dtype=torch.float32),
    )


def get_VEF_from_box(htm, lx, ly, lz):
    """Extract vertices, true edge directions, and face normals from a
    box.

    Parameters
    ----------
    htm : np.ndarray (4,4)
        Homogeneous transformation matrix (world frame).
    lx : float
        Width of the box in meters
    ly : float
        depth of the box in meters
    lz : float
        height of the box in meters

    Returns
    -------
    vertices : torch.Tensor (8, 3)
        Vertices in world frame.
    edges : torch.Tensor (12, 3)
        Unit direction vectors of each undirected edge in world frame.
    normals : torch.Tensor (6, 3)
        Outward unit normals of each face in world frame.
    """

    htm = np.array(htm)
    x = htm[0:3, 0].reshape(3, 1)
    y = htm[0:3, 1].reshape(3, 1)
    z = htm[0:3, 2].reshape(3, 1)
    s = htm[0:3, 3].reshape(3, 1)

    vertices = torch.tensor(
        np.hstack(
            [
                s + 0.5 * lx * x + 0.5 * ly * y + 0.5 * lz * z,
                s + 0.5 * lx * x + 0.5 * ly * y - 0.5 * lz * z,
                s + 0.5 * lx * x - 0.5 * ly * y + 0.5 * lz * z,
                s + 0.5 * lx * x - 0.5 * ly * y - 0.5 * lz * z,
                s - 0.5 * lx * x + 0.5 * ly * y + 0.5 * lz * z,
                s - 0.5 * lx * x + 0.5 * ly * y - 0.5 * lz * z,
                s - 0.5 * lx * x - 0.5 * ly * y + 0.5 * lz * z,
                s - 0.5 * lx * x - 0.5 * ly * y - 0.5 * lz * z,
            ]
        ),
        dtype=torch.float32,
    ).T.contiguous()  # (8, 3) after tranpose

    # Edge direction vectors – one per unique undirected edge line,
    # keeping the C++ Box convention: 6 oriented unit vectors (±X,±Y,±Z)
    edges = torch.tensor(
        np.hstack([x, x, x, x, y, y, y, y, z, z, z, z]), dtype=torch.float32
    ).T.contiguous()  # shape (12,3) after transpose

    # Face normals: ±x, ±y, ±z (unit length)
    normals = torch.tensor(
        np.hstack([x, -x, y, -y, z, -z]), dtype=torch.float32
    ).T.contiguous()  # shape (6,3) after tranpose

    # Normals and edges are already unit vectors
    return vertices, edges, normals
