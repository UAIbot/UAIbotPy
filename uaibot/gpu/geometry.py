import torch
import numpy as np
from scipy.spatial import HalfspaceIntersection
from uaibot.simobjects import Box, ConvexPolytope

# Default only when htm is numpy (so we don't inherit float32 by accident).
_DEFAULT_DTYPE = torch.float32


def _resolve_device_dtype(htm, device=None, dtype=None):
    if device is None:
        if isinstance(htm, torch.Tensor):
            device = htm.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        if isinstance(htm, torch.Tensor):
            dtype = htm.dtype
        else:
            dtype = _DEFAULT_DTYPE

    return torch.device(device), dtype


def _to_torch(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(
        x,
        device=device,
        dtype=dtype,
    )


def _to_numpy_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


# --------------------------------------------------------------------------
# Cacheable local-frame geometry (numpy only — HalfspaceIntersection is not
# differentiable and does not depend on htm, only on A, b)
# --------------------------------------------------------------------------


def _local_geometry_polytope(A_np, b_np, rtol=1e-8):
    halfspaces = np.hstack([A_np, -b_np.reshape(-1, 1)])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    local_verts = np.unique(hs.intersections.round(decimals=10), axis=0)

    nv, nf = local_verts.shape[0], A_np.shape[0]

    edge_indices = []
    for i in range(nv):
        vi = local_verts[i]
        for j in range(i + 1, nv):
            vj = local_verts[j]
            common = 0
            for k in range(nf):
                on_i = abs(np.dot(A_np[k], vi) - b_np[k]) <= rtol
                on_j = abs(np.dot(A_np[k], vj) - b_np[k]) <= rtol
                if on_i and on_j:
                    common += 1
            if common >= 2:
                edge_indices.append((i, j))

    local_edges = []
    for i, j in edge_indices:
        vec = local_verts[j] - local_verts[i]
        n = np.linalg.norm(vec)
        if n > rtol:
            local_edges.append(vec / n)
    local_edges = np.array(local_edges) if local_edges else np.empty((0, 3))

    return local_verts, local_edges, A_np


def _local_geometry_platonic(A_np, b_np, rtol=1e-5):
    halfspaces = np.hstack([A_np, -b_np.reshape(-1, 1)])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    local_verts = np.unique(hs.intersections.round(decimals=10), axis=0)

    nv = local_verts.shape[0]
    diff = local_verts[:, None, :] - local_verts[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    mask = sq_dist > 1e-12
    if not mask.any():
        raise ValueError("No edges found – degenerate vertices?")
    edge_len_sq = np.min(sq_dist[mask])
    tol = edge_len_sq * rtol

    edge_set = set()
    for i in range(nv):
        for j in range(i + 1, nv):
            if abs(sq_dist[i, j] - edge_len_sq) < tol:
                edge_set.add((i, j))

    edges_idx = np.array(sorted(edge_set))
    edge_vectors = local_verts[edges_idx[:, 1]] - local_verts[edges_idx[:, 0]]
    edge_vectors /= np.linalg.norm(edge_vectors, axis=1, keepdims=True)

    return local_verts, edge_vectors, A_np


def _cache_key(kind, A_np, b_np, rtol):
    A_np = np.ascontiguousarray(A_np)
    b_np = np.ascontiguousarray(b_np)

    return (
        kind,
        A_np.shape,
        A_np.dtype.str,
        A_np.tobytes(),
        b_np.shape,
        b_np.dtype.str,
        b_np.tobytes(),
        float(rtol),
    )


# --------------------------------------------------------------------------
# Public API: world-frame VEF from htm (differentiable w.r.t. htm)
# --------------------------------------------------------------------------


def get_VEF_from_polytope(
    htm, A, b, rtol=1e-8, device=None, dtype=None, local_cache=None
):
    device, dtype = _resolve_device_dtype(htm, device, dtype)
    A_np = _to_numpy_cpu(A)
    b_np = _to_numpy_cpu(b).ravel()

    if local_cache is not None:
        key = _cache_key("polytope", A_np, b_np, rtol)
        if key not in local_cache:
            local_cache[key] = _local_geometry_polytope(A_np, b_np, rtol)
        local_verts, local_edges, local_normals = local_cache[key]
    else:
        local_verts, local_edges, local_normals = _local_geometry_polytope(
            A_np, b_np, rtol
        )

    htm_t = _to_torch(htm, device, dtype)
    R = htm_t[:3, :3]
    t = htm_t[:3, 3]

    verts = torch.as_tensor(local_verts, device=device, dtype=dtype)
    edges = torch.as_tensor(local_edges, device=device, dtype=dtype)
    normals = torch.as_tensor(local_normals, device=device, dtype=dtype)

    # Differentiable in htm: `const @ R.T + t` keeps the graph through R, t.
    return verts @ R.T + t, edges @ R.T, normals @ R.T


def get_VEF_from_platonic(
    htm, A, b, rtol=1e-5, device=None, dtype=None, local_cache=None
):
    device, dtype = _resolve_device_dtype(htm, device, dtype)
    A_np = _to_numpy_cpu(A)
    b_np = _to_numpy_cpu(b).ravel()

    if local_cache is not None:
        key = _cache_key("platonic", A_np, b_np, rtol)
        if key not in local_cache:
            local_cache[key] = _local_geometry_platonic(A_np, b_np, rtol)
        local_verts, local_edges, local_normals = local_cache[key]
    else:
        local_verts, local_edges, local_normals = _local_geometry_platonic(
            A_np, b_np, rtol
        )

    htm_t = _to_torch(htm, device, dtype)
    R = htm_t[:3, :3]
    t = htm_t[:3, 3]

    verts = torch.as_tensor(local_verts, device=device, dtype=dtype)
    edges = torch.as_tensor(local_edges, device=device, dtype=dtype)
    normals = torch.as_tensor(local_normals, device=device, dtype=dtype)

    return verts @ R.T + t, edges @ R.T, normals @ R.T


def get_VEF_from_box(
    htm,
    lx,
    ly,
    lz,
    device=None,
    dtype=None,
):
    device, dtype = _resolve_device_dtype(htm, device, dtype)

    htm = _to_torch(htm, device, dtype)

    x = htm[:3, 0].reshape(3, 1)
    y = htm[:3, 1].reshape(3, 1)
    z = htm[:3, 2].reshape(3, 1)
    s = htm[:3, 3].reshape(3, 1)

    # ----------------------------------------------------------
    # Vertices: exactly the original ordering
    # ----------------------------------------------------------
    vertices = torch.cat(
        [
            s + 0.5 * lx * x + 0.5 * ly * y + 0.5 * lz * z,
            s + 0.5 * lx * x + 0.5 * ly * y - 0.5 * lz * z,
            s + 0.5 * lx * x - 0.5 * ly * y + 0.5 * lz * z,
            s + 0.5 * lx * x - 0.5 * ly * y - 0.5 * lz * z,
            s - 0.5 * lx * x + 0.5 * ly * y + 0.5 * lz * z,
            s - 0.5 * lx * x + 0.5 * ly * y - 0.5 * lz * z,
            s - 0.5 * lx * x - 0.5 * ly * y + 0.5 * lz * z,
            s - 0.5 * lx * x - 0.5 * ly * y - 0.5 * lz * z,
        ],
        dim=1,
    ).T.contiguous()

    # ----------------------------------------------------------
    # Edges: 4 parallel edges per axis
    # ----------------------------------------------------------
    edges = torch.cat(
        [x, x, x, x, y, y, y, y, z, z, z, z],
        dim=1,
    ).T.contiguous()

    # ----------------------------------------------------------
    # Face normals: ±x, ±y, ±z
    # ----------------------------------------------------------
    normals = torch.cat(
        [x, -x, y, -y, z, -z],
        dim=1,
    ).T.contiguous()

    return vertices, edges, normals


def extract_VEF(obj, htm=None, device=None, dtype=None, local_cache=None):
    """World-frame VEF: differentiable w.r.t. ``htm``; supports CUDA/numpy.

    Parameters
    ----------
    htm : torch.Tensor (4, 4) or numpy (4, 4), optional
        Falls back to ``obj.htm`` when omitted.
    device, dtype : optional overrides
        Defaults to ``htm.device`` / ``htm.dtype`` if ``htm`` is a tensor,
        else CUDA-if-available / float64.
    local_cache : dict, optional
        If supplied, memoises the local-frame HalfspaceIntersection result
        per ``(A_local, b_local, rtol)``. Pass the same dict across calls
        to skip recomputation when only ``htm`` varies (critical for the
        autograd-Hessian path).
    """
    if htm is None:
        htm = obj.htm

    if isinstance(obj, Box):
        return get_VEF_from_box(
            htm,
            obj.width,
            obj.depth,
            obj.height,
            device=device,
            dtype=dtype,
        )

    elif isinstance(obj, ConvexPolytope):
        return get_VEF_from_polytope(
            htm,
            obj.A_local,
            obj.b_local,
            device=device,
            dtype=dtype,
            local_cache=local_cache,
        )

    else:
        raise TypeError(f"Unsupported object type for GPU distance: {type(obj)}")
