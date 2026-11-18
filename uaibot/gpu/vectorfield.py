import math
import numpy as np
from typing import Optional, Tuple, Union
import torch
from uaibot.gpu.lie import exp_se3, inv_smap_se3, smap_se3


class VectorFieldSE3:
    """
    Batched, GPU-friendly vector field on SE(3).

    All methods operate on tensors of shape (..., 4, 4) (or (..., 6) for
    Lie algebra vectors) and use broadcasting so that the whole
    `vectorfield_SE3` computation is a single fused pipeline.
    """

    def __init__(
        self,
        delta: float = 1e-3,
        ds: float = 1e-3,
        max_cos_theta: float = 0.999,
        theta_zero: float = 1e-6,
        normalize_tangent: bool = False,
        normalize_normal: bool = False,
        scale_pos: float = 1.0,
    ):
        self.delta = delta
        self.ds = ds
        self.max_cos_theta = max_cos_theta
        self.theta_zero = theta_zero
        self.normalize_tangent = normalize_tangent
        self.normalize_normal = normalize_normal
        self.scale_pos = scale_pos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _eye(n: int, like: torch.Tensor) -> torch.Tensor:
        return torch.eye(n, device=like.device, dtype=like.dtype)

    # ------------------------------------------------------------------
    # EEdistSE3Variables – fully vectorised
    # ------------------------------------------------------------------
    def eedist_se3_variables(self, X: torch.Tensor):
        """
        X: (..., 4, 4)
        Returns Q, t, M, theta, cos_theta, sin_theta, alpha
        all batched with shape (...,) / (..., 3, 3) / (..., 3)
        """
        Q = X[..., :3, :3]  # (..., 3, 3)
        t = X[..., :3, 3]  # (..., 3)

        Q_inv = torch.linalg.inv(Q)  # batched inverse

        trace = Q.diagonal(dim1=-2, dim2=-1).sum(-1)  # (...,)
        cos_theta0 = 0.5 * (trace - 1.0)

        diff = Q - Q_inv
        fro_norm = torch.linalg.matrix_norm(diff, ord="fro")  # (...,)
        s2 = torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype))
        sin_theta0 = fro_norm / (2.0 * s2)

        theta = torch.atan2(sin_theta0, cos_theta0)  # (...,)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # ------------------------------------------------------------------
        # alpha with numeric safety for cos_theta ~ 1 (uses torch.where
        # which still evaluates both branches => guard the denominator).
        # ------------------------------------------------------------------
        mask = cos_theta > self.max_cos_theta
        numerator = 2.0 - 2.0 * cos_theta - theta**2
        denom = 4.0 * (1.0 - cos_theta) ** 2
        denom_safe = torch.where(mask, torch.ones_like(denom), denom)
        alpha_general = numerator / denom_safe
        alpha = torch.where(
            mask, torch.full_like(cos_theta, -1.0 / 12.0), alpha_general
        )

        I3 = self._eye(3, Q)
        M = (
            alpha[..., None, None] * (Q + Q_inv)
            + (1.0 - 2.0 * alpha[..., None, None]) * I3
        )

        return Q, t, M, theta, cos_theta, sin_theta, alpha

    # ------------------------------------------------------------------
    # EEdistance – broadcasting over arbitrary batch shapes
    # ------------------------------------------------------------------
    def eedistance(
        self,
        V: torch.Tensor,
        W: torch.Tensor,
        scale_pos: float = None,
    ) -> torch.Tensor:
        """
        V, W: (..., 4, 4) with broadcasting.
        """
        if scale_pos is None:
            scale_pos = self.scale_pos
        Z = torch.linalg.inv(V) @ W  # (..., 4, 4)
        Q, t, M, theta, _, _, _ = self.eedist_se3_variables(Z)

        t_col = t.unsqueeze(-1)  # (..., 3, 1)
        tMt = (t.unsqueeze(-2) @ M @ t_col).squeeze(-1).squeeze(-1)
        distance_sq = 2.0 * theta**2 + (scale_pos**2) * tMt

        return torch.sqrt(distance_sq.clamp_min(0.0))

    # ------------------------------------------------------------------
    # ECdistance – closest point on curve (chunked to bound memory)
    # ------------------------------------------------------------------
    def ecdistance(
        self,
        states: torch.Tensor,  # (N, 4, 4)
        curve: torch.Tensor,  # (M, 4, 4)
        chunk_size: int = 256,  # states per chunk (tuned for mem)
        scale_pos: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully vectorised brute-force closest-point search on SE(3).

        Eliminates the O(C*M) 3x3 inverses and (C,M,3,3) intermediates by
        exploiting the structure Z = V^{-1} W and expressing
            d^2(V,W) = 2*theta^2 + t^T M t
        purely in terms of scalar (C,M) quantities.
        """
        if scale_pos is None:
            scale_pos = self.scale_pos
        N, M = states.shape[0], curve.shape[0]
        device, dtype = states.device, states.dtype

        # ---- per-state data (once, not per chunk) ------------------------
        R_V = states[..., :3, :3]  # (N, 3, 3)
        p_V = states[..., :3, 3]  # (N, 3)
        R_V_T = R_V.transpose(-1, -2)  # (N, 3, 3) = R_V^{-1}

        # ---- per-curve data (once) ---------------------------------------
        R_W = curve[..., :3, :3]  # (M, 3, 3)
        p_W = curve[..., :3, 3]  # (M, 3)

        min_dists = torch.empty(N, device=device, dtype=dtype)
        min_indices = torch.empty(N, dtype=torch.long, device=device)

        # Auto-chunk so C*M stays under ~5e6 (≈ 800 MB peak in f64)
        if chunk_size is None:
            chunk_size = max(1, int(5_000_000 // max(M, 1)))

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            C = end - start

            R_V_c = R_V[start:end]  # (C, 3, 3)
            R_V_T_c = R_V_T[start:end]  # (C, 3, 3)
            p_V_c = p_V[start:end]  # (C, 3)

            # ---- trace(Q) = <R_V, R_W>_F : (C, M) ------------------------
            # 9 scalar outer products — never materialises (C, M, 3, 3).
            trace_Q = R_V_c[:, 0, 0, None] * R_W[None, :, 0, 0]
            for i in range(3):
                for j in range(3):
                    if i == 0 and j == 0:
                        continue
                    trace_Q = trace_Q + R_V_c[:, i, j, None] * R_W[None, :, i, j]

            # ---- t = R_V^T (p_W - p_V) : (C, M, 3) -----------------------
            RVT_pW = torch.empty(C, M, 3, device=device, dtype=dtype)
            for i in range(3):
                acc = R_V_T_c[:, i, 0, None] * p_W[None, :, 0]
                for j in range(1, 3):
                    acc = acc + R_V_T_c[:, i, j, None] * p_W[None, :, j]
                RVT_pW[:, :, i] = acc
            RVT_pV = torch.einsum("cij,cj->ci", R_V_T_c, p_V_c)  # (C, 3)
            t = RVT_pW - RVT_pV[:, None, :]  # (C, M, 3)

            t_sq = (t * t).sum(-1)  # (C, M)

            # ---- t^T Q t = d_W · (R_W t) --------------------------------
            RW_t = torch.empty(C, M, 3, device=device, dtype=dtype)
            for i in range(3):
                acc = R_W[None, :, i, 0] * t[:, :, 0]
                for j in range(1, 3):
                    acc = acc + R_W[None, :, i, j] * t[:, :, j]
                RW_t[:, :, i] = acc

            # d_W = p_W - p_V  (broadcast to (C, M, 3) implicitly)
            tQt = (p_W[None, :, 0] - p_V_c[:, None, 0]) * RW_t[:, :, 0]
            for i in range(1, 3):
                tQt = tQt + (p_W[None, :, i] - p_V_c[:, None, i]) * RW_t[:, :, i]

            # ---- angle θ : (C, M) ---------------------------------------
            cos_theta = (trace_Q - 1.0) * 0.5
            cos_theta = cos_theta.clamp(-1.0, 1.0)
            sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp_min(0.0))
            theta = torch.atan2(sin_theta, cos_theta)

            # ---- α : (C, M) ---------------------------------------------
            mask = cos_theta > self.max_cos_theta
            one_c = (1.0 - cos_theta).clamp_min(1e-12)
            alpha = torch.where(
                mask,
                torch.full_like(theta, -1.0 / 12.0),
                (2.0 - 2.0 * cos_theta - theta**2) / (4.0 * one_c**2),
            )

            # ---- t^T M t = 2α·tQt + (1−2α)·‖t‖² : (C, M) -----------------
            tMt = 2.0 * alpha * tQt + (1.0 - 2.0 * alpha) * t_sq
            tMt *= scale_pos**2

            d = torch.sqrt((2.0 * theta**2 + tMt).clamp_min(0.0))  # (C, M)

            d_min, i_min = d.min(dim=1)
            min_dists[start:end] = d_min
            min_indices[start:end] = i_min

        return min_dists, min_indices

    # ------------------------------------------------------------------
    # Main vector field – everything batched
    # ------------------------------------------------------------------

    def L_eedistance(
        self,
        V: torch.Tensor,
        W: torch.Tensor,
        scale_pos: float = None,
    ) -> torch.Tensor:
        """Return L(V,W)^T with shape (..., 6, 1), supporting broadcasting.

        V, W must be SE(3) matrices. Requires relative rotation angle < pi,
        positive distance, and an input away from the alpha branch boundary.
        At zero distance, returns zero by convention; the classical derivative
        of the distance generally does not exist there.

        Uses exactly the alpha and the constant-alpha mask in the supplied
        eedist_se3_variables method. Device and dtype are preserved.
        """
        if scale_pos is None:
            scale_pos = self.scale_pos
        s2 = scale_pos**2

        Z = torch.linalg.inv(V) @ W
        Q, t, M, theta, cos_theta, _, alpha = self.eedist_se3_variables(Z)
        Mt = (M @ t.unsqueeze(-1)).squeeze(-1)
        distance_sq = 2.0 * theta**2 + s2 * (t * Mt).sum(-1)

        distance = torch.sqrt(distance_sq.clamp_min(0.0))

        # Rotation vector r = log(Q)^vee. torch.sinc(x) = sin(pi*x)/(pi*x).
        skew = Q - Q.transpose(-1, -2)
        vee = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
        sinc_theta = torch.sinc(theta / torch.pi)
        r = vee / (2.0 * sinc_theta).unsqueeze(-1)

        # beta = alpha'(theta)/theta. Its limit is -1/45 for the general
        # alpha formula, but it is ZERO on the constant-alpha branch.
        u = theta**2
        beta_series = -1.0 / 45.0 + u * (
            -13.0 / 3780.0
            + u
            * (-23.0 / 75600.0 + u * (-101.0 / 4989600.0 - u * 5263.0 / 4670265600.0))
        )
        small = theta < 0.5
        d = 2.0 * torch.sin(0.5 * theta) ** 2  # 1-cos(theta), stably
        d_safe = torch.where(small, torch.ones_like(d), d)
        beta_general = ((u - d) * sinc_theta - d) / (2.0 * d_safe**3)
        beta = torch.where(small, beta_series, beta_general)
        beta = torch.where(cos_theta > self.max_cos_theta, torch.zeros_like(beta), beta)

        Qt = (Q @ t.unsqueeze(-1)).squeeze(-1)
        # h = t^T (Q + Q^T - 2I) t = -||Qt-t||^2 on SO(3).
        h = -((Qt - t) ** 2).sum(-1)

        # Numerators of the translation and relative-rotation derivatives.
        a = s2 * Mt
        b = (-2.0 - 0.5 * s2 * h * beta).unsqueeze(-1) * r
        b = b + (s2 * alpha).unsqueeze(-1) * torch.linalg.cross(t, Qt, dim=-1)

        R = V[..., :3, :3]
        Ra = (R @ a.unsqueeze(-1)).squeeze(-1)
        Rb = (R @ b.unsqueeze(-1)).squeeze(-1)
        # Expand W's position because torch.linalg.cross requires equal ranks.
        p_W = W[..., :3, 3] + torch.zeros_like(Ra)
        numerator = torch.cat((-Ra, Rb - torch.linalg.cross(p_W, Ra, dim=-1)), dim=-1)

        nonzero = distance > 0
        denominator = torch.where(nonzero, distance, torch.ones_like(distance))
        result = numerator / denominator.unsqueeze(-1)
        result = torch.where(nonzero.unsqueeze(-1), result, torch.zeros_like(result))
        return result.unsqueeze(-1)

    def vectorfield_se3(
        self,
        states: torch.Tensor,  # (N, 4, 4)
        curve: torch.Tensor,  # (M, 4, 4)
        kt1: float,
        kt2: float,
        kt3: float,
        kn1: float,
        kn2: float,
        curve_derivative: Optional[Union[torch.Tensor, list]] = None,
        scale_pos: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (twist: (N, 6), distances: (N,), closest_indices: (N,)).
        """
        if scale_pos is None:
            scale_pos = self.scale_pos
        device, dtype = states.device, states.dtype
        N, M = states.shape[0], curve.shape[0]

        # ---- closest point on curve for each state (batched) --------------
        min_dists, closest_indices = self.ecdistance(states, curve)
        closest_points = curve[closest_indices]  # (N, 4, 4)

        # ---- normal: finite differences over 6 basis dirs (batched) ------
        I6 = self._eye(6, states)
        basis = smap_se3(I6)  # (6, 4, 4)

        # (6, 1, 4, 4) @ (1, N, 4, 4) -> (6, N, 4, 4)
        # variations = self.exp_se3(basis[:, None, :, :] * self.delta) @ states.unsqueeze(
        #     0
        # )
        #
        # variations_flat = variations.reshape(6 * N, 4, 4)
        # closest_expanded = closest_points.repeat(6, 1, 1)  # (6N, 4, 4)
        # dists_pert = self.eedistance(variations_flat, closest_expanded).reshape(6, N)
        #
        # LvDhat = (dists_pert - min_dists.unsqueeze(0)) / self.delta  # (6, N)
        # normal = -LvDhat.t().contiguous()  # (N, 6)

        # ---- normal: analytical derivative -------------------------------
        L_columns = self.L_eedistance(
            states,
            closest_points,
            scale_pos=scale_pos,
        )  # (N, 6, 1)

        normal = -L_columns.squeeze(-1).contiguous()  # (N, 6)

        LvDhat = L_columns.squeeze(-1).transpose(0, 1).contiguous()  # (6, N)

        if self.normalize_normal:
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # ---- tangent: forward-difference or supplied derivative ----------
        if curve_derivative is None:
            next_indices = (closest_indices + 1) % M  # closed curve
            dHd = (curve[next_indices] - closest_points) / self.ds  # (N, 4, 4)
        else:
            if not isinstance(curve_derivative, torch.Tensor):
                curve_derivative = torch.stack(
                    [
                        torch.as_tensor(d, device=device, dtype=dtype)
                        for d in curve_derivative
                    ]
                )
            dHd = curve_derivative[closest_indices]

        tangent = inv_smap_se3(dHd @ torch.linalg.inv(closest_points))  # (N, 6)
        if self.normalize_tangent:
            tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # ---- distance-dependent gains ------------------------------------
        eps = 1e-3
        dist_mod = torch.sqrt(min_dists + eps**2) - eps
        kn = kn1 * torch.tanh(kn2 * dist_mod)  # (N,)
        kt = kt1 * (1.0 - kt2 * torch.tanh(kt3 * dist_mod))  # (N,)

        twist = kt.unsqueeze(-1) * tangent + kn.unsqueeze(-1) * normal  # (N, 6)
        return twist, min_dists, closest_indices


# ----------------------------------------------------------------------
# Functional shortcut (mirrors the C++ signature)
# ----------------------------------------------------------------------
def vectorfield_se3(
    states: torch.Tensor,
    curve: torch.Tensor,
    kt1: float,
    kt2: float,
    kt3: float,
    kn1: float,
    kn2: float,
    curve_derivative: Optional[Union[torch.Tensor, list]] = None,
    **kwargs,
):
    return VectorFieldSE3(**kwargs).vectorfield_se3(
        states, curve, kt1, kt2, kt3, kn1, kn2, curve_derivative
    )


@torch.no_grad()
def ec_distance_and_pos_ori(
    vf_solver: VectorFieldSE3,
    states_scaled: torch.Tensor,  # (N, 4, 4) true poses, scaled
    curve_scaled: torch.Tensor,  # (M, 4, 4) curve, scaled
    states_unscaled: Optional[torch.Tensor] = None,  # (N, 4, 4) for error metrics
    curve_unscaled: Optional[torch.Tensor] = None,  # (M, 4, 4) for error metrics
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched closest-point search + pos/ori error metrics on SE(3).

    The SE(3) distance / closest index are computed in the *scaled* space
    (same convention the controller uses). Position & orientation errors
    are computed in the *unscaled* space, against the closest point found
    in scaled space, so they are consistent with the current logging
    convention but now driven by the true pose.

    Returns:
        dists:       (N,)    SE(3) distance from true pose to curve
        idx:         (N,)    closest curve index
        pos_err_cm:  (N,)    ||p_near - p_true|| in cm
        ori_err_deg: (N,)    angle(R_near, R_true) in degrees
    """
    if states_unscaled is None:
        states_unscaled = states_scaled
    if curve_unscaled is None:
        curve_unscaled = curve_scaled

    # ---- 1. brute-force closest point (chunked, fully vectorised) -----
    dists, idx = vf_solver.ecdistance(
        states_scaled, curve_scaled, chunk_size=chunk_size
    )

    # ---- 2. pos/ori errors vs the true closest point ------------------
    closest = curve_unscaled[idx]  # (N, 4, 4)
    R_near = closest[..., :3, :3]  # (N, 3, 3)
    p_near = closest[..., :3, 3]  # (N, 3)
    R_true = states_unscaled[..., :3, :3]
    p_true = states_unscaled[..., :3, 3]

    pos_err_cm = torch.linalg.norm(p_near - p_true, dim=-1) * 100.0

    # trace(R_near^T R_true) = sum(R_near * R_true) — much faster than
    # matmul + diagonal, and avoids forming the 3x3 product at all.
    trace = (R_near * R_true).sum(dim=(-2, -1))
    cos_ang = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    ori_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)

    return dists, idx, pos_err_cm, ori_err_deg
