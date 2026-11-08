"""Track D: data-free readout of the mirror structure from the weights alone (claim C6, P5, gate G4).

D1  Layer-0 operator from the patch embedding and the position embeddings only.
D2  MLP twin pairing from the fc1 rows only, given an operator A.
D3  (stretch) Propagation of the weight-only cross-moment through blocks, giving A_1, A_2, ...

Conventions follow mirror/fit.py, mirror/matching.py and mirror/surgery.py. Tokens are row vectors
of width d; A is a symmetric orthogonal involution; "channel-centred" means multiplied by
J = I - 11^T / d, exactly as mirror.stats does before accumulating. The cross-moment of the data
path is C = sum_t x_t^T xt_t, where xt_t is the mirrored image's token at the mirrored position.
Twins satisfy w_{pi(i)} = A w_i (P W1 = W1 A, fc1 rows w_i of the folded W1).

D1. The layer-0 token of a patch p at grid position (r, c) is x = W vec(p) + k(r, c), where W is the
patch-embedding matrix [d, 3 P P] and k(r, c) the constant part (patch bias plus position embedding;
for prefix tokens, the CLS or register token and its position embedding). The mirrored image puts
the horizontally flipped patch at (r, W - 1 - c), so xt = W F vec(p) + k(r, W - 1 - c), with F the
pixel permutation that flips a patch (W F = the flipped kernels). With an isotropic prior on patch
content (vec(p) zero mean, covariance sigma^2 I, independent of position) the expected cross-moment
of one image is
    C_w = sigma^2 HW  J W F W^T J  +  sum_t J k_t kt_t^T J ,
the second sum running over every token (patches, CLS, registers), which is what the data path sums.
The only free number is sigma^2, which fixes the weighting between the two terms. Equivalently, the
content energy share
    rho = content energy / (content + constant-part energy),
content energy = sigma^2 HW ||J W||_F^2 and constant-part energy = sum_t ||J k_t||^2.
Primary choice: sigma^2 = 1, an isotropic N(0, I) prior on the normalised pixels. ImageNet mean/std
normalisation (DINOv2) gives every pixel unit variance over the data set, so this prior matches the
marginal moments the preprocessing fixes and ignores only the correlations between pixels; it uses
no data and sets the absolute scale of the stream, which D3 needs. Sensitivity choices: rho = 1/2
(equal energy) and a data-calibrated rho. A_0^w = fit_involution(C_w). If the embedding is exactly
mirror-equivariant (W F = A W, k(r, W-1-c) = A k(r, c)), C_w = M A with M PSD and commuting with A,
so A_0^w = A whatever sigma^2 is: the weighting only matters where the two terms disagree.

D2. S[i, j] = cos(w_i, A w_j) (symmetric because A is), then the same involution matching as
mirror.matching (edge weights 2 S_ij - S_ii - S_jj, top-k candidates, exact blossom).

D3. A moment model per token: state (C, G, mu) = centred cross-moment, centred second moment and
mean of the stream. Attention is replaced by its value-output map with the attention mixing taken
as the identity (x_mid = x + M LN(x) + m0, M = Wo Wv after folding norm1 and LayerScale), which is
linear, so the moments pass through exactly. The MLP output y = W2 u + b2 with u = GELU(z) is
modelled with independent Gaussian pre-activations z_j (mean and variance from the state), the twin
relation ut = P u from the given pairing, and no correlation between the MLP output and the stream
it reads (cross terms between residual and MLP output reduce to products of means). Then
C_{l+1} = C_mid + C_y + mean terms, centred; A_{l+1}^w = fit_involution(C_{l+1}).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .attention import fold_attn
from .fit import fit_involution, random_orthogonal
from .matching import MatchResult, match_many, random_pairing
from .stats import unflip_tokens
from .surgery import fold_mlp, write_absorbed_mlp


# ----------------------------------------------------------------------------------------------
# Linear algebra: involutions and principal angles
# ----------------------------------------------------------------------------------------------
def centre_rows(X: np.ndarray) -> np.ndarray:
    """Channel-centre row vectors (X J): subtract each row's mean over the last axis."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=-1, keepdims=True)


def centre_both(C: np.ndarray) -> np.ndarray:
    """J C J for a d x d matrix."""
    C = np.asarray(C, dtype=np.float64)
    C = C - C.mean(axis=0, keepdims=True)
    return C - C.mean(axis=1, keepdims=True)


def clean_involution(A: np.ndarray) -> tuple[np.ndarray, dict]:
    """Nearest exact symmetric orthogonal involution: V sign(lam) V^T of the symmetric part.

    Used on the float32 operators stored by scripts/run_e1.py. Returns (A_exact, info) with the
    largest deviation of an eigenvalue of the stored matrix from +-1.
    """
    A = np.asarray(A, dtype=np.float64)
    S = 0.5 * (A + A.T)
    lam, V = np.linalg.eigh(S)
    signs = np.where(lam > 0, 1.0, -1.0)
    out = (V * signs) @ V.T
    out = 0.5 * (out + out.T)
    info = {"max_eig_dev": float(np.max(np.abs(np.abs(lam) - 1.0))) if lam.size else 0.0,
            "asym": float(np.max(np.abs(A - A.T))) if A.size else 0.0,
            "d_plus": int((signs > 0).sum()), "d_minus": int((signs < 0).sum())}
    return out, info


def eigenspaces(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal bases (columns) of the +1 and -1 eigenspaces of an involution."""
    A = np.asarray(A, dtype=np.float64)
    lam, V = np.linalg.eigh(0.5 * (A + A.T))
    return V[:, lam > 0], V[:, lam <= 0]


def principal_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between span(U1) and span(U2) (orthonormal columns),
    in decreasing order; min(k1, k2) values, clipped to [0, 1]."""
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def compare_operators(A_ref: np.ndarray, A_test: np.ndarray) -> dict:
    """Agreement of two involutions.

    mean_cos_even / mean_cos_odd: mean cosine of the principal angles between the +1 (resp. -1)
    eigenspaces, over min(k_ref, k_test) angles (the P5 measure). overlap_*: ||U_ref^T U_test||_F^2
    / max(k_ref, k_test), which also penalises a dimension mismatch. trace_agreement: tr(A B) / d.
    """
    Pr, Mr = eigenspaces(A_ref)
    Pt, Mt = eigenspaces(A_test)
    ce, co = principal_cosines(Pr, Pt), principal_cosines(Mr, Mt)
    d = A_ref.shape[0]

    def overlap(U, V):
        k = max(U.shape[1], V.shape[1])
        return float(np.sum((U.T @ V) ** 2) / k) if k else float("nan")

    return {
        "d_plus_ref": int(Pr.shape[1]), "d_plus_test": int(Pt.shape[1]),
        "d_minus_ref": int(Mr.shape[1]), "d_minus_test": int(Mt.shape[1]),
        "mean_cos_even": float(ce.mean()) if ce.size else float("nan"),
        "mean_cos_odd": float(co.mean()) if co.size else float("nan"),
        "min_cos_even": float(ce.min()) if ce.size else float("nan"),
        "min_cos_odd": float(co.min()) if co.size else float("nan"),
        "overlap_even": overlap(Pr, Pt), "overlap_odd": overlap(Mr, Mt),
        "trace_agreement": float(np.trace(np.asarray(A_ref, np.float64) @ np.asarray(A_test, np.float64)) / d),
    }


def random_involution_fixing_ones(d: int, d_plus: int, seed: int) -> np.ndarray:
    """Random orthogonal involution with A 1 = 1 (so it commutes with channel centring) and
    d_plus eigenvalues +1 (the all-ones direction is one of them)."""
    if not 1 <= d_plus <= d:
        raise ValueError("need 1 <= d_plus <= d")
    rng = np.random.default_rng(seed)
    one = np.ones(d) / math.sqrt(d)
    Q, _ = np.linalg.qr(np.concatenate([one[:, None], rng.standard_normal((d, d - 1))], axis=1))
    B = Q[:, 1:]  # orthonormal basis of the complement of the ones direction
    R = random_orthogonal(d - 1, rng)
    signs = np.concatenate([np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = np.outer(one, one) + B @ (R * signs) @ R.T @ B.T
    return 0.5 * (A + A.T)


# ----------------------------------------------------------------------------------------------
# D1: layer-0 operator from the weights
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def layer0_parts(adapter, check_seed: int = 0) -> dict:
    """Weight-only ingredients of the layer-0 tokens, all channel-centred, float64 numpy.

    W, Wf   [d, 3 P P]  patch-embedding matrix and the same with each kernel flipped left-right
    K, Kt   [T, d]      constant part of every token (zero-content image) and the same at the
                        mirrored position (prefix tokens unchanged)
    pos, pos_t [HW, d]  patch position embeddings and the same at (r, W - 1 - c)
    Also checks that the adapter's layer-0 map is affine with exactly this patch-embedding part
    (max abs difference relative to the token scale, on one seeded random input).
    """
    m = adapter.model
    proj = m.patch_embed.proj
    wt = proj.weight.detach().to(torch.float64).cpu()  # [d, C, P, P]
    d = wt.shape[0]
    H, Wg = adapter.grid_hw
    P = adapter.patch_size
    if tuple(proj.kernel_size) != (P, P) or tuple(proj.stride) != (P, P):
        raise ValueError("patch embedding must be a P x P convolution with stride P")
    npre = adapter.n_prefix_tokens
    dtype, device = adapter.dtype, adapter.device
    S = adapter.img_size

    zero = torch.zeros(1, 3, S, S, dtype=dtype, device=device)
    K = adapter.layer0_tokens(zero)  # [1, T, d]
    g = torch.Generator().manual_seed(check_seed)
    x = torch.randn(1, 3, S, S, generator=g, dtype=torch.float64).to(dtype=dtype, device=device)
    tok = adapter.layer0_tokens(x)
    conv = torch.nn.functional.conv2d(x, proj.weight, None, stride=P)  # [1, d, H, W]
    conv = conv.flatten(2).transpose(1, 2)  # [1, HW, d] row-major (H, W) like the token order
    diff_patch = (tok[:, npre:] - K[:, npre:] - conv).abs().max()
    diff_prefix = (tok[:, :npre] - K[:, :npre]).abs().max() if npre else torch.zeros(())
    scale = tok.abs().max().clamp_min(1e-30)
    affine_rel_err = float(max(diff_patch, diff_prefix) / scale)

    Kt = unflip_tokens(K, (H, Wg), npre)
    pos = adapter.pos_embed_grid().detach().to(torch.float64).cpu()  # [H, W, d]
    W = wt.reshape(d, -1).numpy()
    Wf = wt.flip(-1).reshape(d, -1).numpy()
    return {
        "W": W - W.mean(axis=0, keepdims=True),       # J W (centre over the d output channels)
        "Wf": Wf - Wf.mean(axis=0, keepdims=True),
        "K": centre_rows(K[0].to(torch.float64).cpu().numpy()),
        "Kt": centre_rows(Kt[0].to(torch.float64).cpu().numpy()),
        "pos": centre_rows(pos.reshape(H * Wg, d).numpy()),
        "pos_t": centre_rows(pos.flip(1).reshape(H * Wg, d).numpy()),
        "HW": H * Wg, "n_prefix": npre, "d": d, "grid": (H, Wg), "patch": P,
        "affine_rel_err": affine_rel_err,
    }


def layer0_moments(parts: dict, sigma2: float | None = 1.0, rho: float | None = None,
                   terms: str = "full") -> dict:
    """Weight-only layer-0 moments per token under the isotropic content prior N(0, sigma2 I).

    terms = "full": constant part of every token (patch bias + position, CLS and registers), the
    exact analogue of the data path. terms = "position": only sum_(r,c) pos(r,c)^T pos(r,W-1-c)
    (the literal formula, no bias and no prefix tokens).
    Weighting: either sigma2 (default 1, the primary choice) or rho = content energy share (then
    sigma2 is derived; rho = 1 keeps only the content term, rho = 0 only the constant term).
    Returns C (cross-moment), G (second moment), mu (mean), all centred and per token, plus sigma2
    and the effective rho.
    """
    if rho is not None:
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
    elif sigma2 is None or sigma2 < 0:
        raise ValueError("give sigma2 >= 0 or rho")
    if terms == "full":
        K, Kt = parts["K"], parts["Kt"]
    elif terms == "position":
        K, Kt = parts["pos"], parts["pos_t"]
    else:
        raise ValueError(f"unknown terms {terms}")
    T = K.shape[0]
    HW, W, Wf = parts["HW"], parts["W"], parts["Wf"]
    Gc = HW * (W @ W.T)        # content second moment of one image, sigma^2 = 1
    Cc = HW * (W @ Wf.T)       # content cross-moment of one image, sigma^2 = 1
    Gk, Ck = K.T @ K, K.T @ Kt
    Ec, Ek = float(np.trace(Gc)), float(np.trace(Gk))
    if rho is None:
        a, b = float(sigma2), 1.0
    elif rho < 1.0:
        a, b = rho / (1.0 - rho) * Ek / Ec, 1.0
    else:
        a, b = 1.0, 0.0
    C = (a * Cc + b * Ck) / T
    G = (a * Gc + b * Gk) / T
    mu = b * K.sum(axis=0) / T
    rho_eff = a * Ec / (a * Ec + b * Ek)
    return {"C": C, "G": G, "mu": mu, "sigma2": a, "rho": rho_eff, "terms": terms, "n_tokens": T,
            "E_content_img": a * Ec, "E_const_img": b * Ek}


def fit_layer0(parts: dict, sigma2: float | None = 1.0, rho: float | None = None, terms: str = "full") -> dict:
    mom = layer0_moments(parts, sigma2=sigma2, rho=rho, terms=terms)
    A, dp, dm, info = fit_involution(mom["C"], return_info=True)
    return {"A": A, "d_plus": dp, "d_minus": dm, "n_zero": info["n_zero"], "moments": mom}


def content_share_from_data(trace_G0: float, n_images: int, parts: dict) -> dict:
    """Data-calibrated content energy share (secondary, not data-free).

    tr(G_0) of the data path = sum over fit tokens of ||J x||^2; the constant parts carry
    n_images * sum_t ||J k_t||^2 of it exactly, so rho_data = 1 - that / tr(G_0). The remainder
    includes content-constant cross terms (patch content is not exactly zero-mean), so rho is only
    clipped to stay strictly inside (0, 1)."""
    Ek = float(np.sum(parts["K"] ** 2))
    raw = 1.0 - n_images * Ek / float(trace_G0)
    return {"rho_raw": raw, "rho": float(min(max(raw, 1e-6), 1.0 - 1e-6)), "E_const_img": Ek}


# ----------------------------------------------------------------------------------------------
# D2: twin pairing from the weights
# ----------------------------------------------------------------------------------------------
def twin_scores(W1, A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """S[i, j] = cos(w_i, A w_j) for the rows w_i of the folded fc1 weight [h, d] (symmetrised)."""
    W = W1.detach().cpu().numpy() if isinstance(W1, torch.Tensor) else np.asarray(W1)
    W = W.astype(np.float64)
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), eps)
    S = Wn @ np.asarray(A, dtype=np.float64) @ Wn.T
    return 0.5 * (S + S.T)


def match_twins(S_list: list[np.ndarray], top_k: int = 10, workers: int = 1) -> list[MatchResult]:
    """Involution matching exactly as mirror.matching (S is already symmetric)."""
    return match_many(S_list, top_k=top_k, workers=workers)


def pair_set(perm) -> set:
    perm = np.asarray(perm)
    idx = np.flatnonzero(perm != np.arange(len(perm)))
    return {(int(i), int(perm[i])) for i in idx if i < perm[i]}


def pairing_agreement(perm_ref, perm_test) -> dict:
    """Recovery of the reference pairs by the test pairing (pairs equal as unordered sets).

    recovery = common pairs / reference pairs (the P5 measure); precision = common / test pairs;
    self_kept = reference self-twins that are self-twins in the test pairing / reference
    self-twins; chance_recovery = expected recovery of a random involution with the test pairing's
    number of pairs (n_test / C(h, 2)).
    """
    perm_ref, perm_test = np.asarray(perm_ref), np.asarray(perm_test)
    h = len(perm_ref)
    if len(perm_test) != h:
        raise ValueError("pairings of different sizes")
    for p in (perm_ref, perm_test):
        if not np.array_equal(p[p], np.arange(h)):
            raise ValueError("pairing is not an involution")
    R, T = pair_set(perm_ref), pair_set(perm_test)
    common = len(R & T)
    self_r = perm_ref == np.arange(h)
    self_t = perm_test == np.arange(h)
    return {
        "h": h, "n_pairs_ref": len(R), "n_pairs_test": len(T), "n_common": common,
        "recovery": common / len(R) if R else float("nan"),
        "precision": common / len(T) if T else float("nan"),
        "n_self_ref": int(self_r.sum()), "n_self_test": int(self_t.sum()),
        "self_kept": float((self_r & self_t).sum() / self_r.sum()) if self_r.any() else float("nan"),
        "chance_recovery": len(T) / (h * (h - 1) / 2) if h > 1 else float("nan"),
    }


def pair_score_summary(S: np.ndarray, perm) -> dict:
    """Mean weight score over the pairs and over the self-twins of a pairing."""
    perm = np.asarray(perm)
    h = len(perm)
    paired = perm != np.arange(h)
    vals = S[np.arange(h), perm]
    return {"mean_S_pairs": float(vals[paired].mean()) if paired.any() else float("nan"),
            "mean_S_self": float(vals[~paired].mean()) if (~paired).any() else float("nan")}


# ----------------------------------------------------------------------------------------------
# D3 (stretch): moment propagation through blocks
# ----------------------------------------------------------------------------------------------
def act_moments(mean: np.ndarray, var: np.ndarray, act: nn.Module | None = None, n_nodes: int = 64):
    """E[g(z)] and Var[g(z)] for z ~ N(mean, var) per entry (Gauss-Hermite quadrature).

    g is the block's activation module (exact GELU when None)."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / math.sqrt(2.0 * math.pi)
    z = np.asarray(mean, np.float64)[:, None] + np.sqrt(np.maximum(np.asarray(var, np.float64), 0.0))[:, None] * x[None]
    f = act if act is not None else nn.GELU()
    with torch.no_grad():
        g = f(torch.from_numpy(z)).numpy()
    nu = g @ w
    q = np.maximum((g * g) @ w - nu * nu, 0.0)
    return nu, q


@torch.no_grad()
def propagate_block(block: nn.Module, state: dict, perm, n_nodes: int = 64) -> dict:
    """One block of the moment model (see the module docstring). state = {C, G, mu}, centred, per token."""
    C, G, mu = (np.asarray(state[k], np.float64) for k in ("C", "G", "mu"))
    d = C.shape[0]
    perm = np.asarray(perm)
    # Attention branch as its value-output map (norm1 affine and ls1 folded in).
    fa = fold_attn(block)
    M = (fa["Wo"] @ fa["Wv"]).cpu().numpy()
    m0 = (fa["Wo"] @ fa["bv"] + fa["bo"]).cpu().numpy()
    s1 = math.sqrt(d / max(float(np.trace(G)), 1e-300))
    T1 = np.eye(d) + s1 * M
    C_mid = T1 @ C @ T1.T + np.outer(T1 @ mu, m0) + np.outer(m0, T1 @ mu) + np.outer(m0, m0)
    G_mid = T1 @ G @ T1.T + np.outer(T1 @ mu, m0) + np.outer(m0, T1 @ mu) + np.outer(m0, m0)
    mu_mid = T1 @ mu + m0
    C_mid, G_mid, mu_mid = centre_both(C_mid), centre_both(G_mid), centre_rows(mu_mid)
    # MLP branch: independent Gaussian pre-activations, twins swap under the mirror.
    fm = fold_mlp(block)
    W1, b1 = fm["W1"].cpu().numpy(), fm["b1"].cpu().numpy()
    W2, b2 = fm["W2"].cpu().numpy(), fm["b2"].cpu().numpy()
    s2 = math.sqrt(d / max(float(np.trace(G_mid)), 1e-300))
    Sig = G_mid - np.outer(mu_mid, mu_mid)
    mean_z = s2 * (W1 @ mu_mid) + b1
    var_z = s2 * s2 * ((W1 @ Sig) * W1).sum(axis=1)
    act = getattr(getattr(block, "mlp", None), "act", None)
    nu, q = act_moments(mean_z, var_z, act, n_nodes)
    W2nu, W2nu_p = W2 @ nu, W2 @ nu[perm]
    C_y = (W2 * q[None, :]) @ W2[:, perm].T + np.outer(W2nu, W2nu_p) + np.outer(W2nu, b2) + np.outer(b2, W2nu_p) + np.outer(b2, b2)
    G_y = (W2 * q[None, :]) @ W2.T + np.outer(W2nu, W2nu) + np.outer(W2nu, b2) + np.outer(b2, W2nu) + np.outer(b2, b2)
    m_y, m_yt = W2nu + b2, W2nu_p + b2
    C_new = C_mid + C_y + np.outer(mu_mid, m_yt) + np.outer(m_y, mu_mid)
    G_new = G_mid + G_y + np.outer(mu_mid, m_y) + np.outer(m_y, mu_mid)
    return {"C": centre_both(C_new), "G": centre_both(G_new), "mu": centre_rows(mu_mid + m_y),
            "energy_mid": float(np.trace(G_mid)), "energy_mlp": float(np.trace(centre_both(G_y)))}


def propagate_chain(blocks, state0: dict, n_blocks: int, perms: list | None = None, top_k: int = 10) -> dict:
    """Data-free chain A_0^w -> P_1 -> A_1^w -> P_2 -> ... over the first n_blocks blocks.

    perms[l] gives the pairing of block l (0-based); None (or no list) means: pair from the
    weights with the current data-free operator (D2). Returns operators A[0..n], the pairings
    used, the per-block pairings found from the weights, and the states.
    """
    states = [state0]
    A0, _, _ = fit_involution(state0["C"])
    ops, used = [A0], []
    for l in range(n_blocks):
        p = perms[l] if perms is not None and l < len(perms) else None
        if p is None:
            S = twin_scores(fold_mlp(blocks[l])["W1"], ops[-1])
            p = match_twins([S], top_k=top_k, workers=1)[0].perm
        used.append(np.asarray(p))
        st = propagate_block(blocks[l], states[-1], p)
        states.append(st)
        ops.append(fit_involution(st["C"])[0])
    return {"A": ops, "perms": used, "states": states}


# ----------------------------------------------------------------------------------------------
# Planted mirror-symmetric tiny model (tests and --tiny)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def plant_mirror_symmetry(adapter, A: np.ndarray, seed: int = 0, n_pairs: int | None = None,
                          block: int = 0) -> np.ndarray:
    """Make a tiny timm ViT's layer 0 and one block exactly mirror-equivariant with operator A.

    Patch kernels: W <- (W + A W F) / 2, so the kernels come in flipped pairs (W F = A W); patch bias
    and CLS (with its position embedding) even; position embeddings mirror-consistent,
    pos(r, W-1-c) = A pos(r, c). In `block`, LayerScale ls1 is set to 0 (attention branch off) and
    the MLP is projected onto exact twins for a random pairing with n_pairs pairs (written in
    absorbed form). A must satisfy A 1 = 1 (see random_involution_fixing_ones). Returns the pairing.
    """
    m = adapter.model
    if getattr(m, "reg_token", None) is not None or getattr(m, "no_embed_class", False):
        raise ValueError("planting supports the plain timm ViT (CLS token, no registers)")
    dt = m.pos_embed.dtype
    At = torch.as_tensor(np.asarray(A), dtype=dt)
    d = At.shape[0]
    if not torch.allclose(At @ torch.ones(d, dtype=dt), torch.ones(d, dtype=dt), atol=1e-8):
        raise ValueError("A must fix the all-ones direction")
    pe = m.patch_embed.proj
    w = pe.weight
    pe.weight.copy_(0.5 * (w + torch.einsum("ab,bcij->acij", At, w.flip(-1))))
    if pe.bias is not None:
        pe.bias.copy_(0.5 * (pe.bias + At @ pe.bias))
    H, Wg = adapter.grid_hw
    pos = m.pos_embed.data[0]
    npre = adapter.n_prefix_tokens
    grid = pos[npre:].reshape(H, Wg, d)
    pos[npre:] = (0.5 * (grid + grid.flip(1) @ At)).reshape(H * Wg, d)
    for i in range(npre):
        pos[i] = 0.5 * (pos[i] + At @ pos[i])
    m.cls_token.data[0, 0] = 0.5 * (m.cls_token.data[0, 0] + At @ m.cls_token.data[0, 0])
    blk = m.blocks[block]
    blk.ls1.gamma.zero_()
    f = fold_mlp(blk, dtype=dt)
    h = f["W1"].shape[0]
    perm = random_pairing(h, h // 3 if n_pairs is None else n_pairs, seed=seed)
    p = torch.as_tensor(perm, dtype=torch.long)
    W1 = 0.5 * (f["W1"] + f["W1"][p] @ At)
    b1 = 0.5 * (f["b1"] + f["b1"][p])
    W2 = 0.5 * (f["W2"] + (At @ f["W2"])[:, p])
    b2 = 0.5 * (f["b2"] + At @ f["b2"])
    write_absorbed_mlp(blk, W1, b1, W2, b2)
    return perm
