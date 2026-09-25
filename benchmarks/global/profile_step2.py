import time, numpy as np, cupy as cp
C, nb, F, N, E = 4096, 12, 4, 20, 22
P = 2 * E + 1
rng = np.random.default_rng(0)
G = cp.asarray(rng.normal(size=(P, C, nb, F)) + 1j * rng.normal(size=(P, C, nb, F)))
vals = cp.asarray(rng.normal(size=(C, F)) + 1j * rng.normal(size=(C, F)))
ns = np.arange(-N, N + 1)
def t(label, fn, reps=5):
    fn(); cp.cuda.Device().synchronize(); t0 = time.perf_counter()
    for _ in range(reps): out = fn()
    cp.cuda.Device().synchronize(); print(f"{label:40s} {1e3*(time.perf_counter()-t0)/reps:8.2f} ms"); return out
L2 = t("gather L2 (fancy index)", lambda: G[2 - ns + E][:, :, :, 0:F].reshape(ns.size, C, nb * F))
L3 = t("gather L3", lambda: G[ns + E][:, :, :, 0:F].reshape(ns.size, C, nb * F))
def per_a():
    part = cp.empty((ns.size, F, nb * F, nb * F), dtype=cp.complex128)
    for a in range(F):
        X = L2 * vals[:, a][None, :, None]
        part[:, a] = cp.matmul(X.transpose(0, 2, 1), L3)
    return part
t("per-a matmul (current)", per_a)
def folded():
    A = (L2[:, :, None, :] * vals[None, :, :, None]).reshape(ns.size, C, F * nb * F)
    return cp.matmul(A.transpose(0, 2, 1), L3)
t("folded nA into M", folded)
def folded_T():
    A = (L2[:, :, None, :] * vals[None, :, :, None]).reshape(ns.size, C, F * nb * F)
    At = cp.ascontiguousarray(A.transpose(0, 2, 1))
    return cp.matmul(At, L3)
t("folded + explicit transpose", folded_T)
def folded_R():
    A = (L2[:, :, None, :] * vals[None, :, :, None]).reshape(ns.size, C, F * nb * F)
    return cp.matmul(L3.transpose(0, 2, 1), A)   # (N, K3, M)
t("folded, L3^T @ A", folded_R)
x = cp.ones(1)
t("flops ref: zgemm 2x(41x192x48x4096)", lambda: cp.matmul(cp.ones((41, 192, 4096), cp.complex128), cp.ones((41, 4096, 48), cp.complex128)))
S = cp.asarray(rng.normal(size=(C, nb, 3, 2, F, F)) + 1j * rng.normal(size=(C, nb, 3, 2, F, F)))
Sv = S[:, :, 2, 0][:, :, 0:F, 0:F]
t("einsum ca,cbxy->abxy", lambda: cp.einsum("ca,cbxy->abxy", vals, Sv))
t("tensordot version", lambda: cp.tensordot(vals, Sv, axes=([0], [0])))
part = per_a()
part6 = part.reshape(ns.size, F, nb, F, nb, F)
corr = cp.einsum("ca,cbxy->abxy", vals, Sv)
def bloop():
    for b in range(nb):
        part6[:, :, b, :, b, :] -= corr[None, :, b, :, :]
t("diag loop", bloop)
out = cp.zeros((C, nb, F * P + 96), cp.complex128)
t("zeros out", lambda: cp.zeros((C, nb, F * P + 96), cp.complex128))
t("G transpose", lambda: cp.ascontiguousarray(out[:, :, :F * P].reshape(C, nb, F, P).transpose(3, 0, 1, 2)))
res = cp.zeros((ns.size, F, nb, F, nb, F), cp.complex128)
t("to_numpy final", lambda: cp.asnumpy(res.transpose(0, 1, 3, 5, 2, 4)))
