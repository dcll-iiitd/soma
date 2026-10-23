"""OP-SOMA-PostNS5-Frob: OP-SOMA-PostNS5 with the Frobenius norm instead of the spectral norm in the final normalisation.

Identical to OPSOMAPostNS5 (optimizers/soma_projected.py) in every step except the last line of the perturbation:
    OP-SOMA-PostNS5      : E_i = rho * Q_i / ||Q_i||_op     (spectral norm = largest singular value, needs an SVD)
    OP-SOMA-PostNS5-Frob : E_i = rho * Q_i / ||Q_i||_F      (Frobenius norm, one reduction, no SVD)
with Q_i = P_perp(O_prev_i, rho * NS5(G_prev_i - sigma (1 - beta) M_i)) and E_i = 0 if Q_i = 0.
Note that ||E_i||_F = rho (so ||E_i||_op <= rho), whereas the spectral variant has ||E_i||_op = rho exactly; for the same rho the
Frobenius variant therefore perturbs less in spectral norm. rho is NOT re-tuned for this variant.
"""
import torch

from optimizers.soma_projected import OPSOMAPostNS5, ns5_f64, p_perp, Q_ZERO_RTOL


class OPSOMAPostNS5Frob(OPSOMAPostNS5):
    def _perturbation(self, group, p, state):
        assert "G_prev" in state and "M" in state and "O_prev" in state, \
            "OP-SOMA-PostNS5-Frob: t >= 2 needs G_prev, M and O_prev from step t - 1"
        d = state["G_prev"].double() - group["fsam_sigma"] * (1.0 - group["momentum"]) * state["M"]
        z = group["rho"] * ns5_f64(d)
        q = p_perp(state["O_prev"], z)
        qn = q.norm()                                    # Frobenius norm
        if qn.item() <= Q_ZERO_RTOL * z.norm().item():
            return torch.zeros_like(q)
        return group["rho"] * q / qn
