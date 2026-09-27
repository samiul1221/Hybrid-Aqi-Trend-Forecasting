"""Knowledge distillation: targets, precondition and losses ([S12], [S13.1], [S13.3], [S13.4]).

- RF (Shmuel-style output KD): y_kd = alpha * y_true + (1 - alpha) * y_teacher_oof, alpha in {1.0, 0.5, 0.0}.
- CNN out_KD: lambda * Huber(y_hat, y) + (1 - lambda) * Huber(y_hat, y_T), lambda in {0.5, 0.7}.
- CNN full_KD (Na et al. 2026 Eq. 11-17): L = lp * L_pred + ((1 - lp) / 2) * r(e) * (L_feat + L_afd),
  lp = 0.4, r(e) = min(1, e / E_warmup); L_feat = Huber(F_S, psi(F_T)); L_afd weights the per-patch feature
  error by M = softmax(z-score(sum_i mean_heads A_T[i, j])); psi = trainable Linear(D_T -> D), discarded after.
- Precondition [S12.1]: distil only where the teacher beats the no-KD student on validation, per horizon.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

F32 = npt.NDArray[np.float32]
ALPHAS = (1.0, 0.5, 0.0)  # [S13.2]
LAMBDAS = (0.5, 0.7)  # [S13.4] out_KD
LAMBDA_P = 0.4  # [S13.3] Na et al.
E_WARMUP = 10


def rf_targets(y_true: F32, y_teacher: F32, alpha: float) -> F32:
    """Blended RF targets; windows without a teacher row keep the true target."""
    y_t = np.where(np.isfinite(y_teacher), y_teacher, y_true)
    return (alpha * y_true + (1.0 - alpha) * y_t).astype(np.float32)


def model_b_teacher(y_teacher_a: F32, ln_pm_t: F32, ln_pm_last: F32) -> F32:
    """[S14]: teacher forecasts from base t re-expressed relative to the last valid PM (t - k):
    y_T' = ln(1+PM_T[t+h]) - ln(1+PM_last) = y_T + ln(1+PM_t) - ln(1+PM_last), per pollutant."""
    h = y_teacher_a.shape[1] // 2
    shift = np.repeat(ln_pm_t - ln_pm_last, h, axis=1)  # [N x 2] -> pollutant-major [N x 2H]
    return (y_teacher_a + shift).astype(np.float32)


def precondition(teacher_rmse: dict[str, float], student_rmse: dict[str, float]) -> dict[str, Any]:
    """Per output: teacher RMSE < no-KD student RMSE (validation). KD is used only where True."""
    per = {k: {"teacher": teacher_rmse[k], "student": student_rmse[k], "pass": teacher_rmse[k] < student_rmse[k]}
           for k in student_rmse}  # fmt: skip
    return {"per_output": per, "all_pass": all(v["pass"] for v in per.values())}


def huber(a: Any, b: Any, delta: float = 1.0) -> Any:
    d = tf.abs(a - b)
    return tf.where(d <= delta, 0.5 * d * d, delta * (d - 0.5 * delta))


def out_kd_loss(y_true: Any, y_pred: Any, lam: float, n_out: int) -> Any:
    """y_true packs [y (n_out) | y_teacher (n_out)] (both already normalised like the student target)."""
    y, y_t = y_true[:, :n_out], y_true[:, n_out:]
    return lam * tf.reduce_mean(huber(y_pred, y)) + (1.0 - lam) * tf.reduce_mean(huber(y_pred, y_t))


def afd_mask(attn_patch: Any) -> Any:
    """[B x heads x L x L] patch attention -> M [B x L] = softmax(z-score(sum_i mean_h A[i, j])) (Na Eq. 13-14)."""
    w = tf.reduce_sum(tf.reduce_mean(attn_patch, axis=1), axis=1)
    mu, sd = tf.reduce_mean(w, axis=1, keepdims=True), tf.math.reduce_std(w, axis=1, keepdims=True)
    return tf.nn.softmax((w - mu) / (sd + 1e-6), axis=1)


def full_kd_loss(y: Any, y_hat: Any, f_s: Any, f_t_proj: Any, attn_patch: Any | None, epoch: int,
                 lp: float = LAMBDA_P, e_warmup: int = E_WARMUP) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    """Na et al. total loss. f_s, f_t_proj: [B x L x D]. Without attention: L_afd = 0 ([S12.2])."""
    l_pred = tf.reduce_mean(huber(y_hat, y))
    err = huber(f_s, f_t_proj)  # [B x L x D]
    l_feat = tf.reduce_mean(err)
    if attn_patch is None:
        l_afd = tf.constant(0.0)
    else:
        m = afd_mask(attn_patch)  # [B x L]
        l_afd = tf.reduce_mean(tf.reduce_sum(m[:, :, None] * err, axis=1)) / tf.cast(
            tf.shape(err)[1], err.dtype
        )
    r = min(1.0, epoch / float(e_warmup))
    total = lp * l_pred + ((1.0 - lp) / 2.0) * r * (l_feat + l_afd)
    return total, {"l_pred": l_pred, "l_feat": l_feat, "l_afd": l_afd, "r": r}


class FullKDTrainer:
    """Custom training loop for full_KD: student feature model + trainable psi (Linear D_T -> D)."""

    def __init__(self, feature_model: keras.Model, d_teacher: int, lr: float = 1e-3) -> None:
        self.model = feature_model
        d = feature_model.outputs[1].shape[-1]
        self.psi = keras.layers.Dense(d, name="psi")  # discarded after training ([S13.4])
        self.psi.build((None, None, d_teacher))
        self.opt = keras.optimizers.Adam(lr)

    def step(self, x: F32, y: F32, f_t: F32, attn: F32 | None, epoch: int) -> dict[str, float]:
        with tf.GradientTape() as tape:
            y_hat, f_s = self.model(x, training=True)
            total, parts = full_kd_loss(y, y_hat, f_s, self.psi(f_t), attn, epoch)
        variables = self.model.trainable_variables + self.psi.trainable_variables
        self.opt.apply_gradients(zip(tape.gradient(total, variables), variables, strict=True))
        return {"total": float(total), **{k: float(v) for k, v in parts.items()}}
