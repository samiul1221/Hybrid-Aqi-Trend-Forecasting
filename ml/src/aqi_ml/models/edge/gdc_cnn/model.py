"""GDC-CNN student [S13.3] (Na et al. 2026) / [S13.4] (our configuration). ADR 0010.

Preprocessing (instance norm, patching, target de-norm) is done OUTSIDE the graph (C on device,
`preprocess.py` in Python) so the TFLite graph only holds TFLM-supported ops ([S13.4] tflm_rules).
"""

from __future__ import annotations

from typing import Any

import keras
from keras import layers, ops

DILATIONS = (1, 2, 4)  # receptive field 1 + 2*(1+2+4) = 15 patches >= L = 12
KERNEL = 3


@keras.saving.register_keras_serializable(package="aqi")
class PositionalEmbedding(layers.Layer):  # type: ignore[misc]
    """Learned positional embedding [L x D] added to the patch encodings ([S13.4] step 3)."""

    def build(self, input_shape: Any) -> None:
        self.pos = self.add_weight(
            shape=(input_shape[1], input_shape[2]), initializer="zeros", name="pos"
        )

    def call(self, x: Any) -> Any:
        return x + self.pos


@keras.saving.register_keras_serializable(package="aqi")
class GLU(layers.Layer):  # type: ignore[misc]
    """Gated linear unit A * sigmoid(B) over the last axis (TFLM: STRIDED_SLICE + LOGISTIC + MUL)."""

    def call(self, x: Any) -> Any:
        d = x.shape[-1] // 2
        return x[..., :d] * ops.sigmoid(x[..., d:])


def gdc_block(x: Any, d_model: int, dilation: int, name: str, glu: bool = True,
              layernorm: bool = True, depthwise: bool = True) -> Any:  # fmt: skip
    """LayerNorm -> causal dilated depthwise conv -> PW conv -> GLU -> PW conv + residual (Na Eqs. 6-9).

    Flags exist only for the [S21] E8 ablations (GLU -> ReLU, depthwise -> standard conv).
    """
    h = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln")(x) if layernorm else x
    h = layers.ZeroPadding1D(padding=((KERNEL - 1) * dilation, 0), name=f"{name}_causal_pad")(h)
    if depthwise:
        h = layers.DepthwiseConv1D(
            KERNEL, dilation_rate=dilation, padding="valid", name=f"{name}_dwconv"
        )(h)
    else:
        h = layers.Conv1D(
            d_model, KERNEL, dilation_rate=dilation, padding="valid", name=f"{name}_conv"
        )(h)
    if glu:
        h = layers.Dense(2 * d_model, name=f"{name}_pw1")(h)
        h = GLU(name=f"{name}_glu")(h)
    else:
        h = layers.Dense(d_model, activation="relu", name=f"{name}_pw1")(h)
    h = layers.Dense(d_model, name=f"{name}_pw2")(h)
    return layers.Add(name=f"{name}_res")([x, h])


def build(n_patches: int, patch_dim: int, d_model: int, outputs: int = 6, blocks: tuple[int, ...] = DILATIONS,
          batch_size: int | None = None, **ablation: bool) -> tuple[keras.Model, keras.Model]:  # fmt: skip
    """Returns (student, feature_model). student: patches [L, P*C] -> y_norm [outputs].
    feature_model additionally returns F_S = last GDC output [L, D] (for full_KD, [S13.4] step 5).
    batch_size=1 builds the export graph: static shapes, so no SHAPE/GATHER/REDUCE_PROD ops (not in TFLM)."""
    inp = keras.Input(shape=(n_patches, patch_dim), batch_size=batch_size, name="patches")
    x = layers.Dense(d_model, name="encoder")(inp)  # shared across patches ([S13.4] step 3)
    x = PositionalEmbedding(name="pos_emb")(x)
    for i, dil in enumerate(blocks):
        x = gdc_block(x, d_model, dil, f"gdc{i}", **ablation)
    feats = x
    y = layers.Dense(outputs, name="decoder")(layers.Flatten(name="flatten")(feats))
    return keras.Model(inp, y, name="gdc_cnn"), keras.Model(
        inp, [y, feats], name="gdc_cnn_features"
    )


def export_model(trained: keras.Model, n_patches: int, patch_dim: int, d_model: int,
                 blocks: tuple[int, ...] = DILATIONS, **ablation: bool) -> keras.Model:  # fmt: skip
    """Same architecture with batch 1 (static shapes for TFLM) carrying the trained weights."""
    m, _ = build(
        n_patches, patch_dim, d_model, trained.output_shape[-1], blocks, batch_size=1, **ablation
    )
    m.set_weights(trained.get_weights())
    return m
