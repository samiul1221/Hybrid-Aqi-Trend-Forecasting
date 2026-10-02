# 0021 — TI-STGNN implementation choices (N3)

Status: accepted (2026-10-02)

- **I26 GAT prior direction:** [R07.3] defines w[i, j] as source i → target j, but [R07.5] adds β·log(w_ij)
  to the logits L[i, j] where i is the RECEIVER. Literally, a receiver would favour senders downwind of it.
  `gat.py` biases L[i, j] with w[j, i] (the incoming edge). Test `test_gat_bias_uses_incoming_edge`.
- **Diffusion test:** the CR test "diffusion never increases max |U|" does not hold for the normalised
  Laplacian with unequal degrees (I + cL̂ is not diagonally dominant). We test the provable bound instead:
  the L2 norm per channel never grows (spectral norm of (I + cL̂)⁻¹ ≤ 1).
- **Invalid nodes** (no sensor data in the window, or dropped by augmentation): send nothing in the GAT,
  neither send nor receive in advection, are cut out of the diffusion graph. Their targets stay in the loss
  (the network must infer them from neighbours).
- **Diffusion basis** is computed per batch from the valid-node graph with `torch.linalg.eigh` (N ≤ 30, cheap)
  instead of once per station set, so masked nodes are handled exactly.
- Decoder uses `nn.TransformerDecoderLayer` (norm_first); the encoder is a small custom pre-norm layer
  because the built-in one cannot return attention weights (needed for L_AFD).
- PyTorch pinned to 2.14.1 (CPU wheel locally; Kaggle uses its preinstalled CUDA build).
