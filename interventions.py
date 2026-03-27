"""
Intervention implementations for all categories (W1-W5, A1-A3).
Each intervention returns a dict of modifications (for Class W) or
registers/deregisters hooks (for Class A).
"""
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import numpy as np

from weight_manager import WeightManager


# =============================================================================
# CLASS W: WEIGHT-SPACE INTERVENTIONS
# =============================================================================

def w1_permutation(
    wm: WeightManager,
    layer_idx: int,
    component: str,  # e.g., "self_attn.q_proj.weight"
    strategy: str,  # "random", "l2_asc", "l2_desc", "reverse", "interleave"
    coupled: bool,
    seed: int = 42,
    calibration_activations: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """W1: Intra-layer weight permutation.

    Args:
        wm: WeightManager
        layer_idx: which layer
        component: which weight matrix (short name within the layer)
        strategy: permutation ordering strategy
        coupled: if True, apply inverse permutation to downstream matrix
        seed: random seed for reproducible permutations

    Returns:
        dict of param_name -> modified tensor
    """
    rng = np.random.RandomState(seed)
    params = wm.get_layer_param_names(layer_idx)
    full_name = params.get(component)
    if full_name is None:
        raise ValueError(f"Component {component} not found in layer {layer_idx}. Available: {list(params.keys())}")

    weight = wm.get_param(full_name)  # [out_features, in_features]
    n_cols = weight.shape[1]

    # Generate permutation
    if strategy == "random":
        perm = rng.permutation(n_cols)
    elif strategy == "l2_asc":
        col_norms = weight.norm(dim=0).numpy()
        perm = np.argsort(col_norms)
    elif strategy == "l2_desc":
        col_norms = weight.norm(dim=0).numpy()
        perm = np.argsort(col_norms)[::-1].copy()
    elif strategy == "reverse":
        perm = np.arange(n_cols)[::-1].copy()
    elif strategy == "interleave":
        even = np.arange(0, n_cols, 2)
        odd = np.arange(1, n_cols, 2)
        perm = np.concatenate([even, odd])
    elif strategy == "activation_magnitude" and calibration_activations is not None:
        # Sort by activation magnitude from calibration set
        act_norms = calibration_activations.get(full_name, weight.norm(dim=0).numpy())
        perm = np.argsort(act_norms)
    else:
        raise ValueError(f"Unknown permutation strategy: {strategy}")

    perm_tensor = torch.from_numpy(perm.astype(np.int64))
    modifications = {}

    # Permute columns of the target matrix
    new_weight = weight[:, perm_tensor]
    modifications[full_name] = new_weight

    if coupled:
        # Apply inverse permutation to downstream matrix rows
        inv_perm = np.argsort(perm)
        inv_perm_tensor = torch.from_numpy(inv_perm.astype(np.int64))

        downstream = _get_downstream_component(component)
        if downstream:
            ds_full_name = params.get(downstream)
            if ds_full_name:
                ds_weight = wm.get_param(ds_full_name)
                new_ds_weight = ds_weight[inv_perm_tensor, :]
                modifications[ds_full_name] = new_ds_weight

        # Handle Q/K coupling: if permuting q_proj, also permute k_proj
        coupled_same = _get_coupled_same_component(component)
        if coupled_same:
            cs_full_name = params.get(coupled_same)
            if cs_full_name:
                cs_weight = wm.get_param(cs_full_name)
                new_cs_weight = cs_weight[:, perm_tensor]
                modifications[cs_full_name] = new_cs_weight

    return modifications


def _get_downstream_component(component: str) -> Optional[str]:
    """Get the downstream component for coupled permutations."""
    mapping = {
        "self_attn.v_proj.weight": "self_attn.o_proj.weight",
        "mlp.up_proj.weight": "mlp.down_proj.weight",
        "mlp.gate_proj.weight": "mlp.down_proj.weight",
    }
    return mapping.get(component)


def _get_coupled_same_component(component: str) -> Optional[str]:
    """Get the component that must receive the same permutation (not inverse)."""
    mapping = {
        "self_attn.q_proj.weight": "self_attn.k_proj.weight",
        "self_attn.k_proj.weight": "self_attn.q_proj.weight",
        "mlp.up_proj.weight": "mlp.gate_proj.weight",
        "mlp.gate_proj.weight": "mlp.up_proj.weight",
    }
    return mapping.get(component)


def w2_transplant(
    wm: WeightManager,
    source_layer: int,
    target_layer: int,
    component: str,  # e.g., "self_attn.q_proj.weight"
) -> dict[str, torch.Tensor]:
    """W2: Cross-layer weight transplant.
    Copy a weight matrix from source_layer to target_layer."""
    src_params = wm.get_layer_param_names(source_layer)
    tgt_params = wm.get_layer_param_names(target_layer)

    src_name = src_params.get(component)
    tgt_name = tgt_params.get(component)

    if src_name is None or tgt_name is None:
        raise ValueError(f"Component {component} not found in source or target layer")

    source_weight = wm.get_param(src_name)
    return {tgt_name: source_weight}


def w3_reinitialize(
    wm: WeightManager,
    layer_idx: int,
    granularity: str,  # "attention", "mlp", "single_proj", "layernorm"
    distribution: str,  # "kaiming", "xavier", "zero", "scaled_noise"
    component: Optional[str] = None,  # for single_proj granularity
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """W3: Partial reinitialization."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    params = wm.get_layer_param_names(layer_idx)

    # Select target parameters based on granularity
    targets = {}
    if granularity == "attention":
        for name, full_name in params.items():
            if 'self_attn' in name and 'weight' in name:
                targets[full_name] = wm.get_param(full_name)
    elif granularity == "mlp":
        for name, full_name in params.items():
            if 'mlp' in name and 'weight' in name:
                targets[full_name] = wm.get_param(full_name)
    elif granularity == "single_proj":
        if component is None:
            raise ValueError("component required for single_proj granularity")
        full_name = params.get(component)
        if full_name is None:
            raise ValueError(f"Component {component} not found in layer {layer_idx}")
        targets[full_name] = wm.get_param(full_name)
    elif granularity == "layernorm":
        for name, full_name in params.items():
            if 'norm' in name.lower() or 'ln' in name.lower():
                targets[full_name] = wm.get_param(full_name)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    modifications = {}
    for full_name, original in targets.items():
        shape = original.shape
        fan_in = shape[-1] if len(shape) >= 2 else shape[0]
        fan_out = shape[0] if len(shape) >= 2 else shape[0]

        if distribution == "kaiming":
            std = math.sqrt(2.0 / fan_in)
            new_weight = torch.randn(shape, generator=rng) * std
        elif distribution == "xavier":
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            new_weight = torch.empty(shape).uniform_(-limit, limit)
        elif distribution == "zero":
            new_weight = torch.zeros(shape)
        elif distribution == "scaled_noise":
            sigma = original.float().std().item()
            new_weight = torch.randn(shape, generator=rng) * sigma
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        modifications[full_name] = new_weight.to(original.dtype)

    return modifications


def w4_head_surgery(
    wm: WeightManager,
    layer_idx: int,
    operation: str,  # "ablate", "duplicate", "average", "negate"
    head_i: int,
    head_j: Optional[int] = None,  # for duplicate/average
) -> dict[str, torch.Tensor]:
    """W4: Attention head surgery."""
    params = wm.get_layer_param_names(layer_idx)
    head_dim = wm.get_head_dim()
    n_heads = wm.get_n_heads()

    # Get o_proj weight
    o_proj_name = None
    for name, full_name in params.items():
        if 'o_proj' in name and 'weight' in name:
            o_proj_name = full_name
            break
    if o_proj_name is None:
        raise ValueError(f"o_proj not found in layer {layer_idx}")

    o_weight = wm.get_param(o_proj_name)  # [hidden_size, hidden_size]
    modifications = {}

    # o_proj input dimension is organized as [n_heads * head_dim]
    # head i occupies columns [i*head_dim : (i+1)*head_dim]
    start_i = head_i * head_dim
    end_i = (head_i + 1) * head_dim

    if operation == "ablate":
        # Zero out head i's columns in o_proj (input side)
        new_o = o_weight.clone()
        new_o[:, start_i:end_i] = 0.0
        modifications[o_proj_name] = new_o

    elif operation == "negate":
        new_o = o_weight.clone()
        new_o[:, start_i:end_i] *= -1.0
        modifications[o_proj_name] = new_o

    elif operation == "duplicate":
        if head_j is None:
            raise ValueError("head_j required for duplicate operation")
        # Copy head i's Q, K, V, O slices over head j
        start_j = head_j * head_dim
        end_j = (head_j + 1) * head_dim

        for name, full_name in params.items():
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']) and 'weight' in name:
                w = wm.get_param(full_name)
                new_w = w.clone()
                # For Q/K/V, rows correspond to heads: [head_dim * n_heads, hidden]
                new_w[start_j:end_j, :] = new_w[start_i:end_i, :]
                modifications[full_name] = new_w

        new_o = o_weight.clone()
        new_o[:, start_j:end_j] = new_o[:, start_i:end_i]
        modifications[o_proj_name] = new_o

    elif operation == "average":
        if head_j is None:
            raise ValueError("head_j required for average operation")
        start_j = head_j * head_dim
        end_j = (head_j + 1) * head_dim

        for name, full_name in params.items():
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']) and 'weight' in name:
                w = wm.get_param(full_name)
                new_w = w.clone()
                avg = (new_w[start_i:end_i, :] + new_w[start_j:end_j, :]) / 2.0
                new_w[start_i:end_i, :] = avg
                new_w[start_j:end_j, :] = avg
                modifications[full_name] = new_w

        new_o = o_weight.clone()
        avg_o = (new_o[:, start_i:end_i] + new_o[:, start_j:end_j]) / 2.0
        new_o[:, start_i:end_i] = avg_o
        new_o[:, start_j:end_j] = avg_o
        modifications[o_proj_name] = new_o

    return modifications


def w5_spectral_edit(
    wm: WeightManager,
    layer_idx: int,
    component: str,
    operation: str,  # "top_k", "bottom_k", "spectral_inversion", "uniform_spectrum"
    k: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """W5: Rank-selective spectral editing."""
    params = wm.get_layer_param_names(layer_idx)
    full_name = params.get(component)
    if full_name is None:
        raise ValueError(f"Component {component} not found in layer {layer_idx}")

    weight = wm.get_param(full_name).float()  # SVD needs float32
    rank = min(weight.shape)

    # Skip invalid k values
    if k is not None and (k <= 0 or k >= rank):
        raise ValueError(f"Invalid k={k} for matrix with rank={rank}")

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    if operation == "top_k":
        # Keep only top-k singular values
        new_S = S.clone()
        new_S[k:] = 0.0
    elif operation == "bottom_k":
        # Remove the k smallest singular values
        new_S = S.clone()
        new_S[-k:] = 0.0
    elif operation == "spectral_inversion":
        # Reverse the order of singular values
        new_S = S.flip(0)
    elif operation == "uniform_spectrum":
        # Replace all singular values with their mean
        new_S = torch.full_like(S, S.mean().item())
    else:
        raise ValueError(f"Unknown spectral operation: {operation}")

    # Reconstruct: W = U @ diag(S) @ Vh
    new_weight = U @ torch.diag(new_S) @ Vh
    return {full_name: new_weight.to(wm.get_param(full_name).dtype)}


# =============================================================================
# CLASS A: ACTIVATION-TIME INTERVENTIONS
# =============================================================================

class ActivationHook:
    """Manages forward hooks for activation-time interventions."""

    def __init__(self):
        self._handles = []

    def register(self, module: nn.Module, hook_fn: Callable):
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def remove_all(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def a2_head_scaling_hook(
    model: nn.Module,
    layer_idx: int,
    head_idx: int,
    scalar: float,
    head_dim: int,
) -> ActivationHook:
    """A2: Attention head output scaling.
    Scales head output after o_proj, before residual addition."""
    hook = ActivationHook()

    # Find the attention output projection module
    target_module = None
    for name, module in model.named_modules():
        if f'layers.{layer_idx}.' in name and 'o_proj' in name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"o_proj not found in layer {layer_idx}")

    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    def hook_fn(module, input, output):
        # output shape: [batch, seq_len, hidden_size]
        # The input to o_proj has the head dimension flattened
        # We scale the contribution of head_idx by modifying the output
        # Since o_proj is linear: output = input @ W^T + bias
        # We want to scale head_idx's contribution: scale columns start:end of W
        # But we have the full output. We need to decompose.
        # Simpler: modify the input to o_proj and recompute
        # Actually, for a linear layer output = x @ W^T, the contribution
        # of input columns [start:end] to the output is x[:,:,start:end] @ W[:,start:end]^T
        # We want to scale this by `scalar`.

        # Get the original contribution of this head
        inp = input[0]  # input is a tuple
        W = module.weight  # [hidden, hidden]
        head_contribution = inp[:, :, start:end] @ W[:, start:end].T

        # The modified output should be: output + (scalar - 1) * head_contribution
        return output + (scalar - 1.0) * head_contribution

    hook.register(target_module, hook_fn)
    return hook


def a1_residual_injection_hook(
    model: nn.Module,
    layer_idx: int,
    vector: torch.Tensor,  # [hidden_size] on the correct device
) -> ActivationHook:
    """A1: Residual stream injection.
    Add a fixed vector after layer_idx's output (before next layer)."""
    hook = ActivationHook()

    # Find the layer module and hook after it
    target_module = None
    for name, module in model.named_modules():
        if name.endswith(f'layers.{layer_idx}'):
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Layer {layer_idx} not found")

    def hook_fn(module, input, output):
        # output is typically a tuple; the first element is the hidden states
        if isinstance(output, tuple):
            hidden = output[0]
            modified = hidden + vector.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            return output + vector.unsqueeze(0).unsqueeze(0)

    hook.register(target_module, hook_fn)
    return hook


def a3_layer_skip_hook(
    model: nn.Module,
    layer_idx: int,
    variant: str = "full",  # "full", "attention_only", "mlp_only"
) -> ActivationHook:
    """A3: Layer skip (bypass).
    Skip layer N's computation — residual stream passes through unchanged."""
    hook = ActivationHook()

    if variant == "full":
        # Hook the entire layer to return input unchanged
        target = None
        for name, module in model.named_modules():
            if name.endswith(f'layers.{layer_idx}'):
                target = module
                break
        if target is None:
            raise ValueError(f"Layer {layer_idx} not found")

        def hook_fn(module, input, output):
            # Return the input hidden states unchanged
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            if isinstance(output, tuple):
                return (inp,) + output[1:]
            return inp

        hook.register(target, hook_fn)

    elif variant == "attention_only":
        # Skip only the attention sublayer
        target = None
        for name, module in model.named_modules():
            if f'layers.{layer_idx}.' in name and ('self_attn' in name or 'attention' in name):
                # Get the attention module (not its children)
                if name.count('.') <= 3:  # model.layers.N.self_attn
                    target = module
                    break
        if target is None:
            raise ValueError(f"Attention module not found in layer {layer_idx}")

        def hook_fn(module, input, output):
            # Return zeros so the residual add is a no-op
            if isinstance(output, tuple):
                hidden = output[0]
                zeros = torch.zeros_like(hidden)
                return (zeros,) + output[1:]
            return torch.zeros_like(output)

        hook.register(target, hook_fn)

    elif variant == "mlp_only":
        # Skip only the MLP sublayer
        target = None
        for name, module in model.named_modules():
            if f'layers.{layer_idx}.' in name and 'mlp' in name:
                if name.count('.') <= 3:  # model.layers.N.mlp
                    target = module
                    break
        if target is None:
            raise ValueError(f"MLP module not found in layer {layer_idx}")

        def hook_fn(module, input, output):
            return torch.zeros_like(output)

        hook.register(target, hook_fn)

    return hook
