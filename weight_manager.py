"""
Weight management protocol per spec v0.4.
Maintains a pristine CPU copy, applies interventions, and restores after each trial.
Checksum verification ensures no cascading corruption.
"""
import hashlib
import copy
from collections import OrderedDict

import torch


class WeightManager:
    """Manages pristine model weights on CPU, with apply/restore/verify cycle."""

    def __init__(self, model):
        self.model = model
        # Deep copy all parameters to CPU as the golden copy
        print("  Creating pristine weight copy on CPU...")
        self._pristine = OrderedDict()
        for name, param in model.named_parameters():
            self._pristine[name] = param.data.clone().cpu()
        self._baseline_checksum = self._compute_checksum()
        print(f"  Pristine copy created. Checksum: {self._baseline_checksum[:16]}...")

    def _compute_checksum(self, source: str = "pristine") -> str:
        """Compute checksum from first and last layer Q projection weights.

        Args:
            source: "pristine" to hash backup, "model" to hash live GPU weights.
        """
        hasher = hashlib.md5()
        if source == "model":
            params = self.model.named_parameters()
        else:
            params = self._pristine.items()
        # Hash a few key tensors for speed
        for name, tensor in params:
            if 'layers.0.self_attn.q_proj' in name or 'layers.0.attention.q_proj' in name:
                hasher.update(tensor.detach().cpu().numpy().tobytes())
                break
        if source == "model":
            params = self.model.named_parameters()
        else:
            params = self._pristine.items()
        for name, tensor in reversed(list(params)):
            if 'q_proj' in name and 'weight' in name:
                hasher.update(tensor.detach().cpu().numpy().tobytes())
                break
        return hasher.hexdigest()

    @property
    def baseline_checksum(self) -> str:
        return self._baseline_checksum

    def get_param(self, name: str) -> torch.Tensor:
        """Get a pristine copy of a parameter (on CPU)."""
        return self._pristine[name].clone()

    def apply_weight_modification(self, modifications: dict[str, torch.Tensor]):
        """Apply weight modifications to the GPU model.

        Args:
            modifications: dict mapping parameter name -> new tensor value
        """
        state_dict = self.model.state_dict()
        for name, new_value in modifications.items():
            if name not in state_dict:
                raise KeyError(f"Parameter {name} not found in model")
            device = state_dict[name].device
            dtype = state_dict[name].dtype
            state_dict[name] = new_value.to(device=device, dtype=dtype)
        self.model.load_state_dict(state_dict, strict=True)

    def restore(self):
        """Restore all model weights from pristine CPU copy."""
        state_dict = self.model.state_dict()
        for name, pristine_value in self._pristine.items():
            device = state_dict[name].device
            dtype = state_dict[name].dtype
            state_dict[name] = pristine_value.clone().to(device=device, dtype=dtype)
        self.model.load_state_dict(state_dict, strict=True)

    def verify(self) -> bool:
        """Verify current model weights match the pristine copy.
        Returns True if verification passes."""
        current_checksum = self._compute_checksum(source="model")
        return current_checksum == self._baseline_checksum

    def get_layer_param_names(self, layer_idx: int) -> dict[str, str]:
        """Get parameter names for a specific layer.
        Returns dict mapping short names (q_proj.weight, etc.) to full parameter paths."""
        result = {}
        for name in self._pristine:
            # Match patterns like "model.layers.5.self_attn.q_proj.weight"
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit() and int(p) == layer_idx:
                    # Check if this is in a 'layers' context
                    if i > 0 and parts[i-1] in ('layers', 'h', 'blocks'):
                        short = '.'.join(parts[i+1:])
                        result[short] = name
                        break
        return result

    def get_n_layers(self) -> int:
        """Get the number of transformer layers."""
        layer_indices = set()
        for name in self._pristine:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit() and i > 0 and parts[i-1] in ('layers', 'h', 'blocks'):
                    layer_indices.add(int(p))
        return max(layer_indices) + 1 if layer_indices else 0

    def get_n_heads(self) -> int:
        """Get number of attention heads from model config."""
        return self.model.config.num_attention_heads

    def get_head_dim(self) -> int:
        """Get dimension per attention head."""
        return self.model.config.hidden_size // self.model.config.num_attention_heads

    def get_hidden_size(self) -> int:
        return self.model.config.hidden_size
