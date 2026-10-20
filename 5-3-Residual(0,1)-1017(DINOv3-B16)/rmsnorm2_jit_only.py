"""The pinned rmsnorm2 baseline with generation weights DINO=0 and JiT=1.

The original model is constructed first, preserving its shared initial weights
and RNG position. Only then are the unused generation modules unregistered,
before DDP, AdamW, EMA, or device placement can include them. Training loss,
teacher preprocessing, alignment, CFG, and samplers remain the reference code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rmsnorm2_reference_loader import load_reference

reference = load_reference()
from dino_jit_denoiser import DINOJiTDenoiser  # noqa: E402


class RMSNorm2JitOnlyDenoiser(DINOJiTDenoiser):
    """JiT-only input, optionally removing its extra pre-APE RMSNorm."""

    def __init__(self, args):
        if getattr(args, "dino_generation_path_enabled", False) is not False:
            raise ValueError("This ablation requires dino_generation_path_enabled=false.")
        if getattr(args, "dino_jit_input_mode", None) != "time_blend":
            raise ValueError("JiT-only rmsnorm2 must retain the baseline time_blend mode.")
        if not getattr(args, "dino_jit_pre_blend_rmsnorm", False):
            raise ValueError("JiT-only rmsnorm2 requires the baseline pre-blend RMSNorm.")
        if getattr(args, "dino_input_projector", None) != "pixeldit2":
            raise ValueError("Construct the baseline pixeldit2 projector before removing it.")
        if getattr(args, "use_ddt_head", True):
            raise ValueError("This ablation uses the baseline native JiT output head.")

        # Do not omit any construction or use fork_rng here: the alignment
        # projector is initialized after the DINO student and must receive the
        # same random values as the original experiment.
        super().__init__(args)
        self.dino_generation_path_enabled = False
        self._capture_skipped_generation_rng()

        # dino_student_projector is NOT a generation module: it projects JiT
        # block-4 features into the frozen teacher's alignment target space.
        removed = (self.dino_student, self.dino_input_projector, self.dino_pre_blend_norm)
        self.removed_generation_trainable_parameters = sum(
            parameter.numel()
            for module in removed
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.dino_student = None
        self.dino_input_projector = None
        self.dino_pre_blend_norm = nn.Identity()
        # Construct first to preserve all shared initialization/RNG. Removing
        # the extra input norm must not affect block/QK/final-layer norms.
        self.jit_input_rmsnorm = bool(getattr(args, "jit_input_rmsnorm", True))
        if not self.jit_input_rmsnorm:
            self.jit_pre_blend_norm = nn.Identity()

    def _capture_skipped_generation_rng(self) -> None:
        """Describe the pinned MLS student's only stochastic forward operations.

        DINOv3-B resamples RoPE coordinates for every transformer block, even
        when all attention/LoRA/drop-path probabilities are zero. Keep these
        individual uniform calls so dropping the branch does not change the
        next training batch's label-drop, timestep, or image-noise RNG stream.
        """
        student = self.dino_student
        if student is None or self.dino_input_projector is None:
            raise ValueError("The full baseline generation modules must be constructed first.")
        if (
            student.aggregation != "mls"
            or student._layer_indices != [2, 5, 8, 11]
            or len(student.model.blocks) != 12
            or student._total_blocks != 12
        ):
            raise ValueError("RNG preservation supports the baseline 12-block MLS [2,5,8,11].")
        for module in (student, self.dino_input_projector):
            for child in module.modules():
                if isinstance(child, nn.modules.dropout._DropoutNd) and child.p != 0:
                    raise ValueError("Skipping generation with nonzero dropout is unsupported.")
                if float(getattr(child, "sample_drop_ratio", 0.0)) != 0.0:
                    raise ValueError("Skipping generation with stochastic depth is unsupported.")

        rope = getattr(student.model, "rope_embed", None)
        if rope is None:
            raise ValueError("The baseline DINOv3-B RoPE module is required.")
        draws = []
        if rope.shift_coords is not None:
            bound = float(rope.shift_coords)
            if not np.isfinite(bound) or bound < 0:
                raise ValueError("Unsupported RoPE coordinate shift.")
            draws.append((2, -bound, bound))
        for name, count in (("jitter_coords", 2), ("rescale_coords", 1)):
            value = getattr(rope, name)
            if value is not None:
                if not np.isfinite(value) or value < 1:
                    raise ValueError(f"Unsupported RoPE {name}.")
                bound = float(np.log(value))
                draws.append((count, -bound, bound))
        self._generation_rng_draws = tuple(draws)
        self._generation_rng_dtype = rope.dtype
        self._generation_rng_blocks = len(student.model.blocks)

    def _consume_skipped_generation_rng(self, device: torch.device) -> None:
        if self.training:
            for _ in range(self._generation_rng_blocks):
                for count, low, high in self._generation_rng_draws:
                    # One call per original RoPE operation is essential on CUDA:
                    # a single vectorized draw advances Philox differently.
                    torch.empty(
                        count, device=device, dtype=self._generation_rng_dtype,
                    ).uniform_(low, high)

    def _encode_noisy_student(self, noisy_z, timesteps=None):
        """Used by both the reference training and sampling entry points."""
        self._consume_skipped_generation_rng(noisy_z.device)
        return None

    def _compose_jit_input_tokens(
        self, noisy_image, dino_tokens, timesteps, label_drop_mask=None,
    ):
        if dino_tokens is not None:
            raise ValueError("The disabled generation branch must not produce DINO tokens.")
        if timesteps.numel() != noisy_image.shape[0]:
            raise ValueError("Expected one timestep per image.")
        native_tokens = self.jit_pre_blend_norm(self.net.x_embedder(noisy_image))
        zeros = torch.zeros_like(native_tokens)
        # Keep the baseline diagnostics schema without creating unused
        # trainable parameters or running DINO/PgT. All samples have DINO=0.
        return native_tokens, {
            "native_tokens": native_tokens,
            "dino_projected_tokens": zeros,
            "jit_weighted_tokens": native_tokens,
            "dino_weighted_tokens": zeros,
            "grounding_drop_mask": torch.ones(
                noisy_image.shape[0], device=noisy_image.device, dtype=torch.bool,
            ),
        }
