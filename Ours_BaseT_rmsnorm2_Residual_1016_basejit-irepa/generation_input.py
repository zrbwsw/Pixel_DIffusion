"""Optional generation input on top of the unmodified iREPA JiT/Denoiser.

``baseline`` constructs only the original Denoiser. It does not import or create
the generation DINO, attach hooks, or add parameters. The independent frozen
alignment encoder is still constructed and used by iREPA's training entrypoint.

``rmsnorm2`` replaces only the output of JiT's native patch embedder, before the
original forward adds its positional embedding. All subsequent JiT operations,
alignment projection/loss, and sampling remain in the original implementation.
"""

from __future__ import annotations

import torch
from torch import nn


def _build_generation_encoder(args):
    """Lazy factory for the noisy-image encoder; never used by baseline mode."""
    from dino_v3_lora import DINOv3PatchTokenEncoder

    # This is the generation branch, not the frozen iREPA alignment encoder.
    # train_norm=False freezes the pretrained final LN gamma/beta; it does NOT
    # discard them. The wrapper retains that LN when extracting MLS features.
    return DINOv3PatchTokenEncoder(
        model_name="dinov3_vitb16",
        weights_path=getattr(args, "dino_weights", None) or None,
        use_lora=True,
        lora_r=int(getattr(args, "dino_lora_r", 64)),
        lora_alpha=float(getattr(args, "dino_lora_alpha", 64.0)),
        lora_dropout=float(getattr(args, "dino_lora_dropout", 0.0)),
        lora_target_modules="qkv,proj,fc1,fc2",
        train_patch_embed=bool(getattr(args, "dino_train_patch_embed", True)),
        train_cls_token=False,
        train_norm=False,
        aggregation="mls",
        layer_indices=[2, 5, 8, 11],
    )


class RMSNormGenerationInput(nn.Module):
    """Learned generation input; regularization remains owned by iREPA.

    For JiT-B/16 at 256px, both branches have shape [B, 256, 768]. The two
    RMSNorms have independent learned gains, initialized to one, and normalize
    over each token's channel dimension. They use no dataset statistics.
    """

    def __init__(self, args, *, hidden_size: int, num_heads: int, patch_size: int):
        super().__init__()
        from pixeldit2_projector import TimestepProjection
        from util.model_util import RMSNorm

        self.patch_size = int(patch_size)
        self.dino = _build_generation_encoder(args)
        if int(self.dino.patch_size) != self.patch_size:
            raise ValueError("Generation DINO and JiT must use the same patch size.")
        self.projector = TimestepProjection(
            in_dim=int(self.dino.embed_dim),
            out_dim=int(hidden_size),
            hidden_size=int(hidden_size),
            num_blocks=1,
            num_heads=int(num_heads),
            mlp_ratio=4.0,
        )
        self.patch_norm = RMSNorm(hidden_size, eps=1e-6)
        self.dino_norm = RMSNorm(hidden_size, eps=1e-6)
        self._pending_timestep = None

    def forward(self, noisy_image, timestep, patch_tokens):
        """Blend native tokens with DINO+PgT tokens using clean-image time t."""
        # Feed raw z=t*x+(1-t)*noise directly, with no image normalization.
        dino_tokens = self.dino(noisy_image)
        grid_hw = tuple(size // self.patch_size for size in noisy_image.shape[-2:])
        projected_tokens = self.projector(
            dino_tokens.to(dtype=self.projector.input_proj.weight.dtype),
            timestep.flatten(),
            grid_hw,
        )
        if patch_tokens.shape != projected_tokens.shape:
            raise ValueError(
                "Native JiT and projected DINO token shapes must match, got "
                f"{tuple(patch_tokens.shape)} and {tuple(projected_tokens.shape)}."
            )
        patch_tokens = self.patch_norm(patch_tokens)
        projected_tokens = self.dino_norm(projected_tokens)
        t = timestep.flatten().to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        if t.numel() != patch_tokens.shape[0]:
            raise ValueError("Expected one blend timestep per sample.")
        t = t.view(-1, 1, 1)
        return (1.0 - t) * patch_tokens + t * projected_tokens

    def _capture_timestep(self, module, inputs, kwargs):
        # Original JiT.forward(x, t, y) may be called positionally or by keyword.
        self._pending_timestep = None
        timestep = inputs[1] if len(inputs) > 1 else kwargs.get("t")
        if timestep is None:
            raise RuntimeError("Generation input requires JiT.forward's timestep t.")
        self._pending_timestep = timestep

    def _replace_patch_tokens(self, module, inputs, kwargs, output):
        # Consume and clear the reference before invoking generation modules.
        # This prevents a direct x_embedder call from reusing an earlier t.
        timestep = self._pending_timestep
        self._pending_timestep = None
        if timestep is None:
            raise RuntimeError("Call JiT.forward to use the generation input adapter.")
        noisy_image = inputs[0] if inputs else kwargs.get("x")
        if noisy_image is None:
            raise RuntimeError("JiT patch embedder did not receive its image input.")
        return self(noisy_image, timestep, output)

    def _clear_timestep(self, module, inputs, output):
        # always_call also clears t when original JiT raises before patchifying.
        self._pending_timestep = None

    def install(self, net):
        """Register parameters under net and intercept only its patch tokens."""
        if hasattr(net, "generation_input"):
            raise ValueError("JiT already has a generation input adapter.")
        net.generation_input = self
        net.register_forward_pre_hook(self._capture_timestep, with_kwargs=True)
        net.x_embedder.register_forward_hook(self._replace_patch_tokens, with_kwargs=True)
        net.register_forward_hook(self._clear_timestep, always_call=True)


def build_denoiser(args):
    """Build the exact baseline, or attach the optional rmsnorm2 generation path.

    Required original Denoiser arguments are unchanged. Additional arguments:
    generation_input_mode (baseline/rmsnorm2), dino_weights, dino_lora_r,
    dino_lora_alpha, dino_lora_dropout, and dino_train_patch_embed. The caller
    must expose the bundled ``jit`` directory on sys.path, as its original
    entrypoint does, so its unmodified absolute imports resolve locally.
    """
    mode = str(getattr(args, "generation_input_mode", "baseline")).lower()
    if mode not in {"baseline", "rmsnorm2"}:
        raise ValueError(f"Unknown generation_input_mode={mode!r}.")

    from denoiser import Denoiser

    denoiser = Denoiser(args)
    if mode == "baseline":
        return denoiser

    adapter = RMSNormGenerationInput(
        args,
        hidden_size=denoiser.net.hidden_size,
        num_heads=denoiser.net.num_heads,
        patch_size=denoiser.net.patch_size,
    )
    adapter.install(denoiser.net)
    return denoiser
