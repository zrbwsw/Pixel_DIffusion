# Third-party source notices

This source snapshot retains existing upstream copyright and license notices.
The following licenses apply to their respective upstream material; they do not
relicense the whole research project or the separately transferred weights.

- DINOv1: `../SSL_code/dino/` contains unmodified official hubconf, ViT and
  utility source from facebookresearch/dino commit
  `7c446df5b9f45747937fb0d72314eb9f7b66930a`, with its Apache 2.0 LICENSE.
- DINOv3: source and licenses remain in `../SSL_code/dinov3/`.
- iREPA: the 21 original JiT Python files are copied byte-for-byte from
  `99ad4ac234efe8de52ce157120f72856e836d09f`; SHA256 values are pinned in
  `../upstream_manifest.json`. Its MIT license is retained in `irepa/LICENSE`.
- RAEv2: the existing representation/evaluation integrations refer to RAEv2.
  Its repository license is retained in `raev2/LICENSE` as an upstream notice.
- PixelDiT2: `pixeldit2/LICENSE` and `pixeldit2/NOTICE` accompany the adapted
  timestep projector in `../pixeldit2_projector.py`.

Encoder and evaluation weights are not redistributed by this Git repository.
