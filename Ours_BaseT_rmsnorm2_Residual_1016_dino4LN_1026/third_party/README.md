# Third-party source notices

This source snapshot retains existing upstream copyright and license notices.
The following licenses apply to their respective upstream material; they do not
relicense the whole research project or the separately transferred weights.

- DINOv3: source and license remain in `../SSL_code/dinov3/`.
- iREPA: the JiT integration derives from the local iREPA source tree. Its MIT
  license is retained in `irepa/LICENSE`; local training, routing, and evaluation
  adaptations are documented in this project's README and source files.
- RAEv2: the existing representation/evaluation integrations refer to RAEv2.
  Its repository license is retained in `raev2/LICENSE` as an upstream notice.
  The vendored evaluator retains its own source references under
  `../jit/evaluation/vendor/fd_evaluator/`.
- PixelDiT2: `pixeldit2/LICENSE` and `pixeldit2/NOTICE` accompany the adapted
  timestep projector in `../pixeldit2_projector.py`.

Encoder and evaluation weights are not redistributed by this Git repository.
