"""复用原 REPA cosine loss，只补充 FP32、teacher 停止梯度和相似度统计。"""

import torch
from jit.projection_loss import make_projection_loss


_repa_cosine = make_projection_loss("cosine")


def patch_cosine_loss(
    student_tokens: torch.Tensor, teacher_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 原 loss 接收 teacher 在前、student 在后；内部沿通道 L2 归一化并逐 patch 取均值。
    loss = _repa_cosine(teacher_tokens.detach().float(), student_tokens.float())
    return loss, -loss  # loss=-mean(cos)，复用该值记录相似度，不重复计算。
