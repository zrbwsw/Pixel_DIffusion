"""PixelDiT v1 的 rectified-flow 训练、REPA 对齐和采样。

时间约定为 x_t=t*x+(1-t)*noise：t=0 是噪声，t=1 是干净图像。
网络学习 velocity=x-noise；REPA 另外约束中间层接近 frozen DINOv2
的干净图像语义。采样从 t=0 积分到 t=1，CFG 同时计算条件和无条件两路。
"""
# Modified from https://github.com/MCG-NJU/PixNerd

import copy
import logging
import os
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .utils import no_grad

logger = logging.getLogger(__name__)


class BaseScheduler:
    """定义 alpha(t)、sigma(t) 及导数，统一训练和采样的时间参数化。"""

    def alpha(self, t) -> Tensor:
        raise NotImplementedError

    def sigma(self, t) -> Tensor:
        raise NotImplementedError

    def dalpha(self, t) -> Tensor:
        raise NotImplementedError

    def dsigma(self, t) -> Tensor:
        raise NotImplementedError

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)

    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t) * self.sigma(t)

    def drift_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dalpha / (alpha + 1e-6)

    def diffuse_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dsigma * sigma - dalpha / (alpha + 1e-6) * sigma ** 2

    def w(self, t):
        return self.sigma(t)


class LinearScheduler(BaseScheduler):
    """线性插值 schedule：alpha=t、sigma=1-t，返回可广播到图像的形状。"""

    def alpha(self, t) -> Tensor:
        return t.view(-1, 1, 1, 1)  # [B] → [B,1,1,1]，干净图像权重

    def sigma(self, t) -> Tensor:
        return (1 - t).view(-1, 1, 1, 1)  # [B,1,1,1]，噪声权重

    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)  # d(alpha)/dt=1

    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)  # d(sigma)/dt=-1


class BaseTrainer(nn.Module):
    """训练器基类：在计算损失前执行 classifier-free label dropout。"""

    def __init__(self, null_condition_p=0.1):
        super().__init__()
        self.null_condition_p = null_condition_p  # 每个样本独立替换为 null 类的概率，默认0.1

    def preproprocess(self, x, condition, uncondition, metadata):
        """按概率把 condition 替换为 uncondition，保持标签形状 [B]。"""
        bsz = x.shape[0]
        if self.null_condition_p > 0:
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p  # [B]，True 表示本样本使用无条件标签
            mask = mask.view(-1, *([1] * (len(condition.shape) - 1))).to(condition.dtype)  # 广播到条件形状
            condition = condition * (1 - mask) + uncondition * mask  # 同时训练条件/无条件生成能力
        return x, condition, metadata

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raise NotImplementedError

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(self, net, ema_net, solver, x, condition, uncondition, metadata=None):
        """先 drop 标签，再调用具体 REPA/flow 训练损失，统一使用 BF16 autocast。"""
        x, condition, metadata = self.preproprocess(x, condition, uncondition, metadata)
        return self._impl_trainstep(net, ema_net, solver, x, condition, metadata)


class BaseSampler(nn.Module):
    """采样器基类：返回最终图像，也可按调用方要求返回完整轨迹。"""

    def __init__(
        self,
        scheduler: BaseScheduler = None,
        guidance_fn: Callable = None,
        num_steps: int = 250,
        guidance: Union[float, List[float]] = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.scheduler = scheduler

    def _impl_sampling(self, net, noise, condition, uncondition):
        raise NotImplementedError

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, net, noise, condition, uncondition, return_x_trajs=False, return_v_trajs=False):
        """noise [B,C,H,W]；condition/uncondition [B]，默认返回 x_trajs[-1]。"""
        x_trajs, v_trajs = self._impl_sampling(net, noise, condition, uncondition)
        if return_x_trajs and return_v_trajs:
            return x_trajs[-1], x_trajs, v_trajs
        elif return_x_trajs:
            return x_trajs[-1], x_trajs
        elif return_v_trajs:
            return x_trajs[-1], v_trajs
        else:
            return x_trajs[-1]


def simple_guidance_fn(out, cfg):
    """合并前后两个 batch：前 B 是无条件，后 B 是条件，输出仍为 [B,...]。"""
    uncondition, condition = out.chunk(2, dim=0)  # 各 [B,...]，与采样时的拼接顺序一致
    out = uncondition + cfg * (condition - uncondition)  # CFG：在无条件结果上沿条件方向外推
    return out


def shift_respace_fn(t, shift=3.0):
    """重排采样时间：t/(t+(1-t)*shift)，保持 t=0 和 t=1 端点不变。"""
    return t / (t + (1 - t) * shift)


def ode_step_fn(x, v, dt, s, w):
    """确定性 ODE 更新：沿预测 velocity 走 dt，s/w 在此模式下不使用。"""
    return x + v * dt


def sde_mean_step_fn(x, v, dt, s, w):
    """SDE 均值漂移项，在 ODE 更新上增加 s*w*dt。"""
    return x + v * dt + s * w * dt


def sde_step_fn(x, v, dt, s, w):
    """SDE 更新：均值漂移外加 sqrt(2*w*dt) 的随机噪声。"""
    return x + v * dt + s * w * dt + torch.sqrt(2 * w * dt) * torch.randn_like(x)


def sde_preserve_step_fn(x, v, dt, s, w):
    """方差保持 SDE 的更新形式，噪声系数为 sqrt(w*dt)。"""
    return x + v * dt + 0.5 * s * w * dt + torch.sqrt(w * dt) * torch.randn_like(x)


class EulerSampler(BaseSampler):
    """一阶 Euler 采样：每步只用当前时间的 velocity 更新 x，并支持 CFG 区间。"""

    def __init__(
        self,
        w_scheduler: BaseScheduler = None,
        timeshift=1.0,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        step_fn: Callable = ode_step_fn,
        last_step=None,
        last_step_fn: Callable = ode_step_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn]
        if self.w_scheduler is not None and self.step_fn == ode_step_fn:
            logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise]
        v_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t, cfg_condition)
            if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out
            s = ((1 / dalpha_over_alpha) * v - x) / (sigma ** 2 - (1 / dalpha_over_alpha) * dsigma_mul_sigma)
            if i < self.num_steps - 1:
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs


class HeunSampler(BaseSampler):
    """二阶 Heun 采样：先 Euler 预测，再用起点/终点 velocity 的均值校正。"""

    def __init__(
        self,
        scheduler: BaseScheduler = None,
        w_scheduler: BaseScheduler = None,
        exact_henu=False,
        timeshift=1.0,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        step_fn: Callable = ode_step_fn,
        last_step=None,
        last_step_fn: Callable = ode_step_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.exact_henu = exact_henu
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max
        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn]
        if self.w_scheduler is not None and self.step_fn == ode_step_fn:
            logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        v_hat, s_hat = 0.0, 0.0
        x_trajs = [noise]
        v_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            alpha_over_dalpha = 1 / self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            t_hat = t_next.repeat(batch_size)
            sigma_hat = self.scheduler.sigma(t_hat)
            alpha_over_dalpha_hat = 1 / self.scheduler.dalpha_over_alpha(t_hat)
            dsigma_mul_sigma_hat = self.scheduler.dsigma_mul_sigma(t_hat)

            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0
            if i == 0 or self.exact_henu:
                cfg_x = torch.cat([x, x], dim=0)
                cfg_t_cur = t_cur.repeat(2)
                out = net(cfg_x, cfg_t_cur, cfg_condition)
                if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                    guidance = self.guidance
                    out = self.guidance_fn(out, guidance)
                else:
                    out = self.guidance_fn(out, 1.0)
                v = out
                s = (alpha_over_dalpha * v - x) / (sigma ** 2 - (alpha_over_dalpha) * dsigma_mul_sigma)
            else:
                v = v_hat
                s = s_hat
            x_hat = self.step_fn(x, v, dt, s=s, w=w)
            if i < self.num_steps - 1:
                cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
                cfg_t_hat = t_hat.repeat(2)
                out = net(cfg_x_hat, cfg_t_hat, cfg_condition)
                if self.guidance_interval_min < t_hat[0] < self.guidance_interval_max:
                    guidance = self.guidance
                    out = self.guidance_fn(out, guidance)
                else:
                    out = self.guidance_fn(out, 1.0)
                v_hat = out
                s_hat = (
                    (alpha_over_dalpha_hat) * v_hat - x_hat
                ) / (sigma_hat ** 2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat)
                v = (v + v_hat) / 2
                s = (s + s_hat) / 2
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs


class FlowDPMSolverSampler(BaseSampler):
    """FlowDPMSolver：首步 Euler，后续用当前/上一 velocity 做 AB2 更新。

    每个采样步把无条件和条件样本拼成 2B 一次前向；只有指定时间区间
    内应用 CFG，区间外将 guidance 置为1，避免无意义的强度放大。
    """

    def __init__(
        self,
        w_scheduler: BaseScheduler = None,
        timeshift=1.0,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        step_fn: Callable = ode_step_fn,
        last_step=None,
        last_step_fn: Callable = ode_step_fn,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)  # 从噪声端 t=0 开始
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)  # 补上干净图像端点 t=1
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)  # 按配置重排时间间隔，端点仍固定

    def _impl_sampling(self, net, noise, condition, uncondition):
        """从 noise [B,C,H,W] 积分到图像，并记录 x/velocity 轨迹。"""
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)  # [2B]：前 B 无条件，后 B 条件

        x = noise  # 初始 x_0 为标准噪声 [B,C,H,W]
        x_trajs = [noise]
        v_trajs = []
        v_prev = None
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur  # 当前积分步长
            t_cur = t_cur.repeat(batch_size)  # 整 batch 共用同一采样时间 [B]

            cfg_x = torch.cat([x, x], dim=0)  # [2B,C,H,W]，条件/无条件共享同一份 noisy state
            cfg_t = t_cur.repeat(2)  # [2B]
            out = net(cfg_x, cfg_t, cfg_condition)  # 一次前向得到两路 velocity：[2B,C,H,W]
            if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)  # 区间内执行 uncond+cfg*(cond-uncond)：[B,C,H,W]
            else:
                out = self.guidance_fn(out, 1.0)  # cfg=1 时退化为条件输出，不再放大
            v = out

            if v_prev is None:
                x = x + v * dt  # 第一步 Euler：x_{n+1}=x_n+dt*v_n
            else:
                x = x + dt * (1.5 * v - 0.5 * v_prev)  # 后续 AB2：利用当前和上一步 velocity
            v_prev = v
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))  # 终点不再额外求 velocity，用0补齐轨迹长度
        return x_trajs, v_trajs


def constant(alpha, sigma):
    """默认 loss_weight_fn：所有时间的 flow matching 权重都为1。"""
    return 1


def time_shift_fn(t, timeshift=1.0):
    """训练时间的 shift：t/(t+(1-t)*timeshift)，保持端点不变。"""
    return t / (t + (1 - t) * timeshift)


class DINOv2(nn.Module):
    """REPA 的 frozen teacher，从干净原图提取语义 patch token。

    它与生成侧的 DINOv3+LoRA 是两条独立路径：本模块参数不参与训练，
    输入是 metadata['raw_image']，不是扩散加噪后的 x_t。
    """

    def __init__(self, model_name: str = "dinov2_vitb14", base_patch_size=16):
        """加载 frozen teacher；优先使用本地源码，避免缓存命中仍访问 GitHub。"""
        super().__init__()
        local_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if os.path.isdir(local_repo):
            self.encoder = torch.hub.load(local_repo, model_name, source="local")  # 本地 hubconf+缓存权重，不访问网络
        else:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2:main", model_name, trust_repo=True,
            )  # 没有准备本地资源时，保留上游自动下载方式
        self.encoder = self.encoder.to(torch.bfloat16)  # teacher 推理使用 BF16，减少显存和计算
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()  # 去掉分类头，只保留视觉特征提取器
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x, resize=True):
        """将 [0,1] 干净原图做 ImageNet 标准化和缩放，输出 [B,N,768]。

        256×256 输入默认缩到224×224，对应 DINOv2-B/14 的16×16 patch 网格。
        返回的是 final LN 后的 patch token，不包含分类结果。
        """
        b, c, h, w = x.shape  # x [B,3,H,W]，干净 RGB 原图而非 noisy x_t
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)  # 按 DINO 预训练分布归一化，形状不变
        if resize:
            x = torch.nn.functional.interpolate(
                x,
                (int(14 * h / self.base_patch_size), int(14 * w / self.base_patch_size)),
                mode="bicubic",
            )  # 匹配 DINOv2 patch14：[B,3,14H/base_patch_size,14W/base_patch_size]
        feature = self.encoder.forward_features(x)["x_norm_patchtokens"]  # frozen teacher patch 特征 [B,N,768]
        feature = feature.to(torch.bfloat16)  # 与训练混合精度对齐，形状不变
        return feature  # [B,N,768]


class REPATrainer(BaseTrainer):
    """联合训练 flow matching 和 REPA 表征对齐。

    总损失 = velocity MSE + feat_loss_weight * (1-cosine)。
    生成网络预测 velocity；第 align_layer 个 patch block 的中间特征经可训练
    MLP 投影后，与 frozen DINOv2 的干净图像 token 做逐位置 cosine 对齐。
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        loss_weight_fn: Callable = constant,
        feat_loss_weight: float = 0.5,
        lognorm_t: bool = False,
        timeshift: float = 1.0,
        encoder: Optional[nn.Module] = None,
        align_layer: int = 8,
        proj_denoiser_dim: int = 256,
        proj_hidden_dim: int = 256,
        proj_encoder_dim: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t  # True 时从 logit-normal 采样干净图像权重 t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight  # 默认0.5，控制 REPA 损失相对扩散 MSE 的强度
        self.align_layer = align_layer  # 从1计数；默认读取 patch_blocks[7] 的输出
        self.encoder = encoder  # frozen DINOv2 teacher，只提供目标，不反传梯度
        no_grad(self.encoder)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )  # 可训练投影：denoiser hidden → teacher hidden → teacher feature_dim

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        """构造 noisy x_t 和 velocity 目标，并同时计算生成/对齐损失。

        x [B,3,H,W] 是 [-1,1] 干净图像；metadata['raw_image'] 是 [0,1]
        原图，供 frozen teacher 使用；y [B] 是已经 drop 的条件标签。
        """
        raw_images = metadata["raw_image"]  # [B,3,H,W]，干净原图 [0,1]
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=torch.float32).sigmoid()  # logit-normal → (0,1) 的 [B]
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)  # 均匀采样 [B]
        t = time_shift_fn(base_t, self.timeshift).to(x.dtype)  # [B]，训练用原始时间，不是 f(t)
        noise = torch.randn_like(x)  # 与 x 同形的标准噪声 [B,3,H,W]
        alpha = self.scheduler.alpha(t)  # [B,1,1,1]，干净图权重
        dalpha = self.scheduler.dalpha(t)  # alpha 对 t 的导数
        sigma = self.scheduler.sigma(t)  # [B,1,1,1]，噪声权重
        dsigma = self.scheduler.dsigma(t)  # sigma 对 t 的导数

        x_t = alpha * x + noise * sigma  # 加噪插值：[B,3,H,W]
        v_t = dalpha * x + dsigma * noise  # 线性 schedule 下 velocity=x-noise：[B,3,H,W]

        src_feature = []

        def forward_hook(net, input, output):
            feature = output
            if isinstance(feature, tuple):
                feature = feature[0]
            src_feature.append(feature)  # 保存对齐层输出 [B,N,C_denoiser]

        handle = net.patch_blocks[self.align_layer - 1].register_forward_hook(forward_hook)  # 在对齐 block 后临时取中间特征

        out = net(x_t, t, y)  # 生成模型预测 velocity：[B,3,H,W]；同时触发上面的 hook
        src_feature = self.proj(src_feature[0])  # 中间特征投影到 teacher 维度：[B,N,768]
        handle.remove()  # 立即移除 hook，避免后续前向重复收集或引用残留

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)  # frozen DINOv2 目标 [B,N_teacher,768]，不建计算图

        if dst_feature.shape[1] != src_feature.shape[1]:
            # DINOv2 patch14 与 PixelDiT patch16 的 token 数不同，按二维网格对齐空间位置。
            bsz, ls, ch = src_feature.shape
            ld = dst_feature.shape[1]
            hs = int(ls ** 0.5)
            hd = int(ld ** 0.5)
            src_spatial = src_feature.view(bsz, hs, hs, ch).permute(0, 3, 1, 2)  # [B,N,C] → [B,C,hs,hs]
            if ld < ls:
                resized = torch.nn.functional.adaptive_avg_pool2d(src_spatial, (hd, hd))  # teacher 网格更小时平均池化
            else:
                resized = torch.nn.functional.interpolate(src_spatial, size=(hd, hd), mode="bilinear", align_corners=False)  # 否则双线性放大
            src_feature = resized.permute(0, 2, 3, 1).reshape(bsz, ld, ch)  # 恢复 [B,N_teacher,C]

        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)  # 每个空间位置一个 cosine：[B,N_teacher]
        cos_loss = 1 - cos_sim  # 越接近 teacher 越小；形状保持 [B,N_teacher]

        weight = self.loss_weight_fn(alpha, sigma)  # 默认全1，可扩展时间相关损失权重
        fm_loss = weight * (out - v_t) ** 2  # 预测 velocity 与解析目标的逐像素 MSE：[B,3,H,W]
        out = dict(
            fm_loss=fm_loss.mean(),  # 扩散生成损失，标量
            cos_loss=cos_loss.mean(),  # REPA 对齐损失，标量
            loss=fm_loss.mean() + self.feat_loss_weight * cos_loss.mean(),  # 用于反向传播的总标量损失
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """只保存可训练投影；frozen teacher 不包含在 checkpoint 中。"""
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars,
        )

