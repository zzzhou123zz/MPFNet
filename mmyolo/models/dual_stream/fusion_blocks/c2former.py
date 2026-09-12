import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from mmengine.model import BaseModule
from mmyolo.registry import MODELS

from .modality_weighting_layer import ModalityWeightingLayer
from .concat_fusion import ConcatFusion

@MODELS.register_module()
class C2Former(BaseModule):
    def __init__(
        self,
        in_channels : int = 256, 
        num_heads : int = 8,
        cca_strides : int = -1,
        groups : int = -1,
        offset_range_factor : int = 1,
        no_off : bool =False,
        attn_drop_rate : float =0.0,
        drop_rate : float=0.0,
        stage_idx : int = 0,
        mod_weight : bool = False,
        fusion: bool = True,
    ) -> None:
        super().__init__()
        self.block = C2FormerBlock(num_heads , in_channels // num_heads , groups , attn_drop_rate ,drop_rate,cca_strides,
                                   offset_range_factor,no_off,stage_idx,fusion)
        self.mw = None
        if mod_weight:
            self.mw = ModalityWeightingLayer(in_channels) 

    def forward(self, inputs1: torch.Tensor,inputs2: torch.Tensor):
        if self.mw is not None:
            inputs1,inputs2 = self.mw(inputs1,inputs2)
        """Forward function."""
        out = self.block(inputs1,inputs2)
        return out

class C2FormerBlock(nn.Module):

    def __init__(
            self, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor,
            no_off, stage_idx,fusion
    ):
        super(C2FormerBlock,self).__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.nc = n_head_channels * n_heads
        self.qnc = n_head_channels * n_heads * 2
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_q_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_combinq = nn.Conv2d(
            self.qnc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_k_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.vis_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.lwir_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.vis_attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.lwir_attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.vis_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)
        self.lwir_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)

        self.out_fusion = None
        if fusion:
            self.out_fusion = ConcatFusion(in_channels=self.nc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self,lwir_x, vis_x):

        B, C, H, W = vis_x.size()
        dtype, device = vis_x.dtype, vis_x.device
        # concat two tensor
        x = torch.cat([vis_x,lwir_x],1)
        combin_q = self.proj_combinq(x)

        q_off = einops.rearrange(combin_q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) 
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        vis_reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        lwir_reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.0)

        if self.offset_range_factor >= 0:
            vis_pos = vis_reference + offset
            lwir_pos = lwir_reference
        else:
            vis_pos = (vis_reference + offset).tanh()
            lwir_pos = lwir_reference.tanh()

        vis_x_sampled = F.grid_sample(
            input=vis_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=vis_pos[..., (1, 0)],  
            mode='bilinear', align_corners=True)  

        lwir_x_sampled = F.grid_sample(
            input=lwir_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=lwir_pos[..., (1, 0)],  
            mode='bilinear', align_corners=True)  

        vis_x_sampled = vis_x_sampled.reshape(B, C, 1, n_sample)
        lwir_x_sampled = lwir_x_sampled.reshape(B, C, 1, n_sample)
        
        q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
        q_lwir = q_lwir.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_vis = self.proj_k_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_vis = self.proj_v_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
        q_vis = q_vis.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_lwir = self.proj_k_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_lwir = self.proj_v_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis)  
        attn_vis = attn_vis.mul(self.scale)
        attn_vis = F.softmax(attn_vis, dim=2)
        attn_vis = self.vis_attn_drop(attn_vis)
        out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis)
        out_vis = out_vis.reshape(B, C, H, W)
        out_vis = self.vis_proj_drop(self.proj_out_vis(out_vis))

        attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir)  
        attn_lwir = attn_lwir.mul(self.scale)
        attn_lwir = F.softmax(attn_lwir, dim=2)
        attn_lwir = self.lwir_attn_drop(attn_lwir)
        out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir)
        out_lwir = out_lwir.reshape(B, C, H, W)
        out_lwir = self.lwir_proj_drop(self.proj_out_lwir(out_lwir))

        out = [out_lwir , out_vis]
        if self.out_fusion is not None:
            out = self.out_fusion(out_lwir,out_vis)
        return out
    
class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

# Modality Norm
class ModalityNorm(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(ModalityNorm, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.BatchNorm2d(nf)

        if self.learnable:
            self.conv= nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out