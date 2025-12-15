from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from lib.models.layers.patch_embed import PatchEmbed
from lib.models.AQTP.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER

        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, num_patches_z, num_patches_x, mask=None):

        z16 = self.rgbl_patch_embed16(x)
        _, _, w, h = z16.shape
        z_c_4 = z16[:, :, int(0.5*w-1):int(0.5*w+1), int(0.5*w-1):int(0.5*w+1)]
        prompt_ms = rearrange(z_c_4, 'b c w h -> b (w h) c')
        prompt_ms = self.rgbl_tp(prompt_ms)

        B = x.shape[0]

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)  # 存在多尺度信息（b,256,2,2,256）(b,256,1,1,512)
            z = blk(z)  # 存在多尺度信息（b,64,2,2,256）(b,64,1,1,512)

        x = x[..., 0, 0, :]
        z = z[..., 0, 0, :]

        z += self.pos_embed_z
        x += self.pos_embed_x

        cls_token = self.cls_token.expand(B, -1, -1)  # 扩展复制为batchsize大小

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        x = torch.cat((cls_token, z, x), dim=1)  # 加入语义token
        # x = combine_tokens(x, z,  mode=self.cat_mode)
        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:-self.prompt_layers]:  # 加入简单的浅层多尺度语义提示器器
            x = blk(x)
        prompt_layer = 0
        for blk in self.blocks[-self.prompt_layers:]:
            vis_token, z, s = x.split([1, num_patches_z, num_patches_x], dim=1)
            if prompt_layer == 0:
                vis_prompt = torch.zeros_like(vis_token)
            vis_prompt = self.rgbl_prompt_blocks[prompt_layer](vis_prompt, prompt_ms, s)
            vis_token = vis_token + vis_prompt
            x = torch.cat((vis_token, z, s), dim=1)
            x = blk(x)
            prompt_layer = prompt_layer + 1

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        x = self.norm_(x)

        return x, aux_dict

    def forward(self, z, x, num_patches_z, num_patches_x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic HiViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, num_patches_z, num_patches_x)

        return x, aux_dict
