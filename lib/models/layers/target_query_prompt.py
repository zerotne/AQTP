from torch import nn, Tensor
import torch.nn.functional as F
from lib.models.transformers import PositionEmbeddingSine1D
import torch
import math
from typing import Optional
from lib.utils.misc import NestedTensor
from timm.models.vision_transformer import trunc_normal_

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim



class TQP(nn.Module):
    def __init__(self, v_dim):
        super(TQP, self).__init__()

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(v_dim, v_dim, bias=False)
        self.norm = nn.LayerNorm(v_dim)

    def forward(self, pre_p, prompt_mz, x, ):
        sim = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(prompt_mz, x))
        sim = torch.mean(sim, dim=1, keepdim=True)
        max_indices = torch.argmax(sim, dim=-1)
        # 使用索引选择对应的向量
        max_indices_expanded = max_indices.unsqueeze(-1).expand(-1,-1, x.size(-1))
        max_vectors = torch.gather(x, 1, max_indices_expanded)
        prompt = max_vectors + pre_p
        prompt = self.proj(prompt)
        prompt = self.norm(prompt)
        return prompt

class VLTQP(nn.Module):
    def __init__(self, dim,num_heads):
        super(VLTQP, self).__init__()

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.proj1 = nn.Linear(dim, dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        self.attn_drop = nn.Dropout(0.1)
        self.proj2 = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, txt_token, text_features, text_mask, vis_token, top_ratio):

        B, q_N, C = text_features.shape
        qkv = self.qkv(text_features).reshape(B, q_N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        top_lens = math.ceil(top_ratio * q_N)

        attn_topk, indices = torch.topk(attn, k=top_lens, dim=-1)
        max_vals, _ = torch.max(attn_topk, dim=-1, keepdim=True)
        attn_topk_softmax = torch.softmax(attn_topk - max_vals, dim=-1)
        new_attn = torch.zeros_like(attn, dtype=attn_topk_softmax.dtype, device=attn_topk_softmax.device,
                                    requires_grad=True)
        attn = torch.scatter(new_attn, -1, indices, attn_topk_softmax)
        attn = attn.masked_fill(
            text_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, q_N, C)
        x = self.proj2(x)
        text_features = text_features + self.proj_drop(x)
        text_features = self.norm2(text_features)


        sim = self.sim_beta + self.sim_alpha * pairwise_cos_sim(vis_token, text_features)
        sim = torch.sigmoid(sim.masked_fill(text_mask.unsqueeze(1), float('-inf'), ))

        _, top_indices = torch.topk(sim, k=top_lens, dim=-1)
        # 将 top_indices 扩展成形状为 (b, 5, 1) 的张量，以便使用 torch.gather()
        top_indices_expanded = top_indices.squeeze(1).unsqueeze(-1).expand(-1, -1, text_features.size(-1))

        # 从 x 中根据 top_indices 取出对应的向量
        top_vectors = torch.gather(text_features, dim=1, index=top_indices_expanded)
        top_vectors_mean = torch.mean(top_vectors, dim=1).unsqueeze(1)

        txt_token = top_vectors_mean + txt_token
        txt_token = self.proj1(txt_token)
        txt_token = self.norm1(txt_token)
        text_features=NestedTensor(text_features, text_mask)
        return txt_token, text_features


class VTP(nn.Module):
    def __init__(self, dim, num_heads):
        super(VTP, self).__init__()
        self.text_pos = PositionEmbeddingSine1D(dim, normalize=True)
        self.v_position_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)
        self.t_position_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.txt_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, txt_token, text_token, text_mask, vis_token, template_token, template_pos):
        #q
        txt_token = self.t_position_embed.weight.unsqueeze(1) + txt_token
        text_features_mask = NestedTensor(text_token, text_mask)
        text_pos = self.text_pos(text_features_mask)
        text_features = text_token+text_pos
        text_features = torch.cat([txt_token, text_features], dim=1)
        #k
        vis_token_k = vis_token + self.v_position_embed.weight.unsqueeze(1)
        template_token_k = template_token+template_pos
        template_token_k = torch.cat([vis_token_k, template_token_k], dim=1)
        #v
        template_token_v = torch.cat([vis_token, template_token], dim=1)
        #q_attn_mask
        attention_mask = torch.cat((torch.zeros((text_mask.size(0), 1), dtype=torch.bool, device=text_mask.device), text_mask),dim=1)
        #attention
        B, q_N, C = text_features.shape
        q = self.q(text_features).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,head,HW,C/head
        k = self.k(template_token_k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(template_token_v).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale

        attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(-1),float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        # enhanced txt_token
        x = self.norm1(text_features+x)
        txt_token_cur, text_token_cur = x[:, :1, :], x[:, 1:, :]

        sim = pairwise_cos_sim(vis_token, text_token_cur)
        sim = torch.sigmoid(sim.masked_fill(text_mask.unsqueeze(1), float('-inf'),))
        max_indices = torch.argmax(sim, dim=-1)
        max_indices_expanded = max_indices.unsqueeze(-1).expand(-1, -1, text_token_cur.size(-1))
        max_vectors = torch.gather(text_token_cur, 1, max_indices_expanded)

        txt_token_cur = txt_token_cur+max_vectors
        txt_token_cur = self.txt_proj(txt_token_cur)
        txt_token_cur = self.norm2(txt_token_cur)

        text_token_cur = NestedTensor(text_token_cur, text_mask)
        return txt_token_cur, text_token_cur

