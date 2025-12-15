"""
Basic AQATrack model.
"""
import math
import os
from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
import torch.nn.functional as F
from lib.models.layers.head import build_box_head
from lib.models.aqatrack.hivit import hivit_small, hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.transformers.position_encoding import PositionEmbeddingSine2D
from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.position_encoding import build_position_encoding
from lib.utils.misc import NestedTensor
from lib.models.transformers import VisionLanguageFusionModule, PositionEmbeddingSine1D
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from lib.models.layers.target_query_prompt import VLTQP

class AQATrack(nn.Module):
    """ This is the base class for AQATrack """

    def __init__(self, transformer, box_head, transformer_dec,z_position_encoding,
                 x_position_encoding, memory_pos_embedding,aux_loss=False, head_type="CORNER",
                 tokenizer=None, text_encoder=None, num_vlfusion_layers=0,img_size=(128, 256), patch_size=16,):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.dim = hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.rgbl_box_head = box_head

        # text encoder
        self.rgbl_tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.rgbl_transformer_dec = transformer_dec
        self.rgbl_vl_fusion = VisionLanguageFusionModule(dim=hidden_dim, num_heads=8, attn_drop=0.1, proj_drop=0.1,
                                                         num_vlfusion_layers=num_vlfusion_layers, )
        self.rgbl_vltqp = VLTQP(dim=hidden_dim,num_heads=8)
        self.rgbl_memory_pos_embedding = memory_pos_embedding
        self.rgbl_z_position_encoding = z_position_encoding
        self.rgbl_x_position_encoding = x_position_encoding

        text_feat_size = self.text_encoder.config.hidden_size
        self.rgbl_text_adj = nn.Sequential(
            nn.Linear(text_feat_size, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(0.1),)
        self.rgbl_text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        self.rgbl_query_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)
        self.rgbl_v_position_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)
        self.rgbl_l_position_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)

        self.num_patches_z = (img_size[0] // patch_size)**2
        self.num_patches_x = (img_size[1] // patch_size)**2
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.rgbl_box_head = _get_clones(self.rgbl_box_head, 6)


    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            cls_token = '[CLS]'
            cls_id = self.rgbl_tokenizer.convert_tokens_to_ids(cls_token) #将cls_id  mask

            tokenized = self.rgbl_tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            text_attention_mask = tokenized.attention_mask
            # 在每个 input_ids 子列表的最前面加入 cls_id
            tokenized['input_ids'] = torch.cat((torch.full((tokenized['input_ids'].size(0), 1), cls_id, dtype=torch.long, device=tokenized['input_ids'].device), tokenized['input_ids']),dim=1)
            # 在每个 attention_mask 子列表的最前面添加 1
            tokenized['attention_mask'] = torch.cat((torch.ones((tokenized['attention_mask'].size(0), 1), dtype=torch.long,device=tokenized['attention_mask'].device), tokenized['attention_mask']), dim=1)
            encoded_text = self.text_encoder(**tokenized)
            text_attention_mask = text_attention_mask.ne(1).bool()
            text_features = encoded_text.last_hidden_state
            text_features = self.rgbl_text_adj(text_features)

            txt_token = text_features[:, :1,:]
            text_features = text_features[:, 1:,:]
            text_features = NestedTensor(text_features, text_attention_mask)

        else:
            raise ValueError("Please make sure the caption is a list of string")
        return text_features, txt_token

    def memory_mask_pos_enc(self, attn_mask, feat_sz):
        """
        attn_mask: (B, img_H, img_W)
        feat_sz: feature size
        """
        batch_size = attn_mask.size(0)
        attn_mask = attn_mask.to(torch.float32)
        attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(feat_sz, feat_sz)).to(torch.bool).squeeze(1)
        pos_embeds = self.rgbl_memory_pos_embedding(attn_mask) # sine position encoding  (B, C, feat_sz, feat_sz)
        pos_embeds = pos_embeds.view(batch_size, self.dim, -1).transpose(1, 2)
        return pos_embeds


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_attn_mask: torch.Tensor,
                search_attn_mask: torch.Tensor,
                exp_str=None,
                return_last_attn=False,
                training=True, #True
                tgt_pre=None,
                text_features=None,
                txt_token=None,
                test_speeds=False
                ):
        b0, num_search = template[0].shape[0], len(search)
        if training:
            search = torch.cat(search, dim=0)
            search_attn_mask = torch.cat(search_attn_mask, dim=0)
            template = template[0].repeat(num_search, 1, 1, 1)
            template_attn_mask = template_attn_mask[0].repeat(num_search,1,1)

            exp_str = exp_str * num_search

            text_features, txt_token = self.forward_text(exp_str, device=search.device)

        if test_speeds:
            text_features, txt_token = self.forward_text(exp_str, device=search.device)

        #NL
        bt, ct, z1, z2 = template.shape
        bm, m1, m2 = template_attn_mask.shape
        template = torch.zeros((bt,ct,z1,z2), requires_grad=True).to(search.device)
        template_attn_mask = torch.ones((bm, m1, m2), requires_grad=True).to(search.device).to(torch.bool)

        img_feat, aux_dict = self.backbone(z=template, x=search, num_patches_z=self.num_patches_z,
                                           num_patches_x=self.num_patches_x, return_last_attn=return_last_attn, ) #x=[B,N,C]
        vis_token, z, x = img_feat.split([1, self.num_patches_z, self.num_patches_x], dim=1)

        # generate position encoding for the encoder output and text features
        template_pos_embeds = self.memory_mask_pos_enc(template_attn_mask, int(self.feat_sz_s*0.5))
        search_pos_embeds = self.memory_mask_pos_enc(search_attn_mask, self.feat_sz_s)
        enc_pos_embeds = torch.cat([template_pos_embeds, search_pos_embeds], dim=1)
        text_pos = self.rgbl_text_pos(text_features)  # [batch_size, length, c]

        # text_enhanced
        txt_token, text_features = self.rgbl_vltqp(txt_token, text_features.tensors, text_features.mask, vis_token, top_ratio=0.5)

        img_feat = torch.cat([z, x], dim=1)
        # Vision Language Multi-Modal Fusion
        img_feat = self.rgbl_vl_fusion(img_feat, text_features.tensors, query_pos=enc_pos_embeds,
                           memory_pos=text_pos, memory_key_padding_mask=text_features.mask, need_weights=False)


        input_dec = img_feat
        batches = [[] for _ in range(b0)]
        visual_query = [[] for _ in range(b0)]
        text_query = [[] for _ in range(b0)]
        for i, input in enumerate(input_dec):
            batches[i % b0].append(input.unsqueeze(0))
        for i, input in enumerate(txt_token):
            text_query[i % b0].append(input.unsqueeze(0))
        for i, input in enumerate(vis_token):
            visual_query[i % b0].append(input.unsqueeze(0))

        x_decs = []
        query_embed = self.rgbl_query_embed.weight.unsqueeze(1)  #
        t_l_pos_embed = self.rgbl_l_position_embed.weight.unsqueeze(1)
        t_v_pos_embed = self.rgbl_v_position_embed.weight.unsqueeze(1)

        for i, batch in enumerate(batches):
            if len(batch) == 0:
                continue
            if training:
                tgt_pre = []
            text_query_tokens = text_query[i]
            visual_query_tokens = visual_query[i]

            for j, input in enumerate(batch):
                #设置 将tat_all[j]=traget_token[b0*j+i:b0*j+i+1,:,:]
                text_query_token = text_query_tokens[j]
                visual_query_token = visual_query_tokens[j]
                z_pos_embed = self.rgbl_z_position_encoding(1)  #位置编码需要注意下
                x_pos_embed = self.rgbl_x_position_encoding(1)
                memory_pos_embed = torch.cat([z_pos_embed, x_pos_embed], dim=0)

                tgt_out = self.rgbl_transformer_dec(input.transpose(0, 1), text_query_token, t_l_pos_embed, visual_query_token, t_v_pos_embed, tgt_pre, pos=memory_pos_embed,  query_pos=query_embed)
                x_decs.append(tgt_out[0])

                if training:
                    tgt_pre.append(tgt_out[0])

            if not training:
                if len(tgt_pre) < 3:  # num_search-1
                    tgt_pre.append(tgt_out[0])
                else:
                    tgt_pre.pop(0)
                    tgt_pre.append(tgt_out[0])

        batch0 = []
        if not training:
            batch0.append(x_decs[0])
        else:
            batch0 = [x_decs[i + j*num_search] for j in range(b0) for i in range(num_search)]

        x_dec = torch.cat(batch0, dim=1)

        # Forward head
        feat_last = img_feat
        if isinstance(img_feat, list):
            feat_last = img_feat[-1]

        out = self.forward_head(feat_last, x_dec, None) # STM and head

        out.update(aux_dict)
        if not training:
            out['tgt'] = tgt_pre
        return out

    def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # STM
        enc_opt = cat_feature[:, -self.feat_len_s:] 
        dec_opt = out_dec.transpose(0,1).transpose(1,2) 
        att = torch.matmul(enc_opt, dec_opt)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        #Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.rgbl_box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.rgbl_box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def track(self, template, search, tgt_pre, exp_str, return_last_attn=False,):

        text_features, txt_token = self.forward_text(exp_str, device=template.tensors.device)

        b0, num_search = template.tensors[0].shape[0], len(search.tensors)

        img_feat, aux_dict = self.backbone(z=template.tensors, x=search.tensors,
                                    return_last_attn=return_last_attn, )

        vis_token, z, x = img_feat.split([1, self.num_patches_z, self.num_patches_x], dim=1)

        template_pos_embeds = self.memory_mask_pos_enc(template.mask, int(self.feat_sz_s * 0.5))
        search_pos_embeds = self.memory_mask_pos_enc(search.mask, self.feat_sz_s)
        enc_pos_embeds = torch.cat([template_pos_embeds, search_pos_embeds], dim=1)
        text_pos = self.rgbl_text_pos(text_features)  # [batch_size, length, c]

        img_feat = torch.cat([z, x], dim=1)
        img_feat = self.rgbl_vl_fusion(img_feat, text_features.tensors, query_pos=enc_pos_embeds,
                                memory_pos=text_pos, memory_key_padding_mask=text_features.mask, need_weights=False)

        # 可以简化代码
        input_dec = img_feat
        batches = [[] for _ in range(b0)]
        visual_query = [[] for _ in range(b0)]
        text_query = [[] for _ in range(b0)]
        for i, input in enumerate(input_dec):
            batches[i % b0].append(input.unsqueeze(0))
        for i, input in enumerate(txt_token):
            text_query[i % b0].append(input.unsqueeze(0))
        for i, input in enumerate(vis_token):
            visual_query[i % b0].append(input.unsqueeze(0))

        x_decs = []
        query_embed = self.rgbl_query_embed.weight.unsqueeze(1) #
        t_l_pos_embed = self.rgbl_l_position_embed.weight.unsqueeze(1)
        t_v_pos_embed = self.rgbl_v_position_embed.weight.unsqueeze(1)
        #可以简化代码
        for i, batch in enumerate(batches):
            if len(batch) == 0:
                continue
            text_query_tokens = text_query[i]
            visual_query_tokens = visual_query[i]

            for j, input in enumerate(batch):
                text_query_token = text_query_tokens[j]
                visual_query_token = visual_query_tokens[j]
                z_pos_embed = self.rgbl_z_position_encoding(1)
                x_pos_embed = self.rgbl_x_position_encoding(1)
                memory_pos_embed = torch.cat([z_pos_embed, x_pos_embed], dim=0)

                tgt_out= self.rgbl_transformer_dec(input.transpose(0, 1), text_query_token,t_l_pos_embed,visual_query_token,t_v_pos_embed,tgt_pre, pos=memory_pos_embed,  query_pos=query_embed)
                x_decs.append(tgt_out[0])

            if len(tgt_pre) < 3:  # num_search-1
                tgt_pre.append(tgt_out[0])
            else:
                tgt_pre.pop(0)
                tgt_pre.append(tgt_out[0])

        batch0 = []
        batch0.append(x_decs[0])

        x_dec = torch.cat(batch0, dim=1)
        feat_last = img_feat
        if isinstance(img_feat, list):
            feat_last = img_feat[-1]
        out = self.forward_head(feat_last, x_dec, None)  # STM and head

        out.update(aux_dict)
        out['tgt'] = tgt_pre
        # print(len(out['tgt']), 'out[tgt]')
        return out




def build_aqatrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('AQATrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,prompt_layers=cfg.MODEL.BACKBONE.PROMPT_LAYERS)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,prompt_layers=cfg.MODEL.BACKBONE.PROMPT_LAYERS)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # Build Text Encoder
    tokenizer, text_encoder = None, None
    if cfg.MODEL.TEXT_ENCODER == 'roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained(
             '/home-gxu/wwj22/MPLT-text/roberta-base',
            local_files_only=True)  # load pretrained RoBERTa Tokenizer
        text_encoder = RobertaModel.from_pretrained(
             '/home-gxu/wwj22/MPLT-text/roberta-base',
            local_files_only=True)  # load pretrained RoBERTa model
    elif cfg.MODEL.TEXT_ENCODER == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('pretrained_networks/bert-base-cased', local_files_only=True)
        text_encoder = BertModel.from_pretrained('bert-base-cased', local_files_only=True)


    transformer_dec = build_transformer_dec(cfg, hidden_dim)

    z_position_encoding = build_position_encoding(cfg, sz=int(cfg.MODEL.BACKBONE.Z_SIZE/cfg.MODEL.BACKBONE.STRIDE))  #
    x_position_encoding = build_position_encoding(cfg, sz=int(cfg.MODEL.BACKBONE.X_SIZE / cfg.MODEL.BACKBONE.STRIDE))
    N_steps = hidden_dim // 2
    memory_pos_embedding = PositionEmbeddingSine2D(N_steps, normalize=True)
    box_head = build_box_head(cfg, hidden_dim)
    model = AQTP(
        backbone,
        box_head,
        transformer_dec,
        z_position_encoding,
        x_position_encoding,
        memory_pos_embedding,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        num_vlfusion_layers=cfg.MODEL.VLFUSION_LAYERS,
        img_size=(int(cfg.MODEL.BACKBONE.Z_SIZE),int(cfg.MODEL.BACKBONE.X_SIZE)),
        patch_size=cfg.MODEL.BACKBONE.STRIDE
    )

    if 'AQTP' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
