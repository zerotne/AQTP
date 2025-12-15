from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F

class AQATrackActor(BaseActor):
    """ Actor for training AQATrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """

        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data, )

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        #assert len(data['template_images']) == 1
        #assert len(data['search_images']) in [4,8,]

        template_list, template_att_list, search_list, search_att_list = [], [], [], []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)

            template_list.append(template_img_i)
            template_att_list.append(template_att_i)

        #search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_att_i = data['search_att'][i].view(-1, *data['search_att'].shape[2:])
            search_list.append(search_img_i)
            search_att_list.append(search_att_i)


           #  exp_str、templat_attn_mask和search_attn_mask是否也需要


        out_dict = self.net(template=template_list,
                            search=search_list,
                            template_attn_mask=template_att_list,
                            search_attn_mask=search_att_list,
                            exp_str=data['exp_str'],
                            return_last_attn=False,
                                )

        return out_dict


    '''def sample_negative(self, logits, gt_bboxes, size):
        bboxes = gt_bboxes # b, 4
        cood_1d = (torch.arange(size)+0.5) / size
        cood = cood_1d.unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda() # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask) # b, h, w
        mask = (mask.reshape(gt_bboxes.shape[0], -1))*(-1e9) # background == 1
        sample_logits = torch.sort(logits.reshape(gt_bboxes.shape[0], -1)+mask, descending=True, dim=-1).values[:, :9]
        return sample_logits

    def contractive_learning(self, logits, gt_bbox):  # b, n, sz, sz
        b, n, sz, sz = logits.shape
        logits = logits.reshape(-1, 1, sz, sz)
        gt_bbox = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, n, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        ctr = (gt_bbox[:, :2] + gt_bbox[:, 2:]).reshape(b*n, 1, 1, 2) / 2
        neg_logits = self.sample_negative(logits, gt_bbox, sz).to(logits)
        sample_points = ctr * 2 - 1
        pos_logits = F.grid_sample(logits.float(), sample_points.float(), padding_mode="border", align_corners=True).reshape(b*n, -1) # b, 1, 1, 10
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        target = torch.zeros(b*n).to(gt_bbox.device).long()
        return logits, target # check'''

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        #gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_bbox = gt_dict['search_anno'].view(-1, 4)
        '''#compute MMloss
        if self.loss_weight['query_aux'] > 0:
            pred_logits, target = self.contractive_learning(pred_dict['logits'], gt_bbox)
            aux_loss = self.objective['query_aux'](pred_logits, target)'''
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss     #+ self.loss_weight['query_aux'] * aux_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      #"Loss/query_aux": aux_loss,
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
