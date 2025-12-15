import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='AQTP', choices=['AQTP'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='multi-hivit-ep150-4frames', help='yaml configure file name')
    parser.add_argument('--online_skip', type=int, default=200, help='the skip interval of mixformer-online')
    parser.add_argument('--display_name', type=str, default='AQTPNLT')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search, template_mask, search_mask, text, display_info='AQTPNLT'):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search, template_mask, search_mask, text, False,False,[], None, None,True),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        text_features, txt_token = model.forward_text(text, device=search.device)
        for i in range(T_w):
            _ = model(template, search, template_mask, search_mask, None, False, False, [], text_features, txt_token,
                      False)
        start = time.time()
        text_features, txt_token = model.forward_text(text, device=search.device)
        for i in range(T_t):
            _ = model(template, search, template_mask, search_mask, None, False,False,[], text_features, txt_token,False)
        torch.cuda.synchronize()
        end = time.time()
        cost_time = end - start
        avg_lat = (end - start) / T_t
        print("Cost time: {}".format(cost_time))
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m".format(display_info, 1.0 / avg_lat))

        #start = time.time()
        #text_features, txt_token = model.forward_text(text, device=search.device)
        #for i in range(T_t):
        #    _ = model(template, search, template_mask, search_mask, None, False,False,[], text_features, txt_token,False)
        #end = time.time()
        #avg_lat = (end - start) / T_t
        #print("The average backbone latency is %.2f ms" % (avg_lat * 1000))


def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    img_mask = torch.randint(0, 2, (bs, sz, sz)).type(torch.float)

    return img_patch, img_mask




if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    #z_sz=int(192)
    #x_sz=int(384)


    text = ['the man in white gym suit with number 41 and black hat']

    if args.script == "AQTPPack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_AQTP
        model = model_constructor(cfg, training=False)
        # get the template and search
        template, template_mask = get_data(bs, z_sz)
        search, search_mask = get_data(bs, x_sz)

        # transfer to device
        model = model.to(device)
        template = template.to(device)
        template_mask=template_mask.to(device)
        search = search.to(device)
        search_mask=search_mask.to(device)



        evaluate_vit(model, template, search,template_mask, search_mask, text, args.display_name)


    else:
        raise NotImplementedError
