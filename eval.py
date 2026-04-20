import torch
import torch.nn as nn

from dataset import get_data_transforms, get_strong_transforms
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from models.uad import ViTill, ViTillv2, ViTill_Dev, ViTill_Dev_V3, ViTill_Dev_Uncertainty, ViTill_Dev_formal
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dataset import RealIADDataset
import argparse
from utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, WarmCosineScheduler, \
    regional_cosine_segmentation, global_cosine_reverse, global_cosine_hm_masknoise
from torch.nn import functional as F
from functools import partial
import warnings
import copy
import logging
import matplotlib.pyplot as plt
import tensorboardX

warnings.filterwarnings("ignore")


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_image(tensor):
    # Calculate the minimum and maximum values of the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Normalize the tensor
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    image = normalized_tensor.numpy()

    image = image.transpose((1, 2, 0))

    plt.imshow(image)
    plt.show()


class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        # sigma = torch.tensor([1.1, 1.5])
        self.sigma = nn.Parameter(sigma)
        # weights = str(np.around((2 * torch.log(sigma) ** 2).detach().cpu().numpy(), decimals=3))

        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * torch.log(self.sigma[i]) ** 2)
        loss += torch.log(self.sigma.pow(2).prod()) + max(0, 9 * self.sigma[0] - self.sigma[1])
        return loss


def train(item, tb, log_name):
    setup_seed(111)
    print_fn(item)
    batch_size = args.batch_size
    # image_size = 256
    # crop_size = 224
    image_size = 448
    crop_size = 392
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    root_path = args.data_path
    test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                               phase="test", beta=args.beta)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder_name = 'dinov2reg_vit_small_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), drop=0.0,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)
    # ViTill_Dev_Uncertainty
    model = ViTill_Dev_formal(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                              mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
                              fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint_path.format(item), map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    print_fn('test image number:{}'.format(len(test_data)))

    results = evaluation_batch(args, model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
    print_fn(
        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
    model.train()

    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'/data/wangfj/datasets/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./check')
    parser.add_argument('--batch_size', type=float, default=16)
    parser.add_argument('--save_name', type=str,
                        default='256224vitS_in6_de4')
    parser.add_argument('--extra_layer_sep', type=int, default=[0, 3], nargs='+', choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--extra_layer_agg', type=int, default=[0, 3, 7])
    parser.add_argument('--is_patchify', type=int, default=0)
    parser.add_argument('--patchifyMapperDim', type=int, default=768, choices=[768, 768 * 2, 768 * 3])
    parser.add_argument('--patch_ratio', type=float, default=0.001)
    parser.add_argument('--image_ratio', type=float, default=0.01)
    parser.add_argument('--self_comparative', type=int, default=0)
    parser.add_argument('--ex_ratio', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/{}_model.pth')
    parser.add_argument('--mask_num', type=int, default=4)
    parser.add_argument('--kappa_std_n', type=int, default=6)
    parser.add_argument('--log_index', type=float, default=0)
    args = parser.parse_args()

    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    result_list = []
    for i, item in enumerate(item_list):
        log_name = "./logs/dev/{}_B{}_sep{}_agg{}_pr{}_ir{}_er{}_256224vitS_in6_de4_mask1_{}".format(item, 32,
                                                                                                     args.extra_layer_sep,
                                                                                                     args.extra_layer_agg,
                                                                                                     args.patch_ratio,
                                                                                                     args.image_ratio,
                                                                                                     args.ex_ratio,
                                                                                                     args.beta)
        tb_ = tensorboardX.SummaryWriter(
            log_dir=log_name)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = train(item, tb_, log_name)
        result_list.append([item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
    log_name = "./logs/dev/RealIad_B{}_sep{}_agg{}_pr{}_ir{}_er{}_256224vitS_in6_de4_mask1_{}".format(32,
                                                                                                      args.extra_layer_sep,
                                                                                                      args.extra_layer_agg,
                                                                                                      args.patch_ratio,
                                                                                                      args.image_ratio,
                                                                                                      args.ex_ratio,
                                                                                                      args.beta)
    tb = tensorboardX.SummaryWriter(
        log_dir=log_name)
    tb.add_scalar('mean_auroc_sp', mean_auroc_sp, 0)
    tb.add_scalar('mean_ap_sp', mean_ap_sp, 0)
    tb.add_scalar('mean_f1_sp', mean_f1_sp, 0)
    tb.add_scalar('mean_auroc_px', mean_auroc_px, 0)
    tb.add_scalar('mean_ap_px', mean_ap_px, 0)
    tb.add_scalar('mean_f1_px', mean_f1_px, 0)
    tb.add_scalar('mean_aupro_px', mean_aupro_px, 0)
