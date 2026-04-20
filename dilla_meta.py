import torch
import torch.nn as nn

from dataset import get_data_transforms, get_strong_transforms
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from models.uad import *
from models import vit_encoder
from models.vision_transformer import bMlp
from dataset import RealIADDataset, RealIADDataset_Pre
import argparse
from utils import *
from torch.nn import functional as F
from functools import partial
import warnings
import copy
import logging
import matplotlib.pyplot as plt
import tensorboardX
from tqdm import tqdm
from torchvision import transforms
from models.mlla import MLLABlock

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
    # print_fn(item)
    total_iters = 5000
    batch_size = args.batch_size
    image_size = args.image_size
    crop_size = args.crop_size
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    root_path = args.data_path
    train_data = RealIADDataset_Pre(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                    phase='train', beta=args.beta)
    test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                               phase="test", beta=args.beta)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder_name = 'dinov2reg_vit_small_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    transform_aug = transforms.RandomRotation(degrees=(-180, 180), expand=True)

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
        blk = MLLABlock(dim=embed_dim, input_resolution=(args.crop_size // 14, args.crop_size // 14),
                        num_heads=num_heads, mlp_ratio=1.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), drop=0.0)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = SSFiler(args=args, encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                                             target_layers=target_layers,
                                             mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
                                             fuse_layer_decoder=fuse_layer_decoder, device=device)
    checkpoints = torch.load(
        os.path.join(
            args.checkpoint_dir,
            '256224vitS_uncertainty8_maskpcaf1t5_noer_up_AC2_{}_noup_2scoreMinMaxnorm_pre1000_mlla_{}.pth'.format(
                args.beta,
                item)), map_location=device)
    model.load_state_dict(state_dict=checkpoints, strict=True)

    model = model.to(device)

    n = args.n
    selected_list = torch.zeros(len(train_data), dtype=torch.float)
    gt_list = torch.zeros(len(train_data), dtype=torch.float)
    for epoch in range(n):
        model.train()
        tqdm_obj = tqdm(range(len(train_dataloader)))
        for iteration, (img, label, idx) in zip(tqdm_obj, train_dataloader):
            # for img, label, idx in train_dataloader:
            img = img.to(device)
            # en, de, decode_include, p_n, de_max = model(img, args, batch_size=batch_size)
            en, de, decode_include, p_n, masks, anomaly_maps = model(img, args, batch_size=batch_size)
            selected_list[idx[decode_include[:p_n]]] += 1
            gt_list[idx] = label.to(torch.float)

    selected_list = selected_list >= args.t
    b = torch.sum(1 - gt_list)
    noise = torch.mean(gt_list[selected_list])
    utilize = torch.sum((1 - gt_list)[selected_list]) / b
    print("{}; Utilization: {}; Contamination Rate: {}".format(item, utilize, noise))
    torch.save(selected_list,
               os.path.join(args.save_dir, "{}_{}_{}_{}.pt".format(args.n, args.t, args.beta, item)))
    return utilize, noise


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'/data/wangfj/datasets/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./check')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=float, default=32)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
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
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--mask_num', type=int, default=4)
    parser.add_argument('--R_std_n', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--L_std_n', type=int, default=2)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--t', type=int, default=2)
    parser.add_argument('--log_index', type=float, default=0)
    args = parser.parse_args()

    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    # print_fn = logger.info
    # A unified method for fully unsupervised anomaly detection
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    print(device)
    result_list = []
    for i, item in enumerate(item_list):
        log_name = "./logs/tmp/{}_B{}_sep{}_agg{}_pr{}_ir{}_er{}_256224vitS422_uncertainty10_maskpca_noer_up_C2_{}_noup_2scoreMinMaxnorm_pre1000_mlla".format(
            item,
            args.batch_size,
            args.extra_layer_sep,
            args.extra_layer_agg,
            args.patch_ratio,
            args.image_ratio,
            args.ex_ratio,
            args.beta)
        tb_ = tensorboardX.SummaryWriter(
            log_dir=log_name)
        utilize, noise = train(item, tb_, log_name)
        result_list.append([item, utilize, noise])
    mean_utilize = np.mean([result[1] for result in result_list])
    mean_noise = np.mean([result[2] for result in result_list])
    print(result_list)
    print(
        'Mean: Utilize:{:.4f}, Noise:{:.4f}'.format(
            mean_utilize, mean_noise))
