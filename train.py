import torch
import torch.nn as nn

from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from models.uad import *
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import bMlp
from dataset import RealIADDataset
import argparse
from utils import *
from torch.nn import functional as F
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
import matplotlib.pyplot as plt
import tensorboardX
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
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    image = normalized_tensor.numpy()

    image = image.transpose((1, 2, 0))

    plt.imshow(image)
    plt.show()


def train(item, tb, log_name):
    setup_seed(111)
    print_fn(item)
    total_iters = 5000
    batch_size = args.batch_size
    image_size = args.image_size
    crop_size = args.crop_size

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    root_path = args.data_path
    train_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
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
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    # 2e-3
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))
    it = 0
    loss_list = [0]
    loss_in_list = [0]
    loss_de_list = [0]
    recall_list = [0]
    fpr_list = [0]
    err_list = []
    noise_list = []
    shoot_list = []
    transform_to_pil = transforms.ToPILImage()
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.numpy()

            en, de, decode_include, p_n, masks, anomaly_maps = model(img, args, batch_size=batch_size)
            noise_list.extend(label[decode_include[:p_n]])
            a = label[decode_include[p_n:]]
            err_list.extend(~a)
            if np.sum(label) != 0:
                recall = np.sum(a) / np.sum(label)
                fpr = np.sum(~a) / np.sum(~label)
                recall_list.append(recall)
                fpr_list.append(fpr)
                shoot_list.extend(a)
            # index_include = [i for i in range(batch_size) if i not in index_exclude]
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)

            loss_in = global_cosine_hm_percent_sum([e[:p_n] for e in en],
                                                   [d[:p_n] for d in de], p=p,
                                                   factor=0.1)
            if masks.shape[0] != 0:
                aug_idx_org = []
                aug_list = []
                for i in range(masks.shape[0]):
                    idx = random.randint(0, p_n - 1)
                    aug_idx_org.append(idx)
                    aug_list.append(decode_include[:p_n][idx])

                en_f = ((en[0] + en[1]) / 2)[aug_idx_org]
                b, c, h, w = en_f.shape
                # 1
                k = 1
                en_f_pca = batch_pca(en_f.view(b, c, h * w).permute(0, 2, 1), k)
                en_f_pca = en_f_pca.view(b, h, w, k).detach().cpu().numpy()
                binary_maps = []
                for i in range(b):
                    image = en_f_pca[i]
                    min_val = np.min(image)
                    max_val = np.max(image)
                    normalized_image = (image - min_val) / (max_val - min_val)
                    image = (normalized_image * 255).astype(np.uint8)
                    _, binary_map = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    binary_map_01 = binary_map / 255
                    t = np.mean(
                        np.concatenate(
                            [binary_map_01[:, 0], binary_map_01[:, -1], binary_map_01[0, :], binary_map_01[-1, :]]))
                    if t > 0.5:
                        # Need to invert
                        binary_map = 255 - binary_map
                    binary_map = torch.tensor(binary_map, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(1)
                    binary_map = F.interpolate(binary_map, size=args.crop_size, mode='nearest')[0][0]
                    binary_maps.append(binary_map)
                aug_blocks = []
                for i in range(masks.shape[0]):
                    mask = torch.tensor(masks[i], device=device, dtype=torch.bool)
                    non_zero_indices = torch.nonzero(mask)
                    y_min, x_min = non_zero_indices.min(dim=0)[0]
                    y_max, x_max = non_zero_indices.max(dim=0)[0]
                    aug_block = (img[decode_include[p_n:][i]] * mask)[:, y_min:y_max + 1, x_min:x_max + 1]
                    aug_blocks.append(aug_block)
                aug_imgs = []
                save_index = 0
                for i, aug_idx in enumerate(aug_list):
                    aug_img = img[aug_idx]
                    binary_map = binary_maps[i]
                    nonzero_indices = torch.nonzero(binary_map)
                    for _ in range(int(it / 1000) + 1):
                        aug_block = random.choice(aug_blocks)
                        aug_block = transform_aug(aug_block)
                        random_index = torch.randint(0, nonzero_indices.size(0), (1,)).item()
                        point = nonzero_indices[random_index]
                        h, w = aug_block.shape[1:]
                        bg = aug_block[0] == 0
                        x = int(point[0].item() * (args.crop_size / binary_map.shape[-1]))
                        y = int(point[1].item() * (args.crop_size / binary_map.shape[-1]))
                        h_l = h // 2
                        h_r = math.ceil(h / 2)
                        w_l = w // 2
                        w_r = math.ceil(w / 2)
                        if x + h_r > crop_size:
                            x = crop_size - h_r
                        if y + w_r > crop_size:
                            y = crop_size - w_r
                        if x - h_l < 0:
                            x = h_l
                        if y - w_l < 0:
                            y = w_l
                        aug_img = aug_img.contiguous()
                        aug_img[:, x - h_l:x + h_r, y - w_l:y + w_r] *= bg
                        aug_img[:, x - h_l:x + h_r, y - w_l:y + w_r] += aug_block
                    aug_imgs.append(aug_img)
                aug_imgs = torch.stack(aug_imgs)
                _, de = model(aug_imgs, args, is_test=True)
                loss_de = global_cosine_hm_percent_sum([e[aug_idx_org] for e in en],
                                                       de, p=p,
                                                       factor=0.1)
                loss = (loss_in + loss_de) / (p_n + len(aug_blocks))
            else:
                loss = loss_in / p_n
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            loss_in_list.append(loss_in.item())
            lr_scheduler.step()

            if (it + 1) % 5000 == 0:
                results = evaluation_batch(args, model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                tb.add_scalar('auroc', auroc_sp, (it + 1))
                tb.add_scalar('ap_sp', ap_sp, (it + 1))
                tb.add_scalar('f1_sp', f1_sp, (it + 1))
                tb.add_scalar('auroc_px', auroc_px, (it + 1))
                tb.add_scalar('ap_px', ap_px, (it + 1))
                tb.add_scalar('f1_px', f1_px, (it + 1))
                tb.add_scalar('aupro_px', aupro_px, (it + 1))
                print_fn(
                    '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                model.train()

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                if len(shoot_list) != 0 and len(err_list) != 0:
                    print_fn('iter [{}/{}], loss:{:.4f}, recall:{}, noise:{}, fpr:{}, shoot:{}, err:{}, cor:{}'.format(
                        it, total_iters, np.mean(loss_list), np.mean(recall_list), np.mean(noise_list),
                        np.mean(fpr_list),
                        np.mean(shoot_list), np.sum(err_list), np.sum(shoot_list)))
                    tb.add_scalar('shoot', np.mean(shoot_list), (it + 1) // 100)
                    tb.add_scalar('err', np.sum(err_list), (it + 1) // 100)
                    tb.add_scalar('cor', np.sum(shoot_list), (it + 1) // 100)
                    tb.add_scalar('noise', np.mean(noise_list), (it + 1) // 100)
                elif len(err_list) != 0:
                    print_fn('iter [{}/{}], loss:{:.4f}, recall:{}, noise:{}, fpr:{}, err:{}'.format(
                        it, total_iters, np.mean(loss_list), np.mean(recall_list), np.mean(noise_list),
                        np.mean(fpr_list), np.sum(err_list)))
                    tb.add_scalar('err', np.sum(err_list), (it + 1) // 100)
                    tb.add_scalar('noise', np.mean(noise_list), (it + 1) // 100)
                else:
                    print_fn('iter [{}/{}], loss:{:.4f}, recall:{}, noise:{}, fpr:{}'.format(
                        it, total_iters, np.mean(loss_list), np.mean(recall_list), np.mean(noise_list),
                        np.mean(fpr_list)))
                tb.add_scalar('loss_in', np.mean(loss_in_list), (it + 1) // 100)
                tb.add_scalar('loss_de', np.mean(loss_de_list), (it + 1) // 100)
                tb.add_scalar('recall', np.mean(recall_list), (it + 1) // 100)

                # print_fn('iter [{}/{}], recall:{}, fpr:{}, shoot:{}'.format(
                #     it, total_iters, np.mean(recall_list), np.mean(fpr_list), np.mean(shoot_list)))
                loss_list = [0]
                loss_in_list = [0]
                loss_de_list = [0]
                recall_list = [0]
                fpr_list = [0]
                err_list = []
                noise_list = []
                shoot_list = []

    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='./data/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./check')
    parser.add_argument('--batch_size', type=float, default=32)
    parser.add_argument('--device', type=int, default=0)
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
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--mask_num', type=int, default=4)
    parser.add_argument('--R_std_n', type=int, default=8)
    parser.add_argument('--L_std_n', type=int, default=2)
    parser.add_argument('--log_index', type=float, default=0)
    args = parser.parse_args()

    item_list = [
        'audiojack',
        'bottle_cap',
        'button_battery',
        'end_cap', 'eraser',
        'fire_hood',
        'mint',
        'mounts', 'pcb',
        'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set',
        'switch', 'tape',
        'terminalblock',
        'toothbrush', 'toy',
        'toy_brick',
        'transistor1',
        'usb',
        'usb_adaptor',
        'u_block',
        'vcpill',
        'wooden_beads',
        'woodstick',
        'zipper'
    ]
    # item_list = ['eraser']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    # A unified method for fully unsupervised anomaly detection
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    result_list = []
    for i, item in enumerate(item_list):
        # A is relatively stable and can be combined with other selection methods
        log_name = "./logs/tmp/{}_B{}_sep{}_agg{}_pr{}_ir{}_er{}_448392vitS_uncertainty8_maskpcaf1t5_noer_up_C2_{}_noup_2scoreMinMaxnorm_pre1000_mlla_ab".format(
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
