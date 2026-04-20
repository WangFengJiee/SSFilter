import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import math
import random
from utils import get_gaussian_kernel
import cv2
from skimage import measure


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # [1568,512,3,3]-->[1568,1,4609]
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


def fill_diagonal_blocks(matrix, block_size, value):
    """
    Fill the diagonal of the matrix with n*n blocks.

    Args:
    - matrix: PyTorch tensor, the matrix to fill.
    - block_size: int, the block size.
    - value: float, the fill value.

    Returns:
    - matrix: PyTorch tensor, the filled matrix.
    """
    m, n = matrix.shape

    assert m == n, "Matrix must be square."

    num_blocks = m // block_size

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size

        matrix[start_idx:end_idx, start_idx:end_idx].fill_(value)

    return matrix


def cal_anomaly_maps(fs_list, ft_list):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


import scipy.ndimage as ndi


def extract_region_from_maximum(saliency_map, threshold):
    # Step 1: Find the maximum value point
    max_coords = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)

    # Step 2: Create a binary map based on the threshold
    binary_map = saliency_map >= threshold

    # Step 3: Perform connected components analysis
    labeled_map, num_features = ndi.label(binary_map)

    # Step 4: Identify the label of the region containing the max point
    max_label = labeled_map[max_coords]

    # Step 5: Create the final mask by keeping only the region with the max point
    final_region = labeled_map == max_label

    return final_region


class SSFiler(nn.Module):
    def __init__(
            self,
            args,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
            device='cuda',
    ) -> None:
        super(SSFiler, self).__init__()
        self.args = args
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size
        self.ii = 0
        self.gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    def encode(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
        return en_list, side

    def decode(self, en_list, side):
        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)
        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None
        x = x[:, 1 + self.encoder.num_register_tokens:, :]
        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            # de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def uncertainty(self, en_list, side, args, n):
        self.decoder.eval()
        scores = []
        anomaly_list = []
        ratio = 10
        for i in range(int(n / ratio)):
            en, de = self.decode([en.repeat(ratio, 1, 1) for en in en_list], side)
            anomaly_map, _ = cal_anomaly_maps(en, de)
            anomaly_map = F.interpolate(anomaly_map, size=args.crop_size, mode='bilinear', align_corners=False)
            anomaly_map = self.gaussian_kernel(anomaly_map)
            anomaly_list.extend(torch.split(anomaly_map, args.batch_size, dim=0))
            anomaly_map = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                       :int(anomaly_map.shape[1] * 0.01)].mean(dim=1)
            scores.extend(torch.split(sp_score, args.batch_size, dim=0))
        self.decoder.train()
        scores = torch.stack(scores)
        anomaly_maps = torch.sum(torch.stack(anomaly_list), dim=0)
        anomaly_list = []
        masks = []
        for i in range(anomaly_maps.shape[0]):
            image = anomaly_maps[i][0].detach().cpu().numpy()
            min_val = np.min(image)
            max_val = np.max(image)
            normalized_image = (image - min_val) / (max_val - min_val)
            image = (normalized_image * 255).astype(np.uint8)
            anomaly_list.append(image)
            max_value = np.max(image)
            threshold = max_value * 0.8  # Set to 50% of the maximum value
            threshold_down = np.quantile(image, q=0.99)
            if threshold < threshold_down:
                threshold = threshold_down
            labeled_region = extract_region_from_maximum(image, threshold)
            masks.append(labeled_region)
        stds = torch.std(scores, dim=0)
        anomaly_list = np.stack(anomaly_list)
        masks = np.stack(masks)
        return stds, masks, anomaly_list

    def forward(self, x, args, full_train=True, is_test=False, batch_size=16):
        if isinstance(x, list):
            en_list, side = x
        else:
            en_list, side = self.encode(x)
        if not is_test:
            with torch.no_grad():
                uncertainty, masks, anomaly_maps = self.uncertainty(en_list, side, args, 10)
                feature_layers = []
                for extra_layer_sep in args.extra_layer_sep:
                    feature_layers.append(en_list[extra_layer_sep][:, 1 + self.encoder.num_register_tokens:, :])
                if args.extra_layer_agg:
                    fuse_feature = []
                    for extra_layer_agg in args.extra_layer_agg:
                        fuse_feature.append(en_list[extra_layer_agg])
                    feature_layers.append(self.fuse_feature(fuse_feature)[:, 1 + self.encoder.num_register_tokens:, :])
                x_ = torch.cat(feature_layers, dim=2)
                b, hw, c = x_.shape
                h_w = int(math.sqrt(hw))
                x_ = x_.flatten(0, 1)

                x_ /= x_.norm(dim=-1, keepdim=True)
                sims = x_ @ x_.T
                self_comparative = args.self_comparative
                if self_comparative:
                    sims.fill_diagonal_(0)
                else:
                    fill_diagonal_blocks(sims, hw, 0)

                sim = torch.mean(torch.topk(sims, k=int(sims.shape[1] * args.patch_ratio), dim=1).values, dim=1)
                anomaly_score_patch_ = 1 - sim.view(batch_size, -1)
                anomaly_score_patch_sorted = torch.sort(anomaly_score_patch_, dim=1, descending=True)
                anomaly_score_patch = anomaly_score_patch_sorted.values
                anomaly_index_patch = anomaly_score_patch_sorted.indices
                top001 = int(anomaly_score_patch.shape[1] * args.image_ratio)
                anomaly_score_patch = anomaly_score_patch[:, :top001]
                anomaly_score_image = torch.mean(anomaly_score_patch, dim=1)

                anomaly_map = anomaly_score_patch_.view(b, 1, h_w, h_w)
                anomaly_map = F.interpolate(anomaly_map, size=self.args.crop_size, mode='bilinear', align_corners=False)
                anomaly_maps = self.gaussian_kernel(anomaly_map)

                if self.ii >= 1000:
                    anomaly_score_image = (minimax_norm(anomaly_score_image) + minimax_norm(torch.tensor(uncertainty,
                                                                                                         dtype=anomaly_score_image.dtype,
                                                                                                         device=anomaly_score_image.device))) / 2
                else:
                    self.ii += 1

                anomaly_index_image_sorted = torch.sort(anomaly_score_image, descending=True).indices
                index_exclude = anomaly_index_image_sorted[:int(batch_size * args.ex_ratio)].cpu().numpy().tolist()
                index_include = anomaly_index_image_sorted[int(batch_size * args.ex_ratio):].cpu().numpy().tolist()
                decode_include = []
                left_u = uncertainty[index_include]
                left_std = torch.std(left_u)
                left_mean = torch.mean(left_u)
                right_u = uncertainty[index_exclude]
                # -------------------------------------------
                indices_right = torch.where(right_u > left_mean + args.R_std_n * left_std)[0].cpu().numpy()
                indices_right2left = torch.where(right_u < left_mean)[0].cpu().numpy()

                # -------------------------------------------
                index_include = np.array(index_include)
                index_exclude = np.array(index_exclude)

                # -------------------------------------------
                # include
                decode_include.extend(index_include)
                decode_include.extend(index_exclude[indices_right2left])
                rec_num = len(decode_include)
                # exclude
                decode_include.extend(index_exclude[indices_right])
                # -------------------------------------------

        if full_train:
            if is_test:
                en, de = self.decode(en_list, side)
                return en, de
            else:
                en, de = self.decode([en[decode_include[:rec_num]] for en in en_list], side)
                return en, de, decode_include, rec_num, masks[decode_include[rec_num:]], anomaly_maps
        else:
            return index_exclude

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


def minimax_norm(data):
    min_val = torch.min(data)
    max_val = torch.max(data)

    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data


def l2_norm(data):
    data_norm = data / torch.norm(data)
    return data_norm


def batch_pca(X, k):
    """
    Perform PCA on a minibatch of data and return the top-k principal components.

    Args:
        X: Input tensor of shape (B, N, D), where B is batch size, N is sample count, D is feature dimension.
        k: Number of principal components to retain.

    Returns:
        principal_components: Tensor of shape (B, N, k) projected onto the principal component space.
    """
    B, N, D = X.shape

    X_mean = X.mean(dim=1, keepdim=True)
    X_centered = X - X_mean

    cov_matrix = torch.matmul(X_centered.transpose(1, 2), X_centered) / (N - 1)

    U, S, V = torch.linalg.svd(cov_matrix, full_matrices=False)

    U_reduced = U[:, :, :k]

    principal_components = torch.matmul(X_centered, U_reduced)

    return principal_components


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

# Calculate entropy; at 0.0, both sides should tend toward uniform distribution
# from scipy.stats import entropy
# entropy_uncertainty = entropy(uncertainty.cpu().detach().numpy(), base=2) / np.log2(batch_size)
# entropy_anomaly = entropy(anomaly_score_image.cpu().detach().numpy(), base=2) / np.log2(batch_size)
# entropy_uncertainty = normalized_entropy(uncertainty.cpu().detach().numpy())
# entropy_anomaly = normalized_entropy(anomaly_score_image.cpu().detach().numpy())
# print((entropy_uncertainty + entropy_anomaly) / 2)
# print(entropy_uncertainty)
# Calculate ranking consistency
# Calculate Kendall's Tau correlation coefficient
# tau, p_value = kendalltau(index_anomaly.cpu().detach().numpy(),
#                           index_uncertainty.cpu().detach().numpy())
# tau, p_value = spearmanr(index_anomaly.cpu().detach().numpy(), index_uncertainty.cpu().detach().numpy())
# Calculate Kendall's Tau distance
# distance = 1 - tau
# print(tau)
# print((distance + entropy_uncertainty) / 2)
