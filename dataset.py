import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


class MVTecLOCODataset_Ab(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'ablation_train':
            self.img_path = root
        elif phase == 'ablation_test':
            self.img_path = os.path.join(root, 'test')
        self.phase = phase
        self.transform = transform
        # load dataset
        self.img_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        return np.array(img_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_type


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        elif phase == 'test':
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        elif phase == 'ablation_train':
            self.img_path = root
        elif phase == 'ablation_test':
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        if self.phase == 'ablation_train' or self.phase == 'ablation_test':
            return img, label
        else:
            return img, gt, label, img_path


class MVTecDatasetNoise():
    def __init__(self):
        pass

    def get_dataset(self, root, transform, gt_transform, phase, beta='0.0'):
        self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.beta = float(beta)
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        train_dataset = MVTecDataset(root=root, transform=transform, gt_transform=gt_transform,
                                     phase='train')
        test_dataset = MVTecDataset(root=root, transform=transform, gt_transform=gt_transform,
                                    phase='test')
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(root + "/" + 'test')
        anomaly_index = [index for index in range(len(test_dataset)) if self.labels[index] == 1]
        train_length = len(train_dataset)
        anomaly_number = len(anomaly_index)
        test_norm_number = len(test_dataset) - anomaly_number
        noise_number = int(self.beta * train_length)
        noise_limit = int(anomaly_number / 2)
        while noise_number > noise_limit:
            train_length -= 1
            noise_number = int(self.beta * train_length)
        train_dataset = Subset(train_dataset, range(train_length))

        # noise_index_path = Path("noise_index" + "/" + "mvtec" + "_noise"
        #                         + str(beta) + "_fold" + str(0))
        #
        # noise_index_path.mkdir(parents=True, exist_ok=True)
        # path = noise_index_path / (subdataset + "-noise" + str(noise) + ".pth")
        # path = Path(
        #     "/home/wangfj/code/AnomalyDetection-SoftPatch-main/noise_index/mvtec_noise{}_fold0/{}-noise{}.pth".format(
        #         beta, root.split('/')[-1], beta))
        # if path.exists():
        #     noise_index = torch.load(path)
        #     assert len(noise_index) == noise_number
        # else:
        #     noise_index = random.sample(anomaly_index, noise_number)
        #     torch.save(noise_index, path)
        #
        noise_index = random.sample(anomaly_index, noise_number)

        noise_dataset = Subset(test_dataset, noise_index)
        train_dataset = ConcatDataset([train_dataset, noise_dataset])

        new_test_data_index = list(set(range(len(test_dataset))) - set(noise_index))
        test_dataset = Subset(test_dataset, new_test_data_index)

        return train_dataset, test_dataset, noise_number, test_norm_number

    def load_dataset(self, load_from):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(load_from)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(load_from, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(load_from, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(load_from, defect_type) + "/*.bmp") + \
                            glob.glob(os.path.join(load_from, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(load_from, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(load_from, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(load_from, defect_type) + "/*.bmp") + \
                            glob.glob(os.path.join(load_from, defect_type) + "/*.jpg")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, beta='0.0'):
        self.img_path = os.path.join(root, 'realiad_512', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons_fuiad_{}'.format(beta), category + '.json')
        # json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_512', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_512', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset_Draem(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, beta='0.0'):
        self.img_path = os.path.join(root, 'realiad_512', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons_fuiad_{}'.format(beta), category + '.json')
        # json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_512', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_512', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset_Filtered(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, index_file, beta='0.0'):
        self.img_path = os.path.join(root, 'realiad_512', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons_fuiad_{}'.format(beta), category + '.json')
        # json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_512', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_512', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        index_file = torch.load(index_file)

        self.img_paths = np.array(self.img_paths)[index_file]
        self.gt_paths = np.array(self.gt_paths)[index_file]
        self.labels = np.array(self.labels)[index_file]
        self.types = np.array(self.types)[index_file]
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset_Pre(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, beta='0.0'):
        self.img_path = os.path.join(root, 'realiad_512', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons_fuiad_{}'.format(beta), category + '.json')
        # json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_512', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_512', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label, idx

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except (IOError, OSError):
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label
