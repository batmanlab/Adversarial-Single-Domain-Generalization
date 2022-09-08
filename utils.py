import math

import monai
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from model.cut_model import PatchNCELoss
import numpy as np
from monai.config import DtypeLike, KeysCollection
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import convert_data_type
from monai.data.image_reader import ITKReader
import torch.nn as nn
from collections import OrderedDict

from monai.transforms import (
ScaleIntensityRangePercentiles,
    Compose,
    LoadImaged,
    AddChanneld,
    ThresholdIntensityd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
    RandAdjustContrastd,
    RandAffined,
    Rand3DElasticd,
RandGaussianNoised,
    RandRotated,
    Resized,
    RandZoomd,
    RandSpatialCropd,
    RandCropByLabelClassesd,
    Identityd,
    MapTransform,
    ToDeviced,
    Rotated
)

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks

def mix_out(x, out1, out2):

    alpha = torch.rand(2,out1.shape[0],1,1,1).cuda()

    out1 = out1 * alpha[0] + (1 - alpha[0]) * x
    out2 = out2 * alpha[1] + (1 - alpha[1]) * x

    out1 = out1 * ((torch.square(x).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True)).sqrt() /
                   (torch.square(out1.detach()).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1,
                                                                                                      keepdim=True)).sqrt()).detach()

    out2 = out2 * ((torch.square(x).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True)).sqrt() /
                   (torch.square(out2.detach()).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1,
                                                                                                      keepdim=True)).sqrt()).detach()
    return out1, out2

class ClipIntensityRangePercentilesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRangePercentiles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower percentile.
        upper: upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max]
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        allow_missing_keys: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.lower = lower
        self.upper = upper
        self.dtype = dtype

    def percentail_clip(self, input):
        upper = np.percentile(input, self.upper)
        lower = np.percentile(input, self.lower)
        input[input<=lower] = lower
        input[input>=upper] = upper

        return input

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            if key in data:
                d[key] = self.percentail_clip(d[key])
        return d

class Transformlabeld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRangePercentiles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower percentile.
        upper: upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max]
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            if key in data:
                d[key] = transform_label_cadiac(d[key])
        return d

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

import logging
import os
def set_up_logger(logs_path, log_file_name):
        # logging settings
        logger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(logs_path, log_file_name), mode="w")
        consoleHandler = logging.StreamHandler()
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s        %(message)s")
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.info("Created " + log_file_name)
        return logger

def DS_class(predict, label, metric, num_classes):

    dice_score = []

    for i in range(num_classes):
        sub_predict = torch.zeros_like(predict)
        sub_label = torch.zeros_like(label)
        sub_predict[predict==i] = 1
        sub_label[label==i] = 1
        dice_score.append(metric(sub_predict, sub_label).detach().cpu().numpy())
    return np.asarray(dice_score)



def MI_loss(src, tgt, nets, args):

    nce_layers = args.nce_layers

    crit  = PatchNCELoss().cuda()
    netG, netF = nets

    n_layers = len(nce_layers)
    feat_q = netG(tgt, nce_layers, encode_only=True)

    feat_k = netG(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = netF(feat_k, 256, None)
    feat_q_pool, _ = netF(feat_q, 256, sample_ids)

    bs = src.shape[0]

    total_nce_loss = 0.0
    for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, nce_layers):
        loss = crit(f_q, f_k, bs) * 1.0
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers




def plot_fn(img, label):
    d = int(img.shape[0] / 2)
    plt.imshow(img[d, 0, :, :].cpu().numpy(), cmap='gray')
    plt.show()
    plt.imshow(label[d, 0,:, :].cpu().numpy() * 63, cmap='gray')
    plt.show()

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def adjust_learning_rate(optimizer, epoch, lr, total_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr*(1 - math.sqrt(epoch/total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))

def cache_transformed_train_data(args, train_files, train_transforms):
    print("Caching training data set...")
    # Define SmartCacheDataset and DataLoader for training and validation
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate= args.cache_rate#,num_workers=2
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=monai.data.list_data_collate,
        worker_init_fn=worker_init_fn
    )
    return train_loader

def cache_transformed_test_data(args, train_files, train_transforms):
    print("Caching training data set...")
    # Define SmartCacheDataset and DataLoader for training and validation
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate= args.cache_rate#, num_workers=2
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=monai.data.list_data_collate,
        worker_init_fn=worker_init_fn
    )
    return train_loader

def transform_label_CT_MRI(label):

    new_label = torch.zeros_like(label)

    new_label[label==6] = 1
    new_label[label==2] = 2
    new_label[label==3] = 3
    new_label[label==1] = 4
    return new_label

def transform_label_cadiac(label):

    new_label = torch.zeros_like(label)

    new_label[label==200] = 1
    new_label[label==500] = 2
    new_label[label==600] = 3
    return new_label

train_transforms = {'Abdominal_CT': Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="PLS"),
                    ThresholdIntensityd(keys=["image"], threshold=-275, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=200, above=False),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear', 'nearest']),
RandAdjustContrastd(keys=["image"],prob=1.0, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["image"]),
RandGaussianNoised(keys=["image"],prob=0.5),
                    RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, mode=['area', 'nearest'],prob=1.0),
                    RandAffined(keys=["image", "label"], rotate_range=(0.1, 0.1, 0.1), shear_range=(0,0.5),
                                mode=['bilinear', 'nearest'],
                                prob=1.0, padding_mode="border"),
                    RandCropByLabelClassesd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=[192,192,24],
                        ratios=[1, 8, 8, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                        num_classes=14,
                        num_samples=1),
                    Rand3DElasticd(keys=["image", "label"],sigma_range=(5,7),magnitude_range=(50,150),prob=0.5,
                                   mode=['bilinear', 'nearest']),
                    # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    ToTensord(keys=["image", "label"]),
                ]),
                'Abdominal_CT_eval': Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="PLS"),
                    ThresholdIntensityd(keys=["image"], threshold=-275, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=200, above=False),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear', 'nearest']),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image", "label"]),
                ]),
                    'CHAOS_MRI':Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="PLS"),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear','nearest']),
                    ClipIntensityRangePercentilesd(keys=["image"], upper=99.5,lower=0.0),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image", "label"]),
                ])
}

prostate_transforms = {'Train': Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ClipIntensityRangePercentilesd(keys=["image"], upper=99.5,lower=0.0),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear', 'nearest']),
                    RandAdjustContrastd(keys=["image"],prob=0.7, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["image"]),
                    RandGaussianNoised(keys=["image"],prob=0.5),
                    RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, mode=['area', 'nearest'],prob=1.0),
                    RandAffined(keys=["image", "label"], rotate_range=(0.2, 0.2, 0.2), shear_range=(0,0.5),
                                mode=['bilinear', 'nearest'],
                                prob=0.7, padding_mode="border"),
                    RandCropByLabelClassesd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=[192,192,24],
                        ratios=[1, 4],
                        num_classes=2,
                        num_samples=1,
                        allow_smaller=True),
                    SpatialPadd(keys=["image", "label"],
                        spatial_size=[192,192,24]),
                    # RandSpatialCropd(
                    #     keys=["image", "label"],roi_size=[192,192,24],max_roi_size=[192,192,24]),
                    Rand3DElasticd(keys=["image", "label"],sigma_range=(5,7),magnitude_range=(50,150),prob=0.5,
                                   mode=['bilinear', 'nearest']),
                    # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    ToTensord(keys=["image", "label"]),
                ]),
                    'Test':Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear','nearest']),
                    ClipIntensityRangePercentilesd(keys=["image"], upper=99.5,lower=0.0),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image", "label"]),
                ])
}


cardiac_transforms = {'Train': Compose(
                [
                    LoadImaged(keys=["image", "label"],reader=ITKReader),
                    AddChanneld(keys=["image", "label"]),
                    # Orientationd(keys=["image", "label"]),
                    ClipIntensityRangePercentilesd(keys=["image"], upper=99.5,lower=0.0),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear', 'nearest']),
                    RandAdjustContrastd(keys=["image"],prob=0.7, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["image"]),
                    # RandGaussianNoised(keys=["image"],prob=0.5),
                    RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, mode=['area', 'nearest'],prob=1.0),
                    RandAffined(keys=["image", "label"], rotate_range=(0.2, 0.2, 0.2), shear_range=(0,0.5),
                                mode=['bilinear', 'nearest'],
                                prob=0.7, padding_mode="border"),
                    ToTensord(keys=["image", "label"]),
                    Transformlabeld(keys=["label"]),
                    RandCropByLabelClassesd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=[192,192,20],
                        ratios=[1, 4,4,4],
                        num_classes=4,
                        num_samples=1,
                        allow_smaller=True),
                    SpatialPadd(keys=["image", "label"],
                        spatial_size=[192,192,20]),
                    # RandSpatialCropd(
                    #     keys=["image", "label"],roi_size=[192,192,36],max_roi_size=[192,192,36]),
                    Rand3DElasticd(keys=["image", "label"],sigma_range=(5,7),magnitude_range=(50,150),prob=0.5,
                                   mode=['bilinear', 'nearest']),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                ]),
                    'Test':Compose(
                [
                    LoadImaged(keys=["image", "label"],reader=ITKReader),
                    AddChanneld(keys=["image", "label"]),
                    # Orientationd(keys=["image", "label"]),
                    Resized(keys=["image", "label"], spatial_size=[192, 192, -1], mode=['trilinear','nearest']),
                    ClipIntensityRangePercentilesd(keys=["image"], upper=99.5,lower=0.0),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image", "label"]),
                    Transformlabeld(keys=["label"])
                ])
}
