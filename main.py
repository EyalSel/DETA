# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torch.utils.data import DataLoader

import datasets.samplers as samplers
import util.misc as utils
from datasets.coco import make_coco_transforms
from engine import get_preds
from models import build_model
from util.misc import get_local_rank, get_local_size


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector',
                                     add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"],
                        type=str,
                        nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'],
                        type=str,
                        nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False,
                        action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights',
                        type=str,
                        default=None,
                        help=("Path to the pretrained model. "
                              "If set, only the mask head will be trained"))

    # * Backbone
    parser.add_argument('--backbone',
                        default='resnet50',
                        type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation',
                        action='store_true',
                        help=("If true, we replace stride with dilation in "
                              "the last convolutional block (DC5)"))
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )
    parser.add_argument('--position_embedding_scale',
                        default=2 * np.pi,
                        type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels',
                        default=4,
                        type=int,
                        help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers',
                        default=6,
                        type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers',
                        default=6,
                        type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument(
        '--dim_feedforward',
        default=1024,
        type=int,
        help=("Intermediate size of the feedforward layers in the "
              "transformer blocks"))
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries',
                        default=300,
                        type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks',
                        action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument(
        '--no_aux_loss',
        dest='aux_loss',
        action='store_false',
        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--assign_first_stage', action='store_true')
    parser.add_argument('--assign_second_stage', action='store_true')
    parser.add_argument('--set_cost_class',
                        default=2,
                        type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox',
                        default=5,
                        type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou',
                        default=2,
                        type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--bigger', action='store_true')

    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune',
                        default='',
                        help='finetune from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode',
                        default=False,
                        action='store_true',
                        help='whether to cache images on memory')
    parser.add_argument('--MEVA_video',
                        default='',
                        help='Path to MEVA avi file')

    return parser


class MEVASensor:

    def __init__(self, data_path):
        cap = cv2.VideoCapture(str(data_path))
        self.cap = cap
        self.total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def total_num_frames(self):
        return self.total_length

    def get_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        assert ret, frame_index
        return {"center_camera_feed": frame}

    def __del__(self):
        self.cap.release()


class MEVADataset(data.Dataset):

    def __init__(self, meva_video_path, bigger) -> None:
        self.local_rank = get_local_rank()
        self.local_size = get_local_size()
        # The MEVASensor uses cv2.VideoCapture, which apparently when set as a
        # variable during the dataset init creates a deadlock in a multi-worker
        # dataloader
        # (https://discuss.pytorch.org/t/multiprocess-cv2-data-loader/54039).
        # One way to apparently get around this is to initialize the
        # cv2.VideoCapture on the first call to __getitem__.

        # However, one issue is that the __len__ function needs the
        # cv2.VideoCapture for the length of the avi video, so we resort to the
        # cumbersome solution where the cv2.VideoLoader is created, the the
        # length is retrieved, and then the cv2.VideoLoader structure is
        # explicitly destroyed.
        temp = MEVASensor(meva_video_path)
        self.length = temp.total_num_frames()
        del temp
        self.sensor = None
        self.meva_video_path = meva_video_path
        self._transforms = make_coco_transforms("val", bigger)

    def __getitem__(self, idx):
        if self.sensor is None:
            self.sensor = MEVASensor(self.meva_video_path)
        img = self.sensor.get_frame(idx)["center_camera_feed"]
        to_pil = torchvision.transforms.ToPILImage()
        img = to_pil(img)
        img, _ = self._transforms(img, None)
        return img, {}

    def __len__(self):
        return self.length


def main(args):
    print(args)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, _, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    # build datasets
    # dataset_val = build_dataset(image_set='val', args=args)
    dataset_val = MEVADataset(args.MEVA_video, args.bigger)

    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val,
                                                          shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val,
                                                      shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [{
        "params": [
            p for n, p in model_without_ddp.named_parameters()
            if not match_name_keywords(n, args.lr_backbone_names)
            and not match_name_keywords(n, args.lr_linear_proj_names)
            and p.requires_grad
        ],
        "lr":
        args.lr,
    }, {
        "params": [
            p for n, p in model_without_ddp.named_parameters() if
            match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
        ],
        "lr":
        args.lr_backbone,
    }, {
        "params": [
            p for n, p in model_without_ddp.named_parameters()
            if match_name_keywords(n, args.lr_linear_proj_names)
            and p.requires_grad
        ],
        "lr":
        args.lr * args.lr_linear_proj_mult,
    }]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts,
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            if 'class_embed' in k:
                print('removing', k)
                del state_dict[k]

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            state_dict, strict=False)
        unexpected_keys = [
            k for k in unexpected_keys
            if not (k.endswith('total_params') or k.endswith('total_ops'))
        ]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        print('finetuning from epoch', checkpoint['epoch'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        unexpected_keys = [
            k for k in unexpected_keys
            if not (k.endswith('total_params') or k.endswith('total_ops'))
        ]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if (not args.eval and 'optimizer' in checkpoint
                and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint):
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from
            # checkpoint and also modify lr scheduler (e.g., decrease lr in
            # advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to '
                    'True, so args.lr_drop would override lr_drop in resumed '
                    'lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(
                    map(lambda group: group['initial_lr'],
                        optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model

    assert args.eval
    video_path = args.MEVA_video
    scenario = Path(video_path).name.replace(".avi", "")
    get_preds(model,
              postprocessors,
              data_loader_val,
              device,
              args.output_dir,
              scenario=scenario)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script',
        parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
