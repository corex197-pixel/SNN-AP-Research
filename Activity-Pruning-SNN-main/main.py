import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import yaml
import random
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from typing import Optional

import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.distributed

import argparse
from thop import profile

from models.submodules.layers import static_firing_rate, get_activation, get_thresolds, static_spike_count

from models import cifar10net, resnet, vggsnn, resnet_cifar
from utils.augment import CIFAR10Policy, ImageNetPolicy, Cutout, DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.utils import RecordDict, GlobalTimer, Timer, count_conv2d
from utils.utils import DatasetSplitter, DatasetWarpper, CriterionWarpper, DVStransform, SOPMonitor
from utils.utils import is_main_process, save_on_master, tb_record, accuracy, safe_makedirs, regularize_spike, get_grad_norm, threshold_update
from spikingjelly.activation_based import functional, layer, base
from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )
    parser = argparse.ArgumentParser(description='Training')

    # dataset options
    parser.add_argument('--dataset', default='ImageNet', help='dataset type')
    parser.add_argument('--data_path', default='./datasets')
    parser.add_argument('--input_size', default=(3, 224, 224), type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=16, type=int)

    parser.add_argument('--no_aug', action='store_true', help='no augmentation')
    parser.add_argument('--re_prob', default=0.0, type=float, help='random erasing prob')
    parser.add_argument('--re_mode', default='const', type=str, help='random erasing mode')
    parser.add_argument('--re_count', default=1, type=int, help='random erasing count')
    parser.add_argument('--re_split', default=False, type=bool, help='random erasing split')
    parser.add_argument('--scale', default=[0.08, 1.0], type=float, nargs='+',
                        help='input re-scale')
    parser.add_argument('--ratio', default=[3.0 / 4.0, 4.0 / 3.0], type=float, nargs='+',
                        help='input re-ratio')
    parser.add_argument('--hflip', default=0.5, type=float, help='horizontal flip prob')
    parser.add_argument('--vflip', default=0.0, type=float, help='vertical flip prob')
    parser.add_argument('--color_jitter', default=0.4, type=float, help='color jitter')
    parser.add_argument('--auto_augment', default='rand-m9-mstd0.5-inc1', type=str,
                        help='auto augment policy')
    parser.add_argument('--num-aug-repeats', default=0, type=int, help='auto augment repeat')
    parser.add_argument('--num-aug-splits', default=0, type=int, help='auto augment split')
    parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation mode')
    parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=float, nargs='+')
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=float, nargs='+')
    parser.add_argument('--crop-pct', default=0.875, type=float)
    parser.add_argument('--crop-pct-eval', default=None, type=float)
    parser.add_argument('--crop-mode', default='random', type=str)

    parser.add_argument('--mixup-alpha', default=0.0, type=float)
    parser.add_argument('--cutmix-alpha', default=0.0, type=float)
    parser.add_argument('--cutmix-minmax', default=None, type=float, nargs='+')
    parser.add_argument('--mixup-prob', default=1.0, type=float)
    parser.add_argument('--mixup-switch-prob', default=0.5, type=float)
    parser.add_argument('--mixup-mode', default='batch', type=str)
    parser.add_argument('--label-smoothing', default=0.1, type=float)

    parser.add_argument('--dvs-augment', action='store_true', help='DVS augment')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('-eta', '--eta', default=1e-4, type=float, help='threshold regularization factor')
    parser.add_argument('-eta2', '--eta2', default=0., type=float, help='activation regularization factor')
    parser.add_argument('--model', default='sew_resnet18', help='model type')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
    parser.add_argument('--cooldown_epochs', default=10, type=int, help='cooldown epochs')
    parser.add_argument('--min-lr', default=0, type=float, help='minimum learning rate')
    parser.add_argument('--warmup-lr', default=0, type=float, help='warmup learning rate')
    parser.add_argument('--warmup-epochs', default=0, type=int, help='warmup epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')
    parser.add_argument('--accumulation-steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--activation', default='bptt', type=str, help="activation function")
    # other options
    parser.add_argument('--output-path', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--save_latest', action='store_true')
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')

    parser.add_argument('--print-freq', default=10, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--tb-interval', type=int, default=10)
    parser.add_argument('--distributed-init-mode', type=str, default='env://')
    parser.add_argument('--zero-init-residual', action='store_true',
                        help='zero init all residual blocks')
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


def setup_logger(output_path):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_path, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def load_data(
    dataset_dir: str,
    dataset_type: str,
    num_classes: int,
    distributed: bool,
    args: argparse.Namespace,
):
    if dataset_type == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                                     download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
    elif dataset_type == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
    elif dataset_type in ['ImageNet', 'ImageNet100', 'TinyImageNet']:
        dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'))
        dataset_test = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'))
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=args.T, split_by='number')
        dataset_train, dataset_test = DatasetSplitter(dataset, 0.9,
                                                      True), DatasetSplitter(dataset, 0.1, False)
    elif dataset_type == 'DVS128Gesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame',
                                      frames_number=args.T, split_by='number')
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame',
                                     frames_number=args.T, split_by='number')
    else:
        raise ValueError(dataset_type)

    if dataset_type in ['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet100']:
        if args.mixup_alpha > 0. or args.cutmix_alpha > 0. or args.cutmix_minmax is not None:
            collate_fn = FastCollateMixup(
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.label_smoothing,
                num_classes=num_classes,
            )
        else:
            collate_fn = None
        data_loader_train = create_loader(
            dataset_train,
            input_size=args.input_size,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=args.no_aug,
            re_prob=args.re_prob,
            re_mode=args.re_mode,
            re_count=args.re_count,
            re_split=args.re_split,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.auto_augment,
            num_aug_repeats=args.num_aug_repeats,
            num_aug_splits=args.num_aug_splits,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            num_workers=args.num_workers,
            distributed=distributed,
            crop_pct=args.crop_pct,
            crop_mode=args.crop_mode,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=args.input_size,
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            num_workers=args.num_workers,
            distributed=distributed,
            crop_pct=args.crop_pct_eval,
            pin_memory=True,
        )
    else:
        if args.dvs_augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=args.input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=args.input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=args.input_size[-2:], antialias=True))
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                       sampler=test_sampler,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True, drop_last=False)

    return dataset_train, dataset_test, data_loader_train, data_loader_test

def easy_load_data(dataset_dir: str,
    dataset_type: str,
    num_classes: int,
    distributed: bool,
    args: argparse.Namespace,
):
    if "CIFAR" in dataset_type and 'DVS' not in dataset_type:
        trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    Cutout(n_holes=1, length=16)
                                    ])
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif "ImageNet" in dataset_type:
        trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                Cutout(n_holes=1, length=8)
                                ])
        trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    if dataset_type == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True, transform=trans_t,
                                                     download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False, transform=trans,
                                                    download=True)
    elif dataset_type == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True, transform=trans_t,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False, transform=trans,
                                                     download=True)
    elif dataset_type in ['ImageNet', 'ImageNet100', 'TinyImageNet']:
        dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'),  transform=trans_t)
        dataset_test = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'),  transform=trans)
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=args.T, split_by='number')
        dataset_train, dataset_test = DatasetSplitter(dataset, 0.9, True), DatasetSplitter(dataset, 0.1, False)
        if args.dvs_augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=args.input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=args.input_size[-2:], antialias=True)]))
        transform_test = DVStransform(transform=transforms.Resize(size=args.input_size[-2:], antialias=True))
        
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
    else:
        raise ValueError(dataset_type)

    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)
    return dataset_train, dataset_test, data_loader_train, data_loader_test


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    print_freq: int,
    factor: int,
    scheduler_per_iter: Optional[BaseSchedulerPerIter] = None,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1,
    one_hot=None,
    args=None
):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]
    gd = None
    rs = None
    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            image, target = image.float().cuda(), target.cuda()
            if scaler is not None:
                with autocast():
                    output = model(image)
                    if one_hot:
                        loss = criterion(output, F.one_hot(target, one_hot).float())
                    else:
                        loss = criterion(output, target)
            else:
                output = model(image)
                if one_hot:
                    loss = criterion(output, F.one_hot(target, one_hot).float())
                else:
                    loss = criterion(output, target)
            loss += regularize_spike(model) * args.eta2
            metric_dict['loss'].update(loss.item())
            loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()  # type:ignore
            else:
                loss.backward()

            if rs is None:
                rs = static_firing_rate(model)
            else:
                rs += static_firing_rate(model)

            if gd is None:
                gd = get_grad_norm(model)
            else:
                gd += get_grad_norm(model)
            if (idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    threshold_update(model, optimizer.param_groups[0]["lr"])
                    scaler.update()
                else:
                    optimizer.step()
                    threshold_update(model, optimizer.param_groups[0]["lr"])
                optimizer.zero_grad()
                

            if scheduler_per_iter is not None:
                scheduler_per_iter.step()

            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1_s, batch_size)
            metric_dict['acc@5'].update(acc5_s, batch_size)

        if print_freq != 0 and ((idx + 1) % int(len(data_loader_train) / (print_freq))) == 0:
            #torch.distributed.barrier()
            metric_dict.sync()
            logger.debug(' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                idx + 1, len(data_loader_train),
                (idx + 1) * batch_size * factor / timer_container[0], metric_dict['loss'].ave,
                metric_dict['acc@1'].ave, metric_dict['acc@5'].ave))
    
    # logger.info('Train average firing rate: {}'.format(rs / len(data_loader_train)))
    logger.info('Train grad norm: {}'.format(gd / len(data_loader_train)))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave, gd / len(data_loader_train)


def evaluate(model, criterion, data_loader, print_freq, logger, one_hot):
    model.eval()
    rs = None
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.float().to(torch.device('cuda'), non_blocking=True)
            target = target.to(torch.device('cuda'), non_blocking=True)
            output = model(image)
            if one_hot:
                loss = criterion(output, F.one_hot(target, one_hot).float())
            else:
                loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            if rs is None:
                rs = static_firing_rate(model)
            else:
                rs += static_firing_rate(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))
                
    logger.info('Test average firing rate: {}'.format(rs / len(data_loader)))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave, rs / len(data_loader)


def test(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    logger: logging.Logger,
):

    safe_makedirs(os.path.join(args.output_path, 'test'))
    mon = SOPMonitor(model)

    rs = None

    logger.info('[Test]')

    model.eval()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image).mean(0)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if rs is None:
                rs = static_spike_count(model)
            else:
                rs += static_spike_count(model)

            if args.print_freq != 0 and ((idx + 1) %
                                         int(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))

    logger.info('Test total spike count: {}'.format(rs.sum() / len(data_loader_test)))

    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    step_mode = 's'
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break

    ops, params = profile(model, inputs=(inputs, ), verbose=False,
                          custom_ops={layer.Conv2d: count_conv2d})[0:2]
    if step_mode == 'm':
        ops, params = (ops / (1000**3)) / args.T, params / (1000**2)
    else:
        ops, params = (ops / (1000**3)), params / (1000**2)
    functional.reset_net(model)
    logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop
    sops = sops / (1000**3)
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / args.batch_size
    if step_mode == 's':
        sops = sops * args.T
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops + 1e-10)))

def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    safe_makedirs(args.output_path)
    logger = setup_logger(args.output_path)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))
    # load data

    dataset_type = args.dataset
    if dataset_type == 'CIFAR10':
        num_classes = 10
        one_hot = 10
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        one_hot = 100
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        one_hot = 1000
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'ImageNet100':
        num_classes = 100
        one_hot = 100
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'TinyImageNet':
        num_classes = 200
        one_hot = None
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'CIFAR10DVS':
        num_classes = 10
        one_hot = 10
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'DVS128Gesture':
        num_classes = 11
        one_hot = 11
        inputs = torch.rand(1, 1, *args.input_size).cuda()
    else:
        raise ValueError(dataset_type)
    
    if 'CIFAR' in dataset_type and 'DVS' not in dataset_type:
        dataset_train, dataset_test, data_loader_train, data_loader_test = easy_load_data(
        args.data_path, dataset_type, num_classes, distributed, args)
    else:
        dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
            args.data_path, dataset_type, num_classes, distributed, args)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    # model
    act = get_activation(args.activation)
    if args.model in cifar10net.__dict__:
        model = cifar10net.__dict__[args.model](T=args.T, 
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta, "tau":0.9}).cuda()
    elif args.model in resnet_cifar.__dict__:
        model = resnet_cifar.__dict__[args.model](T=args.T,
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta, "tau":0.9}).cuda()    
    elif args.model in vggsnn.__dict__:
        model = vggsnn.__dict__[args.model](T=args.T,
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta, "tau":0.9}).cuda()
    elif args.model in resnet.__dict__:
        model = resnet.__dict__[args.model](zero_init_residual=args.zero_init_residual, T=args.T,
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta}).cuda()
    else:
        raise NotImplementedError(args.model)

    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimzer
    # thresholds = [p for name, p in model.named_parameters() if 'thresh' in name]
    # others = [p for name, p in model.named_parameters() if 'thresh' not in name]
    # optimizer = torch.optim.SGD([{'params': others}, {'params': thresholds, 'weight_decay': 0, 'lr':0.01}],
    #                                 lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # loss_fn
    if args.mixup_alpha > 0. or args.cutmix_alpha > 0. or args.cutmix_minmax is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)
    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = CriterionWarpper(criterion_eval)

    # amp speed up
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # lr scheduler
    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=args.lr_scheduler,
        num_epochs=args.epochs,
        cooldown_epochs=args.cooldown_epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
    )

    # DDP
    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module

    # custom scheduler
    scheduler_per_iter = None
    scheduler_per_epoch = None

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        # custom scheduler
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if is_main_process():
            # search_and_absorb_threshold(model)
            test(model_without_ddp, data_loader_test, inputs, args, logger)
        return

    ##################################################
    #                   Train
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_path, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("[Train]")
    glob_gd = []
    glob_thre = []
    glob_rs = []
    glob_train_loss = []
    glob_test_loss = []
    glob_train_acc1 = []
    glob_test_acc1 = []
    glob_train_acc5 = []
    glob_test_acc5 = []
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch, optimizer.param_groups[0]["lr"]))

        with Timer(' Train', logger):
            train_loss, train_acc1, train_acc5, tmp_gd = train_one_epoch(model, criterion, optimizer,
                                                                 data_loader_train, logger,
                                                                 args.print_freq, world_size,
                                                                 scheduler_per_iter, scaler,
                                                                 args.accumulation_steps, one_hot, args)
            
            glob_gd.append(tmp_gd)
            glob_train_loss.append(train_loss)
            glob_train_acc1.append(train_acc1)
            glob_train_acc5.append(train_acc5)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        with Timer(' Test', logger):
            test_loss, test_acc1, test_acc5, rs = evaluate(model, criterion_eval, data_loader_test,
                                                       args.print_freq, logger, one_hot)

        thres = get_thresolds(model)

        logger.info('Thresholds: {}'.format(thres))
        glob_thre.append(thres)
        glob_rs.append(rs)    
        glob_test_loss.append(test_loss)
        glob_test_acc1.append(test_acc1)
        glob_test_acc5.append(test_acc5)

        if is_main_process() and tb_writer is not None:
            tb_record(tb_writer, train_loss, train_acc1, train_acc5, test_loss, test_acc1,
                      test_acc5, epoch)

        logger.info(' Test loss: {:.5f}, Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_loss, test_acc1, test_acc5))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        # custom scheduler

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_path, 'checkpoint_latest.pth'))

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_on_master(checkpoint, os.path.join(args.output_path, 'checkpoint_max_acc1.pth'))

    logger.info('Training completed.')

    np.save(os.path.join(args.output_path, 'firing_rate.npy'), np.array(glob_rs))
    np.save(os.path.join(args.output_path, 'thresholds.npy'), np.array(glob_thre))
    np.save(os.path.join(args.output_path, 'grad_norm.npy'), np.array(glob_gd))
    np.save(os.path.join(args.output_path, 'train_loss.npy'), np.array(glob_train_loss))
    np.save(os.path.join(args.output_path, 'train_acc1.npy'), np.array(glob_train_acc1))
    np.save(os.path.join(args.output_path, 'train_acc5.npy'), np.array(glob_train_acc5))
    np.save(os.path.join(args.output_path, 'test_loss.npy'), np.array(glob_test_loss))
    np.save(os.path.join(args.output_path, 'test_acc1.npy'), np.array(glob_test_acc1))
    np.save(os.path.join(args.output_path, 'test_acc5.npy'), np.array(glob_test_acc5))

    ##################################################
    #                   test
    ##################################################

    ##### reset utils #####

    # reset model
    del model, model_without_ddp

    if args.model in cifar10net.__dict__:
        model = cifar10net.__dict__[args.model](T=args.T,  activation=act, activation_kwargs={"T":args.T, "sp":args.eta, "tau":0.9}).cuda()
    elif args.model in resnet_cifar.__dict__:
        model = resnet_cifar.__dict__[args.model](T=args.T,
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta, "tau":0.9}).cuda()
    elif args.model in vggsnn.__dict__:
        model = vggsnn.__dict__[args.model](T=args.T,
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta, "tau":0.9}).cuda()
    elif args.model in resnet.__dict__:
        model = resnet.__dict__[args.model](zero_init_residual=args.zero_init_residual, T=args.T,
                                            num_classes=num_classes, activation=act, activation_kwargs={"T":args.T, "sp":args.eta}).cuda()
    else:
        raise NotImplementedError(args.model)
    model.cuda()

    try:
        checkpoint = torch.load(os.path.join(args.output_path, 'checkpoint_max_acc1.pth'),
                                map_location='cpu')
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    model.load_state_dict(checkpoint['model'])

    ##### test #####
    if is_main_process():
        test(model, data_loader_test, inputs, args, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()
