import argparse
import os

import torch
from omegaconf import OmegaConf

from visionhub.solver import Trainer
from visionhub.misc import dist_utils
from visionhub.core import LazyConfig


def _bool_action():
    return argparse.BooleanOptionalAction


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '--config-file', '-c', dest='config_file', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--device', default=None,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--pretrain', default=None, help='apply transfer learning to the backbone and encoder using DFINE weights')
    parser.add_argument('--start_epoch', '--start-epoch', dest='start_epoch', default=None, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=None)
    parser.add_argument('--test', action='store_true', default=None)
    parser.add_argument('--find_unused_params', '--find-unused-params', dest='find_unused_params', action='store_true', default=None)

    # distributed training parameters
    parser.add_argument('--world_size', '--world-size', dest='world_size', default=None, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=None, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", "--local-rank", dest='local_rank', default=None, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action=_bool_action(), default=None,
                        help="Train with mixed precision")
    parser.add_argument('--eval_interval', '--eval-interval', dest='eval_interval', default=None, type=int,
                        help='Run COCO evaluation every N epochs (default: 1, always runs on last epoch)')

    # Colab/runtime convenience overrides. These mirror common --options values
    # while keeping notebook command builders short and less fragile.
    parser.add_argument('--output_dir', '--output-dir', dest='output_dir', default=None, type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch_size', '--batch-size', dest='batch_size', default=None, type=int,
                        help='Override total batch size for train/val/test loaders')
    parser.add_argument('--num_workers', '--num-workers', dest='num_workers', default=None, type=int,
                        help='Override DataLoader workers for train/val/test loaders')
    parser.add_argument('--image_size', '--image-size', dest='image_size', default=None, type=int,
                        help='Override square training/eval image size')
    parser.add_argument('--data_root', '--dataset_root', dest='data_root', default=None, type=str,
                        help='Dataset root containing train/ and val/')
    parser.add_argument('--save_checkpoint_interval', '--save-checkpoint-interval', dest='save_checkpoint_interval', default=None, type=int)
    parser.add_argument('--grad_accum_steps', '--grad-accum-steps', dest='grad_accum_steps', default=None, type=int)
    parser.add_argument('--backup_every_n_epochs', '--backup-every-n-epochs', dest='backup_every_n_epochs', default=None, type=int)
    parser.add_argument('--gdrive_backup_path', '--gdrive-backup-path', dest='gdrive_backup_path', default=None, type=str)
    parser.add_argument('--sync_bn', '--sync-bn', dest='sync_bn', action=_bool_action(), default=None)
    parser.add_argument('--use_ema', '--use-ema', dest='use_ema', action=_bool_action(), default=None)
    parser.add_argument('--no_ema', '--no-ema', dest='use_ema', action='store_false', default=None)
    parser.add_argument('--compile_model', '--compile-model', dest='compile_model', action=_bool_action(), default=None,
                        help='Enable torch.compile for the wrapped model')
    parser.add_argument('--compile_mode', '--compile-mode', dest='compile_mode', default=None, type=str,
                        help='torch.compile mode, e.g. reduce-overhead')
    parser.add_argument('--pretrained_backbone', '--pretrained-backbone', dest='pretrained_backbone', action=_bool_action(), default=None,
                        help='Enable/disable backbone pretrained weights before instantiation')
    parser.add_argument('--debug_first_batches', '--debug-first-batches', dest='debug_first_batches', default=None, type=int,
                        help='Print stage timings for the first N training batches')

    return parser


def _set_if_present(cfg, key, value):
    if OmegaConf.select(cfg, key, default=None) is not None:
        OmegaConf.update(cfg, key, value, merge=True)


def _set_dataloader_workers(loader_cfg, num_workers):
    loader_cfg.num_workers = num_workers
    if num_workers == 0:
        if 'persistent_workers' in loader_cfg:
            loader_cfg.persistent_workers = False
        if 'prefetch_factor' in loader_cfg:
            loader_cfg.prefetch_factor = None
    else:
        if 'persistent_workers' in loader_cfg:
            loader_cfg.persistent_workers = True
        if 'prefetch_factor' in loader_cfg and loader_cfg.prefetch_factor is None:
            loader_cfg.prefetch_factor = 2


def _apply_runtime_overrides(cfg, args):
    if args.data_root:
        root = args.data_root
        for split, folder in [('dataset_train', 'train'), ('dataset_val', 'val'), ('dataset_test', 'val')]:
            if split in cfg and 'dataset' in cfg[split]:
                cfg[split].dataset.img_folder = os.path.join(root, folder, 'images')
                cfg[split].dataset.ann_file = os.path.join(root, folder, 'coco_instances.json')

    if bool(cfg.get('DETECTION_ONLY', False)):
        category_mapping = cfg.get('CATEGORY_ID_TO_CONTIGUOUS', None)
        for split in ['dataset_train', 'dataset_val', 'dataset_test']:
            if split not in cfg or 'dataset' not in cfg[split]:
                continue
            dataset_cfg = cfg[split].dataset
            dataset_cfg.num_keypoints = 0
            dataset_cfg.require_keypoints = False
            dataset_cfg.allow_empty = True
            if category_mapping is not None:
                dataset_cfg.category_id_to_contiguous = category_mapping

    if args.batch_size is not None:
        for split in ['dataset_train', 'dataset_val', 'dataset_test']:
            if split in cfg:
                cfg[split].total_batch_size = args.batch_size
                if 'batch_size' in cfg[split]:
                    del cfg[split].batch_size

    if args.num_workers is not None:
        for split in ['dataset_train', 'dataset_val', 'dataset_test']:
            if split in cfg:
                _set_dataloader_workers(cfg[split], args.num_workers)

    if args.image_size is not None:
        size = int(args.image_size)
        for split in ['dataset_train', 'dataset_val', 'dataset_test']:
            if split not in cfg:
                continue
            _set_if_present(cfg, f'{split}.collate_fn.base_size', size)
            transforms = OmegaConf.select(cfg, f'{split}.dataset.transforms', default=None)
            if transforms is not None:
                for name, transform in transforms.items():
                    if not OmegaConf.is_config(transform):
                        continue
                    if 'sizes' in transform:
                        transform.sizes = [[size, size]]
                    if 'max_size' in transform:
                        transform.max_size = size
                    if 'output_size' in transform:
                        transform.output_size = size // 2

        _set_if_present(cfg, 'model.encoder.eval_spatial_size', [size, size])
        _set_if_present(cfg, 'model.transformer.eval_spatial_size', [size, size])

    if args.pretrained_backbone is not None:
        _set_if_present(cfg, 'model.backbone.pretrained', bool(args.pretrained_backbone))

    training_updates = OmegaConf.create()
    direct_training_keys = [
        'config_file', 'device', 'seed', 'resume', 'pretrain', 'start_epoch', 'eval', 'test',
        'find_unused_params', 'world_size', 'rank', 'local_rank', 'amp',
        'eval_interval', 'output_dir', 'epochs', 'save_checkpoint_interval',
        'grad_accum_steps', 'backup_every_n_epochs', 'gdrive_backup_path',
        'sync_bn', 'use_ema', 'compile_model', 'compile_mode',
        'debug_first_batches',
    ]
    for key in direct_training_keys:
        value = getattr(args, key, None)
        if value is not None:
            training_updates[key] = value

    cfg.training_params = OmegaConf.merge(cfg.training_params, training_updates)
    return cfg


def _ensure_training_defaults(cfg):
    defaults = {
        'device': 'cuda',
        'seed': 42,
        'start_epoch': 0,
        'eval': False,
        'test': False,
        'find_unused_params': False,
        'world_size': 1,
        'rank': 0,
        'local_rank': 0,
        'amp': False,
        'eval_interval': 1,
        'compile_model': False,
        'compile_mode': 'reduce-overhead',
        'debug_first_batches': 0,
    }
    for key, value in defaults.items():
        if OmegaConf.select(cfg.training_params, key, default=None) is None:
            cfg.training_params[key] = value
    return cfg


def _normalize_dataloader_worker_settings(cfg):
    for split in ['dataset_train', 'dataset_val', 'dataset_test']:
        if split not in cfg or 'num_workers' not in cfg[split]:
            continue

        num_workers = int(cfg[split].num_workers)
        cfg[split].num_workers = max(num_workers, 0)
        if cfg[split].num_workers == 0:
            if 'persistent_workers' in cfg[split]:
                cfg[split].persistent_workers = False
            if 'prefetch_factor' in cfg[split]:
                cfg[split].prefetch_factor = None
        else:
            if 'persistent_workers' in cfg[split]:
                cfg[split].persistent_workers = True
            if 'prefetch_factor' in cfg[split] and cfg[split].prefetch_factor is None:
                cfg[split].prefetch_factor = 2
    return cfg

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.cache_size_limit = 64

    if args.data_root:
        os.environ['DETRPOSE_DATA_ROOT'] = args.data_root
        os.environ['RTMOPOSE_DATA_ROOT'] = args.data_root
        os.environ['RTMDETPOSE_DATA_ROOT'] = args.data_root
        os.environ['RTMDET_DATA_ROOT'] = args.data_root
        os.environ['DETRDET_DATA_ROOT'] = args.data_root
        os.environ['RTMODET_DATA_ROOT'] = args.data_root
        os.environ['RTMDETDET_DATA_ROOT'] = args.data_root

    cfg = LazyConfig.load(args.config_file)

    if args.options:
        cfg = LazyConfig.apply_overrides(cfg, args.options) 

    cfg = _apply_runtime_overrides(cfg, args)
    cfg = _ensure_training_defaults(cfg)
    cfg = _normalize_dataloader_worker_settings(cfg)
    print(cfg)
    
    solver = Trainer(cfg)

    runtime_args = cfg.training_params
    assert not(runtime_args.eval and runtime_args.test), "you can't do evaluation and test at the same time"

    if runtime_args.eval:
        if hasattr(cfg.model.backbone, 'pretrained'):
            cfg.model.backbone.pretrained = False
        solver.eval()
    elif runtime_args.test:
        if hasattr(cfg.model.backbone, 'pretrained'):
            cfg.model.backbone.pretrained = False
        solver.test()
    else:
        solver.fit()
    dist_utils.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RT-GroupPose training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
