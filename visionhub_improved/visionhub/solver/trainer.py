"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM (https://github.com/Intellindust-AI-Lab/DEIM/)
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/)
Copyright (c) 2023 RT-DETR Authors. All Rights Reserved.
"""

import json
import logging
import datetime
import re
from pathlib import Path
from omegaconf import OmegaConf
from collections import Counter


import time
import atexit
import random
import numpy as np
import shutil
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from ..misc.metrics import BestMetricHolder
from ..misc.profiler import stats
from ..misc import dist_utils
from ..core import instantiate

from .engine import train_one_epoch, evaluate

def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_backup_epoch = -1

    def _primary_eval_type(self):
        evaluator = getattr(self, "evaluator", None)
        if evaluator is None:
            return "keypoints"
        if "keypoints" in getattr(evaluator, "iou_types", []):
            return "keypoints"
        if "bbox" in getattr(evaluator, "iou_types", []):
            return "bbox"
        return "keypoints"

    def _primary_eval_stats_key(self):
        eval_type = self._primary_eval_type()
        if eval_type == "bbox":
            return "coco_eval_bbox"
        return "coco_eval_keypoints"

    def _primary_eval_names(self):
        if self._primary_eval_type() == "bbox":
            return [
                "mAP50:95",
                "mAP50",
                "mAP75",
                "mAP50:95-Small",
                "mAP50:95-Medium",
                "mAP50:95-Large",
            ]
        return [
            "sAP50:95",
            "sAP50",
            "sAP75",
            "sAP50:95-Medium",
            "sAP50:95-Large",
        ]

    def _primary_map_value(self, test_stats):
        stats_key = self._primary_eval_stats_key()
        stats = test_stats.get(stats_key)
        if not stats:
            raise KeyError(f"Expected evaluation stats '{stats_key}' in test results.")
        return stats[0]

    def _link_rtmo_dcc(self, *modules):
        """Share RTMOPose DCC with criterion/postprocessor without changing APIs."""
        model = getattr(self, "model_without_ddp", None)
        dcc = getattr(model, "dcc", None)
        if dcc is None:
            return
        for module in modules:
            if module is not None and hasattr(module, "set_dcc"):
                module.set_dcc(dcc)

    def _backup_output_to_gdrive(self, epoch, should_backup=None):
        """Backup output directory to Google Drive on the configured backup cadence."""
        args = self.cfg.training_params

        if not hasattr(args, 'gdrive_backup_path'):
            return

        if should_backup is None:
            backup_interval = int(getattr(args, 'backup_every_n_epochs', 0) or 0)
            if backup_interval <= 0:
                return
            should_backup = (
                ((epoch + 1) % backup_interval == 0)
                or (epoch == args.epochs - 1)
            )

        if not should_backup:
            return

        # Avoid duplicate backups
        if epoch == self.last_backup_epoch:
            return

        if not dist_utils.is_main_process():
            return

        try:
            backup_path = Path(args.gdrive_backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)

            # Copy all files from output directory to backup
            for item in self.output_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_path / item.name)
                elif item.is_dir() and item.name not in ['summary']:  # Skip tensorboard logs
                    shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)

            self.last_backup_epoch = epoch
            print(f'[Backup] Epoch {epoch+1}: Output backed up to Google Drive at {backup_path}')
        except Exception as e:
            print(f'[Backup] Warning: Backup failed for epoch {epoch+1}: {e}')

    def _normalize_config_module(self, config_value):
        if not config_value:
            return None

        normalized = str(config_value).strip().replace("\\", "/")
        if normalized.endswith(".py"):
            normalized = normalized[:-3]
        normalized = normalized.lstrip("./")

        configs_match = re.search(r"(^|/)(configs(?:/|$).*)", normalized)
        if configs_match:
            normalized = configs_match.group(2)

        if "/" in normalized:
            normalized = normalized.replace("/", ".")
        normalized = normalized.strip(".")
        if not normalized.startswith("configs."):
            return None
        return normalized

    def _infer_model_family(self, config_value):
        value = str(config_value or "").lower()
        if "detrdet" in value or "detr_detect" in value or "detr-detect" in value:
            return "detrdet"
        if "rtmodet" in value or "rtmo_detect" in value or "rtmo-detect" in value:
            return "rtmodet"
        if "rtmdetpose" in value or "rtmdet_pose" in value or "rtmdet-pose" in value:
            return "rtmdetpose"
        if "rtmdetdet" in value or "rtmdet_detect" in value or "rtmdet-detect" in value:
            return "rtmdetdet"
        if "rtmopose" in value or "rtmo" in value:
            return "rtmopose"
        if "rtmdet" in value:
            return "rtmdetdet"
        if "detrpose" in value or "detr" in value:
            return "detrpose"
        return "unknown"

    def _build_model_metadata(self, args):
        config_file = getattr(args, "config_file", None)
        config_module = self._normalize_config_module(config_file)
        model_name = Path(str(config_file)).stem if config_file else None
        size_variant = None
        if model_name and "_hgnetv2_" in model_name:
            suffix = model_name.split("_hgnetv2_", 1)[1]
            size_variant = suffix.split("_", 1)[0]

        metadata = {
            "family": self._infer_model_family(config_file),
            "config_file": config_file,
            "config_module": config_module,
            "model_name": model_name,
            "model_size": size_variant or model_name,
            "num_classes": self.cfg.get("NUM_CLASSES", None),
            "num_body_points": self.cfg.get("NUM_BODY_POINTS", None),
            "class_mappings": self.cfg.get("CLASS_MAPPINGS", {}),
            "skeleton_connections": self.cfg.get("CLASS_SKELETONS", {}),
            "contiguous_to_category_id": self.cfg.get("CONTIGUOUS_TO_CATEGORY_ID", {}),
            "detection_only": bool(self.cfg.get("DETECTION_ONLY", False)),
            "backbone_name": OmegaConf.select(self.cfg, "model.backbone.name", default=None),
            "backbone_use_lab": OmegaConf.select(self.cfg, "model.backbone.use_lab", default=None),
            "image_size": OmegaConf.select(self.cfg, "model.encoder.eval_spatial_size.0", default=None)
            or OmegaConf.select(self.cfg, "dataset_val.collate_fn.base_size", default=None),
            "feat_strides": OmegaConf.select(self.cfg, "model.feat_strides", default=None)
            or OmegaConf.select(self.cfg, "model.transformer.feat_strides", default=None),
            "neck_out_channels": OmegaConf.select(self.cfg, "model.neck_out_channels", default=None),
            "neck_depth_mult": OmegaConf.select(self.cfg, "model.neck_depth_mult", default=None),
            "encoder_in_channels": OmegaConf.select(self.cfg, "model.encoder.in_channels", default=None),
            "encoder_depth_mult": OmegaConf.select(self.cfg, "model.encoder.depth_mult", default=None),
            "encoder_expansion": OmegaConf.select(self.cfg, "model.encoder.expansion", default=None),
            "num_decoder_layers": OmegaConf.select(self.cfg, "model.transformer.num_decoder_layers", default=None),
        }
        return metadata

    def _build_checkpoint_payload(self, args, epoch):
        return {
            'model': self.model_without_ddp.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'warmup_scheduler': self.warmup_scheduler.state_dict() if self.warmup_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
            'class_mappings': self.cfg.get('CLASS_MAPPINGS', {}),
            'skeleton_connections': self.cfg.get('CLASS_SKELETONS', {}),
            'contiguous_to_category_id': self.cfg.get('CONTIGUOUS_TO_CATEGORY_ID', {}),
            'model_metadata': self._build_model_metadata(args),
        }

    def _setup(self,):
        """Avoid instantiating unnecessary classes"""
        dist_utils.init_distributed_mode(self.cfg.training_params)
        args = self.cfg.training_params

        # fix the seed for reproducibility
        seed = args.seed + dist_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_without_ddp = instantiate(self.cfg.model).to(self.device)
        
        # Suppress torch.compile warnings and verbose output
        import warnings
        warnings.filterwarnings('ignore', message='.*skipping cudagraphs.*')
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
        
        # Suppress torch._dynamo verbose output
        import logging
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        logging.getLogger("torch._inductor").setLevel(logging.ERROR)
        
        # Suppress CUDA graphs warnings
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        
        compile_model = bool(getattr(args, 'compile_model', False))
        if compile_model:
            print(
                "[setup] torch.compile is enabled; the first training forward can take several minutes.",
                flush=True,
            )

        self.model = dist_utils.warp_model(
            self.model_without_ddp,  # Already on device, no need to transfer again
            sync_bn=args.sync_bn, 
            find_unused_parameters=args.find_unused_params,
            compile=compile_model,
            compile_mode=getattr(args, 'compile_mode', 'reduce-overhead'),
            )

        self.postprocessor = instantiate(self.cfg.postprocessor)
        self._link_rtmo_dcc(self.postprocessor)
        self.evaluator = instantiate(self.cfg.evaluator)

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.cfg[name])
        global_cfg = self.cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        num_gpus = dist_utils.get_world_size()
        print(f'building {name} with batch_size={bs} and {num_gpus} GPUs ...')
        dataloader = self.cfg[name]
        dataloader.batch_size = bs
        loader = instantiate(dataloader)
        loader.shuffle = dataloader.get('shuffle', False)
        return loader

    def evaluation(self, ):
        self._setup()
        args = self.cfg.training_params
        self.args = args

        if self.cfg.training_params.use_ema:
            self.cfg.ema.model = self.model_without_ddp
            self.ema = instantiate(self.cfg.ema)
        else:
            self.ema = None

        # Load datasets
        if args.eval:
            dataset_val = self.build_dataloader('dataset_val')
        else:
            dataset_val = self.build_dataloader('dataset_test')

        self.dataloader_val = dist_utils.warp_loader(dataset_val, self.cfg.dataset_val.shuffle)

        if hasattr(args, 'resume'):
            self.resume()
        else:
            raise "Use resume during evaluation"

    def train(self,):
        self._setup()
        args = self.cfg.training_params
        self.args = args

        self.writer = SummaryWriter(self.output_dir/"summary")
        atexit.register(self.writer.close)

        if dist_utils.is_main_process():
            self.writer.add_text("config", "{:s}".format(OmegaConf.to_yaml(self.cfg).__repr__()), 0)
        
        if self.cfg.training_params.use_ema:
            self.cfg.ema.model = self.model_without_ddp
            self.ema = instantiate(self.cfg.ema)
        else:
            self.ema = None

        self.criterion = instantiate(self.cfg.criterion).to(self.device)
        self._link_rtmo_dcc(self.criterion, self.postprocessor)

        self.cfg.optimizer.params.model = self.model_without_ddp
        self.optimizer = instantiate(self.cfg.optimizer)

        self.cfg.lr_scheduler.optimizer = self.optimizer
        self.lr_scheduler = instantiate(self.cfg.lr_scheduler)

        if hasattr(self.cfg, 'warmup_scheduler'):
            self.cfg.warmup_scheduler.lr_scheduler = self.lr_scheduler
            self.warmup_scheduler = instantiate(self.cfg.warmup_scheduler)
        else:
            self.warmup_scheduler = None

        # Load datasets
        dataset_train = self.build_dataloader('dataset_train')
        dataset_val = self.build_dataloader('dataset_val')

        self.dataloader_train = dist_utils.warp_loader(dataset_train, self.cfg.dataset_train.shuffle)
        self.dataloader_val = dist_utils.warp_loader(dataset_val, self.cfg.dataset_val.shuffle)

        assert not (hasattr(args, 'resume') and hasattr(args, 'pretrain')) #'You cant resume and pretain at the same time. Choose one.' 
        if hasattr(args, 'resume'):
            self.resume()

        if hasattr(args, 'pretrain'):
            self.pretrain(args.pretrain)

        self.best_map_holder = BestMetricHolder(use_ema=args.use_ema)

    def fit(self,):
        self.train()
        args = self.args
        model_stats = stats(self.model_without_ddp)
        print(model_stats)
        
        print("-" * 42 + "Start training" + "-" * 43)
        
        if hasattr(args, 'resume'):
            module = self.ema.module if self.ema is not None else self.model
            test_stats = evaluate(
                module, 
                self.postprocessor, 
                self.evaluator,
                self.dataloader_val, 
                self.device, 
                self.writer
            )

            map_regular = self._primary_map_value(test_stats)
            _isbest = self.best_map_holder.update(map_regular, args.start_epoch-1, is_ema=False)

        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()

            self.dataloader_train.set_epoch(epoch)
            # self.dataloader_train.dataset.set_epoch(epoch)
            if dist_utils.is_dist_avail_and_initialized():
                self.dataloader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.dataloader_train, 
                self.optimizer, 
                self.cfg.dataset_train.batch_size,
                args.grad_accum_steps,
                self.device, 
                epoch,
                args.clip_max_norm, 
                lr_scheduler=self.lr_scheduler, 
                warmup_scheduler=self.warmup_scheduler, 
                writer=self.writer, 
                args=args,
                ema=self.ema
                )

            if self.warmup_scheduler is None or self.warmup_scheduler.finished():
                self.lr_scheduler.step()

            save_numbered = False
            if self.output_dir:
                # Always save the latest checkpoint (overwrites each epoch)
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']

                # Save numbered checkpoint at intervals
                save_numbered = (epoch + 1) % args.save_checkpoint_interval == 0
                if save_numbered:
                    numbered_checkpoint = self.output_dir / f'checkpoint{epoch+1:04}.pth'  # Use 1-indexed naming
                    checkpoint_paths.append(numbered_checkpoint)

                # Save all checkpoints
                for checkpoint_path in checkpoint_paths:
                    weights = self._build_checkpoint_payload(args, epoch)
                    dist_utils.save_on_master(weights, checkpoint_path)

                # Clean up old numbered checkpoints (keep only the most recent one)
                if save_numbered and dist_utils.is_main_process():
                    import glob
                    numbered_ckpts = sorted(glob.glob(str(self.output_dir / 'checkpoint[0-9]*.pth')))
                    # Keep only the most recent numbered checkpoint
                    for old_ckpt in numbered_ckpts[:-1]:
                        try:
                            os.remove(old_ckpt)
                            print(f'[Cleanup] Removed old checkpoint: {Path(old_ckpt).name}')
                        except Exception as e:
                            print(f'[Cleanup] Warning: Could not remove {old_ckpt}: {e}')

            self._backup_output_to_gdrive(epoch)

            module = self.ema.module if self.ema is not None else self.model

            eval_interval = getattr(args, 'eval_interval', 1)
            should_eval = ((epoch + 1) % eval_interval == 0) or (epoch == args.epochs - 1)

            if should_eval:
                # eval
                test_stats = evaluate(
                    module, 
                    self.postprocessor, 
                    self.evaluator,
                    self.dataloader_val, 
                    self.device, 
                    self.writer
                )

                if self.writer is not None and dist_utils.is_main_process():
                    coco_stats = test_stats[self._primary_eval_stats_key()]
                    coco_names = self._primary_eval_names()
                    for k, val in zip(coco_names, coco_stats):
                        self.writer.add_scalar(f"Test/{k}", val, epoch)

                map_regular = self._primary_map_value(test_stats)
                _isbest = self.best_map_holder.update(map_regular, epoch, is_ema=False)

                if _isbest:
                    print(f"New best achieved @ epoch {epoch+1:04d}!!!...")
                    checkpoint_path = self.output_dir / 'checkpoint_best_regular.pth'
                    weights = self._build_checkpoint_payload(args, epoch)
                    dist_utils.save_on_master(weights, checkpoint_path)
            else:
                test_stats = {}

            log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': model_stats['params']
                }

            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if should_eval and self.evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    primary_eval_type = self._primary_eval_type()
                    if primary_eval_type in self.evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch+1:03}.pth')  # Use 1-indexed naming
                        for name in filenames:
                            torch.save(self.evaluator.coco_eval[primary_eval_type].eval,
                                       self.output_dir / "eval" / name)
        self.writer.close()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))    

    def eval(self, ):
        self.evaluation()
        module = self.ema.module if self.ema is not None else self.model

        # eval
        test_stats = evaluate(
            module, 
            self.postprocessor, 
            self.evaluator,
            self.dataloader_val, 
            self.device, 
        )

    def test(self, ):
        self.evaluation()
        module = self.ema.module if self.ema is not None else self.model

        # eval
        res_json = evaluate(
            module, 
            self.postprocessor, 
            None, #self.evaluator,
            self.dataloader_val, 
            self.device, 
            save_results=True
        )

        print("Saving results in results.json ...")
        with open("results.json", "w") as final:
            json.dump(res_json, final)
        print("Done ...")

    def resume(self,):
        args = self.cfg.training_params
        if hasattr(args, "resume") and len(args.resume)>0:
            print(f"Loading weights from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            
            # Clean state dict: remove module. and _orig_mod. prefixes
            model_state = checkpoint['model']
            # Remove DataParallel/DDP prefix
            model_state = {
                k.replace("module.", "", 1): v
                for k, v in model_state.items()
            }
            # Remove torch.compile() prefix
            model_state = {
                k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                for k, v in model_state.items()
            }
            
            self.model_without_ddp.load_state_dict(model_state, strict=True)
            if self.ema:
                if 'ema' in checkpoint:
                    ema_state = checkpoint['ema']
                    # Clean EMA state dict if it contains 'module' key
                    if 'module' in ema_state:
                        ema_module_state = ema_state['module']
                        # Remove DataParallel/DDP prefix
                        ema_module_state = {
                            k.replace("module.", "", 1): v
                            for k, v in ema_module_state.items()
                        }
                        # Remove torch.compile() prefix
                        ema_module_state = {
                            k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                            for k, v in ema_module_state.items()
                        }
                        ema_state['module'] = ema_module_state
                    self.ema.load_state_dict(ema_state, strict=False)
                else:
                    self.ema.module.load_state_dict(model_state, strict=False)

            if not(args.eval or args.test) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                import copy
                p_groups = copy.deepcopy(self.optimizer.param_groups)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(self.optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
                if self.warmup_scheduler:
                    self.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler'])

                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                args.override_resumed_lr_drop = True
                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    self.lr_scheduler.milestones = Counter(self.cfg.lr_scheduler.milestones)
                    self.lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))
                self.lr_scheduler.step(self.lr_scheduler.last_epoch)
                args.start_epoch = checkpoint['epoch'] + 1
        else:
            print("Initializing the model with random parameters!")


    def pretrain(self, model_name):
        arch_configs = {
            # COCO
            'dfine_n_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth',
            'dfine_s_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth',
            'dfine_m_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth',
            'dfine_l_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth',
            'dfine_x_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth',
            # OBJECT 365
            'dfine_s_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj365.pth',
            'dfine_m_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth',
            'dfine_l_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365.pth',
            'dfine_x_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj365.pth',
            # OBJECT 365 + COCO
            'dfine_s_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth',
            'dfine_m_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth',
            'dfine_l_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco.pth',
            'dfine_x_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth',
        }
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

        local_model_dir = './weights/dfine/'

        try:
            # If model_name is a direct file path, load it without downloading
            if os.path.isfile(model_name):
                if safe_get_rank() == 0:
                    print(f"Loading pretrained weights from file: {model_name}")
                safe_barrier()
                state = torch.load(model_name, map_location="cpu", weights_only=False)
            else:
                if model_name not in arch_configs:
                    raise KeyError(
                        f"'{model_name}' is not a valid model name and is not an existing file path. "
                        f"Valid names: {list(arch_configs.keys())}"
                    )
                download_url = arch_configs[model_name]

                # If the file doesn't exist locally, download from the URL
                if safe_get_rank() == 0:
                    print(
                        GREEN
                        + "If the pretrained D-FINE can't be downloaded automatically. Please check your network connection."
                        + RESET
                    )
                    print(
                        GREEN
                        + "Please check your network connection. Or download the model manually from "
                        + RESET
                        + f"{download_url}"
                        + GREEN
                        + " to "
                        + RESET
                        + f"{local_model_dir}."
                        + RESET
                    )
                    state = torch.hub.load_state_dict_from_url(
                        download_url, map_location="cpu", model_dir=local_model_dir
                    )
                    print(f"Loaded pretrained DFINE from URL.")

                # Wait for rank 0 to download the model
                safe_barrier()

                # All processes load the downloaded model
                model_path = local_model_dir + model_name + ".pth"
                state = torch.load(model_path, map_location="cpu", weights_only=False)

            if "ema" in state:
                print("USING EMA WEIGHTS!!!")
                pretrain_state_dict = state["ema"]["module"]
            else:
                pretrain_state_dict = state["model"]
            
            new_state_dict = {}
            for k in pretrain_state_dict:
                if ("decoder" in k):
                    continue
                new_state_dict[k] = pretrain_state_dict[k]

            print(f"⚠️  Loading weights for the backbone and decoder from {model_name} ⚠️")
            missing_keys, unexpected_keys = self.model_without_ddp.load_state_dict(new_state_dict, strict=False)

            if len(unexpected_keys) > 0:
                print("Warning. The following RTGroupPose does not have the following parameters:")
                for k in unexpected_keys:
                    print(f"    - {k}")
            else:
                print(f'✅ Successfully initilized the backbone and encoder using {model_name} weights ✅')

            missing_keys, unexpected_keys = self.ema.module.load_state_dict(new_state_dict, strict=False)

        except (Exception, KeyboardInterrupt) as e:
            if safe_get_rank() == 0:
                print(f"{str(e)}")
                logging.error(
                    RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET
                )
                logging.error(
                    GREEN
                    + "Please check your network connection. Or download the model manually from "
                    + RESET
                    + f"{download_url}"
                    + GREEN
                    + " to "
                    + RESET
                    + f"{local_model_dir}."
                    + RESET
                )
            exit()
