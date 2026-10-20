from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import copy
import csv
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:
    torchvision_nms = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except Exception:
    plt = None
    patches = None

@dataclass
class Config:
  
    data_root: str      = './public'
    train_ann: str      = 'annotations/train.json'
    val_ann: str        = 'annotations/val.json'
    train_img: str      = 'train/images'
    val_img: str        = 'val/images'

    class_names: List[str] = field(default_factory=lambda: [
        'person', 'car', 'dog', 'cat', 'chair'
    ])
    num_classes: int    = 5

    
    backbone_name: str  = 'resnet101'  
    pretrained_backbone: bool = True
    freeze_backbone_stem: bool = True
    backbone_compress_ratio: float = 0.50
    img_size: int       = 672
    depth_mult: float   = 0.60
    width_mult: float   = 0.85
    reg_max: int        = 16
    strides: List[int]  = field(default_factory=lambda: [8, 16, 32])
    feature_dim: int    = 16

    epochs: int         = 24
    lr_schedule_epochs: int = 24
    batch_size: int     = 12
    val_batch_size: int = 12
    accumulate: int     = 2
    lr0: float          = 0.005
    backbone_lr_mult: float = 0.8
    backbone_freeze_epochs: int = 3
    backbone_ramp_epochs: int = 9
    backbone_frozen_lr_scale: float = 0.0
    freeze_pretrained_bn: bool = True
    lrf: float          = 0.01
    momentum: float     = 0.937
    weight_decay: float = 1.2e-3
    warmup_epochs: int  = 3
    num_workers: int    = 2
    use_adamw: bool     = False
    amp: bool           = True
    channels_last: bool = True
    eval_interval: int  = 3
    full_eval_start_epoch: int = 12
    patience_evals: int = 14
    overfit_guard: bool = True
    overfit_min_epoch: int = 30
    overfit_patience_evals: int = 4
    overfit_val_margin: float = 0.08
    grad_clip: float    = 10.0

    resume_ckpt: str    = ''
    resume_optimizer: bool = False

    box_gain: float     = 9.5
    cls_gain: float     = 1.3
    dfl_gain: float     = 1.80
    smooth_l1_gain: float = 0.30
    smooth_l1_beta: float = 0.05
    feature_gain: float = 0.05
    feature_temperature: float = 0.20
    feature_max_pos: int = 256
    assigner_topk: int  = 16
    target_score_floor: float = 0.05
    
    cls_focal_gamma: float = 0.0
    cls_focal_alpha: float = -1.0
    cls_loss_type: str = 'bce'
    positive_weight_only: bool = True
    normalizer_mode: str = 'positive'

    hard_negative_mining: bool = True
    hard_negative_conf: float = 0.03
    hard_negative_topk: int = 2048
    hard_negative_gain: float = 0.28
    hard_negative_start_epoch: int = 3
    hard_negative_warmup_epochs: int = 4
    hard_negative_class_weights: List[float] = field(default_factory=lambda: [
        1.10, 1.25, 0.6, 0.6, 3.3
    ])
    rank_bce_gain: float = 0.04
    rank_bce_classes: List[int] = field(default_factory=lambda: [1]) 
    rank_bce_margin: float = 0.15
    rank_bce_pos_topk: int = 96
    rank_bce_neg_topk: int = 256
    rank_bce_min_pos_quality: float = 0.20
    rank_bce_start_epoch: int = 4
    rank_bce_warmup_epochs: int = 4

    class_balance_power: float = 0.25
    class_weight_clip: Tuple[float, float] = (0.75, 2.00)
    manual_class_weight_multipliers: List[float] = field(default_factory=lambda: [
        1.08, 1.15, 1.03, 1.03, 1.75
    ])

    use_weighted_sampler: bool = True
    negative_image_weight: float = 1.60
    image_sample_class_multipliers: List[float] = field(default_factory=lambda: [
        1.12, 1.16, 1.03, 1.03, 2.35
    ])
    small_object_sample_boost: float = 0.10
    sample_weight_clip: Tuple[float, float] = (0.45, 3.50)


    box_class_weight_multipliers: List[float] = field(default_factory=lambda: [
        1.08, 1.15, 1.00, 1.00, 2.40
    ])
    
    focus_class_id: int = 4 
    focus_class_ids: List[int] = field(default_factory=lambda: [4])  # chair
    focus_class_weights: List[float] = field(default_factory=lambda: [1.0])
    focus_crop_prob: float = 0.35
    focus_crop_context_range: Tuple[float, float] = (1.4, 2.6)
    focus_crop_min_side_frac: float = 0.3
    focus_crop_min_visible: float = 0.3

    mosaic_prob: float  = 0.95
    mosaic_mid_prob: float = 0.85
    mosaic_mid_epoch_frac: float = 0.65
    mixup_prob: float   = 0.02
    close_mosaic_epochs: int = 8
    hsv_h: float        = 0.012
    hsv_s: float        = 0.50
    hsv_v: float        = 0.32
    flip_lr: float      = 0.5
    random_translate: bool = True

    conf_thres: float   = 0.001

    infer_conf_thres: float = 0.25

    class_conf_thres: List[float] = field(default_factory=lambda: [
        0.32, 0.25, 0.25, 0.25, 0.14
    ])

    class_iou_thres: List[float] = field(default_factory=lambda: [
        0.40, 0.40, 0.40, 0.40, 0.32
    ])

    eval_conf_thres: float = 0.001
    eval_use_class_conf_thres: bool = False

   
    export_conf_thres: float = 0.001
    export_use_class_conf_thres: bool = False

    iou_thres: float    = 0.50
    max_det: int        = 100
    nms_pre_topk: int   = 6000
    use_tta_eval: bool  = False

    
    cm_iou_thres: float = 0.50
    cm_conf_thres: float = 0.22
    cm_use_class_conf_thres: bool = True
    save_confusion_matrix: bool = True

    save_top_k: int     = 0
    run_post_sweep: bool = False
    ensemble_top_k: bool = False
    best_metric: str = 'map50'

    device: str         = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str       = './runs'
    models_dir: str     = './models'
    seed: int           = 42

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def load_annotation_json(json_path: str, image_root: str, class_names):
    """Load assignment JSON and keep one record per image, including negative images."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    classes_in_file = data.get('classes', class_names)
    if list(classes_in_file) != list(class_names):
        print('WARNING: class order in JSON differs from cfg.class_names')
        print('JSON:', classes_in_file)
        print('CFG :', class_names)

    class_to_id = {name: i for i, name in enumerate(class_names)}
    anns_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        cls_name = ann['class']
        if cls_name not in class_to_id:
            continue
        x1, y1, x2, y2 = map(float, ann['bbox'])
        anns_by_img[ann['image_id']].append([class_to_id[cls_name], x1, y1, x2, y2])

    records = []
    for im in data.get('images', []):
        image_id = im['id']
        file_name = im.get('file_name', image_id)
        # The JSON file_name is usually train/images/xxx.jpg or val/images/xxx.jpg.
        # image_root already points to that split, so use image_id for robustness.
        img_path = os.path.join(image_root, image_id)
        if not os.path.exists(img_path):
            img_path = os.path.join(os.path.dirname(image_root), file_name)
        boxes = np.array(anns_by_img.get(image_id, []), dtype=np.float32).reshape(-1, 5)
        records.append({
            'image_id': image_id,
            'file_name': file_name,
            'path': img_path,
            'width': int(im['width']),
            'height': int(im['height']),
            'labels_xyxy': boxes,  # [cls, xmin, ymin, xmax, ymax] in original pixels
        })
    return records

def xyxy_to_yolo_norm(labels_xyxy, w, h):
    """[cls,x1,y1,x2,y2] absolute -> [cls,cx,cy,bw,bh] normalized."""
    labels = np.asarray(labels_xyxy, dtype=np.float32).reshape(-1, 5)
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    out = np.zeros_like(labels, dtype=np.float32)
    out[:, 0] = labels[:, 0]
    x1 = np.clip(labels[:, 1], 0, w)
    y1 = np.clip(labels[:, 2], 0, h)
    x2 = np.clip(labels[:, 3], 0, w)
    y2 = np.clip(labels[:, 4], 0, h)
    bw = np.maximum(x2 - x1, 0)
    bh = np.maximum(y2 - y1, 0)
    keep = (bw >= 1) & (bh >= 1)
    if not keep.any():
        return np.zeros((0, 5), dtype=np.float32)
    out = out[keep]
    x1, y1, x2, y2, bw, bh = x1[keep], y1[keep], x2[keep], y2[keep], bw[keep], bh[keep]
    out[:, 1] = (x1 + x2) / 2 / w
    out[:, 2] = (y1 + y2) / 2 / h
    out[:, 3] = bw / w
    out[:, 4] = bh / h
    out[:, 1:] = np.clip(out[:, 1:], 0, 1)
    return out.astype(np.float32)

def clean_yolo_labels(labels, num_classes: int):
    """Clean YOLO labels [class, cx, cy, w, h] normalized."""
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 5)
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    finite = np.isfinite(labels).all(axis=1)
    labels = labels[finite]
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cls_raw = labels[:, 0]
    cls_round = np.rint(cls_raw).astype(np.int64)
    valid_cls = (np.abs(cls_raw - cls_round) < 1e-3) & (cls_round >= 0) & (cls_round < num_classes)
    labels = labels[valid_cls]
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cls = cls_round[valid_cls].astype(np.float32)
    cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    x1 = np.clip(cx - w / 2, 0, 1)
    y1 = np.clip(cy - h / 2, 0, 1)
    x2 = np.clip(cx + w / 2, 0, 1)
    y2 = np.clip(cy + h / 2, 0, 1)
    w = x2 - x1
    h = y2 - y1
    keep = (w > 1e-4) & (h > 1e-4)
    cleaned = np.stack([cls, (x1+x2)/2, (y1+y2)/2, w, h], axis=1).astype(np.float32)
    return cleaned[keep]

def yolo_norm_to_xyxy(labels, w, h):
    labels = clean_yolo_labels(labels, cfg.num_classes)
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cx = labels[:, 1] * w
    cy = labels[:, 2] * h
    bw = labels[:, 3] * w
    bh = labels[:, 4] * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return np.stack([labels[:, 0], x1, y1, x2, y2], axis=1).astype(np.float32)

def letterbox_with_labels(img, labels_yolo, img_size: int, random_offset=False):
    """Resize with aspect ratio + pad. Returns square image and normalized square labels."""
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    if random_offset:
        max_dx = img_size - nw
        max_dy = img_size - nh
        dx = random.randint(0, max_dx) if max_dx > 0 else 0
        dy = random.randint(0, max_dy) if max_dy > 0 else 0
    else:
        dx = (img_size - nw) // 2
        dy = (img_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized

    labels_xyxy = yolo_norm_to_xyxy(labels_yolo, w, h)
    if len(labels_xyxy):
        labels_xyxy[:, [1, 3]] = labels_xyxy[:, [1, 3]] * scale + dx
        labels_xyxy[:, [2, 4]] = labels_xyxy[:, [2, 4]] * scale + dy
        labels_xyxy[:, [1, 3]] = np.clip(labels_xyxy[:, [1, 3]], 0, img_size)
        labels_xyxy[:, [2, 4]] = np.clip(labels_xyxy[:, [2, 4]], 0, img_size)
        labels_out = xyxy_to_yolo_norm(labels_xyxy, img_size, img_size)
    else:
        labels_out = np.zeros((0, 5), dtype=np.float32)
    return canvas, labels_out

def random_hsv(img, hgain=0.012, sgain=0.5, vgain=0.32):
    if random.random() > 0.35:
        return img
    r = np.random.uniform(-1, 1, 3) * np.array([hgain * 180, sgain, vgain])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + r[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + r[1]), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + r[2]), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def horizontal_flip(img, labels_yolo):
    if len(labels_yolo):
        labels_yolo = labels_yolo.copy()
        labels_yolo[:, 1] = 1.0 - labels_yolo[:, 1]
    return np.ascontiguousarray(img[:, ::-1]), labels_yolo

def focus_crop_with_labels(img, labels_yolo, img_size: int, class_id: int, cfg: Config):
    """Crop around a target class before letterbox to improve small/occluded chair recall."""
    if len(labels_yolo) == 0:
        return letterbox_with_labels(img, labels_yolo, img_size, random_offset=True)

    h, w = img.shape[:2]
    boxes = yolo_norm_to_xyxy(labels_yolo, w, h)
    focus_ids = getattr(cfg, 'focus_class_ids', None)
    if focus_ids is None:
        focus_ids = [class_id]
    focus_ids = [int(c) for c in focus_ids]
    focus_weight_values = getattr(cfg, 'focus_class_weights', None)
    focus_weight_by_class = {cid: 1.0 for cid in focus_ids}
    if focus_weight_values is not None:
        for cid, weight in zip(focus_ids, focus_weight_values):
            focus_weight_by_class[int(cid)] = max(float(weight), 1e-3)
    label_classes = boxes[:, 0].astype(np.int64)
    candidates = np.where(np.isin(label_classes, np.asarray(focus_ids, dtype=np.int64)))[0]
    if len(candidates) == 0:
        return letterbox_with_labels(img, labels_yolo, img_size, random_offset=True)

    cand_boxes = boxes[candidates]
    cand_area = np.maximum((cand_boxes[:, 3] - cand_boxes[:, 1]) * (cand_boxes[:, 4] - cand_boxes[:, 2]), 1.0)
    class_pick_weights = np.array([
        focus_weight_by_class.get(int(c), 1.0) for c in cand_boxes[:, 0].astype(np.int64)
    ], dtype=np.float32)
    pick_weights = class_pick_weights / np.sqrt(cand_area)
    pick_weights = pick_weights / pick_weights.sum()
    target = boxes[int(np.random.choice(candidates, p=pick_weights))]

    tx1, ty1, tx2, ty2 = target[1:5]
    tw = max(tx2 - tx1, 2.0)
    th = max(ty2 - ty1, 2.0)
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5

    lo_ctx, hi_ctx = getattr(cfg, 'focus_crop_context_range', (1.8, 3.2))
    crop_side = max(tw, th) * random.uniform(float(lo_ctx), float(hi_ctx))
    min_side = min(h, w) * float(getattr(cfg, 'focus_crop_min_side_frac', 0.35))
    crop_side = int(round(np.clip(max(crop_side, min_side), max(tw, th) + 4.0, max(h, w))))
    crop_w = min(crop_side, w)
    crop_h = min(crop_side, h)

    slack_x = max(crop_w - tw, 0.0)
    slack_y = max(crop_h - th, 0.0)
    cx = tcx + random.uniform(-0.25, 0.25) * slack_x
    cy = tcy + random.uniform(-0.25, 0.25) * slack_y
    x1 = int(round(np.clip(cx - crop_w * 0.5, 0, max(w - crop_w, 0))))
    y1 = int(round(np.clip(cy - crop_h * 0.5, 0, max(h - crop_h, 0))))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return letterbox_with_labels(img, labels_yolo, img_size, random_offset=True)

    shifted = boxes.copy()
    orig_area = np.maximum((shifted[:, 3] - shifted[:, 1]) * (shifted[:, 4] - shifted[:, 2]), 1.0)
    shifted[:, [1, 3]] -= x1
    shifted[:, [2, 4]] -= y1
    shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], 0, crop_w)
    shifted[:, [2, 4]] = np.clip(shifted[:, [2, 4]], 0, crop_h)
    new_w = shifted[:, 3] - shifted[:, 1]
    new_h = shifted[:, 4] - shifted[:, 2]
    visible = (new_w * new_h) / orig_area
    min_visible = float(getattr(cfg, 'focus_crop_min_visible', 0.35))
    keep = (new_w >= 2.0) & (new_h >= 2.0) & (visible >= min_visible)
    if not np.any(keep):
        return letterbox_with_labels(img, labels_yolo, img_size, random_offset=True)

    crop_labels = xyxy_to_yolo_norm(shifted[keep], crop_w, crop_h)
    return letterbox_with_labels(crop, crop_labels, img_size, random_offset=False)

def mosaic4(images, labels_list, img_size: int):
    """4-image mosaic with random center and correct xyxy clipping after crop."""
    s = img_size
    yc = int(random.uniform(0.50 * s, 1.50 * s))
    xc = int(random.uniform(0.50 * s, 1.50 * s))
    canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
    out_labels = []

    for i, (img, labels) in enumerate(zip(images, labels_list)):
        h, w = img.shape[:2]
        scale = min(s / h, s / w)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if i == 0:
            x1, y1 = xc - nw, yc - nh
        elif i == 1:
            x1, y1 = xc, yc - nh
        elif i == 2:
            x1, y1 = xc - nw, yc
        else:
            x1, y1 = xc, yc

        x2, y2 = x1 + nw, y1 + nh
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(2 * s, x2), min(2 * s, y2)
        if x2c > x1c and y2c > y1c:
            canvas[y1c:y2c, x1c:x2c] = img_resized[y1c-y1:y2c-y1, x1c-x1:x2c-x1]

        if len(labels):
            cx = labels[:, 1] * nw + x1
            cy = labels[:, 2] * nh + y1
            bw = labels[:, 3] * nw
            bh = labels[:, 4] * nh
            x1b = cx - bw / 2
            y1b = cy - bh / 2
            x2b = cx + bw / 2
            y2b = cy + bh / 2
            out_labels.append(np.stack([labels[:, 0], x1b, y1b, x2b, y2b], axis=1))

    crop_x, crop_y = s // 2, s // 2
    canvas = canvas[crop_y:crop_y + s, crop_x:crop_x + s]

    if not out_labels:
        return canvas, np.zeros((0, 5), dtype=np.float32)

    boxes = np.concatenate(out_labels, axis=0)
    boxes[:, [1, 3]] -= crop_x
    boxes[:, [2, 4]] -= crop_y
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, s)
    boxes[:, [2, 4]] = np.clip(boxes[:, [2, 4]], 0, s)
    return canvas, xyxy_to_yolo_norm(boxes, s, s)

def mixup(im1, labels1, im2, labels2):
    r = np.random.beta(32.0, 32.0)
    im = (im1 * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels1, labels2), 0) if len(labels1) and len(labels2) else (labels1 if len(labels1) else labels2)
    return im, labels

class PublicDetectionDataset(Dataset):
    """Assignment dataset reader. Keeps negative images and outputs YOLO-normalized targets."""
    def __init__(self, ann_path: str, image_dir: str, img_size: int = 640,
                 augment: bool = True, mosaic_prob: float = 0.0):
        self.ann_path = ann_path
        self.image_dir = image_dir
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.records = load_annotation_json(ann_path, image_dir, cfg.class_names)
        assert len(self.records) > 0, f'No images found in {ann_path}'

    def __len__(self):
        return len(self.records)

    def close_mosaic(self):
        self.mosaic_prob = 0.0

    def _load_raw(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec['path'])
        if img is None:
            raise FileNotFoundError(rec['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        labels = xyxy_to_yolo_norm(rec['labels_xyxy'], w, h)
        return img, labels

    def _has_focus_class(self, idx):
        labels = self.records[idx]['labels_xyxy']
        focus_ids = getattr(cfg, 'focus_class_ids', None)
        if focus_ids is None:
            focus_ids = [getattr(cfg, 'focus_class_id', 4)]
        focus_ids = np.asarray([int(c) for c in focus_ids], dtype=np.int64)
        return len(labels) > 0 and np.any(np.isin(labels[:, 0].astype(np.int64), focus_ids))

    def __getitem__(self, idx):
        use_focus_crop = (
            self.augment
            and random.random() < float(getattr(cfg, 'focus_crop_prob', 0.0))
            and self._has_focus_class(idx)
        )
        if use_focus_crop:
            img, labels = self._load_raw(idx)
            img, labels = focus_crop_with_labels(
                img, labels, self.img_size,
                class_id=int(getattr(cfg, 'focus_class_id', 4)),
                cfg=cfg
            )
        elif self.augment and random.random() < self.mosaic_prob:
            indices = [idx] + random.choices(range(len(self)), k=3)
            imgs, lbls = zip(*[self._load_raw(i) for i in indices])
            img, labels = mosaic4(list(imgs), list(lbls), self.img_size)
            if random.random() < getattr(cfg, 'mixup_prob', 0.0):
                mix_indices = random.choices(range(len(self)), k=4)
                mix_imgs, mix_lbls = zip(*[self._load_raw(i) for i in mix_indices])
                mix_img, mix_labels = mosaic4(list(mix_imgs), list(mix_lbls), self.img_size)
                img, labels = mixup(img, labels, mix_img, mix_labels)
        else:
            img, labels = self._load_raw(idx)
            img, labels = letterbox_with_labels(
                img, labels, self.img_size,
                random_offset=(self.augment and getattr(cfg, 'random_translate', True))
            )

        if self.augment:
            img = random_hsv(img, cfg.hsv_h, cfg.hsv_s, cfg.hsv_v)
            if random.random() < cfg.flip_lr:
                img, labels = horizontal_flip(img, labels)

        labels = clean_yolo_labels(labels, cfg.num_classes)
        img_t = torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1)).float() / 255.0

        if len(labels):
            target = torch.zeros(len(labels), 6, dtype=torch.float32)
            target[:, 1] = torch.from_numpy(labels[:, 0]).float()
            target[:, 2:] = torch.from_numpy(labels[:, 1:]).float()
        else:
            target = torch.zeros(0, 6, dtype=torch.float32)
        return img_t, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    labeled = []
    for i, t in enumerate(targets):
        if len(t):
            t = t.clone()
            t[:, 0] = i
            labeled.append(t)
    targets = torch.cat(labeled, 0) if labeled else torch.zeros(0, 6)
    return imgs, targets

def estimate_class_weights_from_records(records, num_classes: int, power: float = 0.25,
                                        clip=(0.75, 1.35), manual_multipliers=None):
    counts = np.zeros(num_classes, dtype=np.float64)
    for rec in records:
        labels = rec['labels_xyxy']
        if len(labels):
            cls = labels[:, 0].astype(np.int64)
            counts += np.bincount(cls, minlength=num_classes)[:num_classes]
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()
    if manual_multipliers is not None:
        manual = np.asarray(manual_multipliers, dtype=np.float64)
        if len(manual) != num_classes:
            raise ValueError('manual_class_weight_multipliers length must match num_classes')
        weights = weights * manual
    weights = np.clip(weights, clip[0], clip[1]).astype(np.float32)
    return counts.astype(np.int64), torch.tensor(weights, dtype=torch.float32)

def estimate_image_sample_weights_from_records(records, cfg: Config):
    multipliers = np.asarray(getattr(cfg, 'image_sample_class_multipliers', [1.0] * cfg.num_classes), dtype=np.float64)
    if len(multipliers) != cfg.num_classes:
        raise ValueError('image_sample_class_multipliers length must match num_classes')

    lo, hi = getattr(cfg, 'sample_weight_clip', (0.45, 3.0))
    weights = np.zeros(len(records), dtype=np.float64)
    for i, rec in enumerate(records):
        labels = rec['labels_xyxy']
        if len(labels) == 0:
            weights[i] = getattr(cfg, 'negative_image_weight', 0.45)
            continue

        cls_ids = labels[:, 0].astype(np.int64)
        weight = float(multipliers[np.unique(cls_ids)].max())
        img_area = max(float(rec['width'] * rec['height']), 1.0)
        box_areas = (labels[:, 3] - labels[:, 1]).clip(min=0) * (labels[:, 4] - labels[:, 2]).clip(min=0)
        if len(box_areas) and float(box_areas.min() / img_area) < 0.02:
            weight += getattr(cfg, 'small_object_sample_boost', 0.35)
        weights[i] = np.clip(weight, lo, hi)

    return torch.as_tensor(weights, dtype=torch.double)

def build_dataloaders(cfg: Config):
    root = cfg.data_root
    train_ds = PublicDetectionDataset(
        ann_path=os.path.join(root, cfg.train_ann),
        image_dir=os.path.join(root, cfg.train_img),
        img_size=cfg.img_size, augment=True, mosaic_prob=cfg.mosaic_prob
    )
    val_ds = PublicDetectionDataset(
        ann_path=os.path.join(root, cfg.val_ann),
        image_dir=os.path.join(root, cfg.val_img),
        img_size=cfg.img_size, augment=False, mosaic_prob=0.0
    )
    sampler = None
    if getattr(cfg, 'use_weighted_sampler', False):
        sample_weights = estimate_image_sample_weights_from_records(train_ds.records, cfg)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        print(
            f'Weighted sampler enabled | min={sample_weights.min():.2f} '
            f'mean={sample_weights.mean():.2f} max={sample_weights.max():.2f}'
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        drop_last=True, persistent_workers=(cfg.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.val_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0)
    )
    print(f'Train: {len(train_ds)} imgs | Val: {len(val_ds)} imgs | batch={cfg.batch_size} | val_batch={cfg.val_batch_size}')
    return train_loader, val_loader

def visualize_dataset_batch(loader, class_names, n=8):
    imgs, targets = next(iter(loader))
    cols = 4
    rows = math.ceil(min(n, len(imgs)) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i in range(min(n, len(imgs))):
        ax = axes[i]
        img = (imgs[i].permute(1,2,0).numpy() * 255).astype(np.uint8)
        ax.imshow(img)
        ax.axis('off')
        gt = targets[targets[:, 0] == i]
        for row in gt:
            cls = int(row[1].item())
            cx, cy, bw, bh = (row[2:] * cfg.img_size).tolist()
            x1, y1 = cx - bw/2, cy - bh/2
            rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, max(0, y1-3), class_names[cls], color='white', fontsize=7,
                    bbox=dict(facecolor='green', alpha=0.7, pad=1, edgecolor='none'))
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.suptitle('Training samples after safe augmentation / mosaic')
    plt.tight_layout()
    plt.show()

def autopad(k, p=None, d=1):
    """Auto-pad to keep output same spatial size. k can be int or tuple."""
    if p is not None:
        return p
    if isinstance(k, (list, tuple)):
        # Each dim padded independently
        return [autopad(x, d=d) for x in k]
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2

class Conv(nn.Module):
    """Standard Conv + BN + SiLU."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        pad = autopad(k, p, d)
        if isinstance(pad, list): pad = tuple(pad)
        self.conv = nn.Conv2d(c_in, c_out, k, s, pad,
                              dilation=d, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck used inside C2f. ScratchCSP uses two 3Ã—3 convs."""
    def __init__(self, c_in, c_out, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_mid = int(c_out * e)
        self.cv1 = Conv(c_in,  c_mid, k[0], 1)
        self.cv2 = Conv(c_mid, c_out, k[1], 1, g=g)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """
    CSP Faster â€” replaces C3 in ScratchCSP.
    Splits channels, runs n bottlenecks on one half, then concat all.
    """
    def __init__(self, c_in, c_out, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_mid = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * c_mid, 1, 1)
        self.cv2 = Conv((2 + n) * c_mid, c_out, 1)
        self.m   = nn.ModuleList(
            Bottleneck(c_mid, c_mid, shortcut, g, k=((3,3),(3,3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))  # split along channels
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (3 sequential 5Ã—5 maxpools)."""
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_mid = c_in // 2
        self.cv1 = Conv(c_in,  c_mid, 1, 1)
        self.cv2 = Conv(c_mid * 4, c_out, 1, 1)
        self.m   = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))

class DFL(nn.Module):
    """
    Distribution Focal Loss head â€” learns to predict box edges as distributions.
    reg_max: number of discrete positions.
    """
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        x = torch.arange(reg_max, dtype=torch.float32)
        self.conv.weight.data[:] = x.view(1, reg_max, 1, 1)
        for p in self.conv.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # x: [B, 4*reg_max, H, W]
        b, _, h, w = x.shape
        x = x.view(b, 4, self.reg_max, h * w)
        x = x.softmax(dim=2)
        x = x.permute(0, 1, 3, 2).contiguous().view(b * 4, self.reg_max, h, w)
        x = self.conv(x)
        return x.view(b, 4, h * w)

class ScratchCSPBackbone(nn.Module):
    """
    Backbone channels (ScratchCSPs): width_mult=0.5
    Base channels: [64, 128, 256, 512, 1024]
    After scale: [32, 64, 128, 256, 512]

    Stages and repeats (depth_mult=0.33):
    C2f repeats base: [3, 6, 6, 3]  â†’ after scale: [1, 2, 2, 1]
    """
    def __init__(self, width: float = 0.5, depth: float = 0.33):
        super().__init__()
        def ch(n): return max(round(n * width), 1)
        def rep(n): return max(round(n * depth), 1)

        # Stem
        self.stem = Conv(3, ch(64), 3, 2)            # /2  320Ã—320

        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(ch(64), ch(128), 3, 2),              # /4  160Ã—160
            C2f(ch(128), ch(128), n=rep(3), shortcut=True)
        )

        # Stage 2  â†’ P3
        self.stage2 = nn.Sequential(
            Conv(ch(128), ch(256), 3, 2),             # /8  80Ã—80
            C2f(ch(256), ch(256), n=rep(6), shortcut=True)
        )

        # Stage 3  â†’ P4
        self.stage3 = nn.Sequential(
            Conv(ch(256), ch(512), 3, 2),             # /16 40Ã—40
            C2f(ch(512), ch(512), n=rep(6), shortcut=True)
        )

        # Stage 4 + SPPF  â†’ P5
        self.stage4 = nn.Sequential(
            Conv(ch(512), ch(1024), 3, 2),            # /32 20Ã—20
            C2f(ch(1024), ch(1024), n=rep(3), shortcut=True),
            SPPF(ch(1024), ch(1024), k=5)
        )

        self.out_channels = [ch(256), ch(512), ch(1024)]  # P3, P4, P5

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # stride 8
        p4 = self.stage3(p3)  # stride 16
        p5 = self.stage4(p4)  # stride 32
        return p3, p4, p5

class FeatureCompressionBlock(nn.Module):
    """Compress ImageNet feature maps before the custom neck."""
    def __init__(self, c_in: int, c_out: int, ratio: float = 0.5):
        super().__init__()
        ratio = max(float(ratio), 0.125)
        c_mid = max(16, int(round(c_out * ratio)))
        c_mid = min(c_mid, c_out)
        self.block = nn.Sequential(
            Conv(c_in, c_mid, 1, 1),
            Conv(c_mid, c_mid, 3, 1, g=c_mid),
            Conv(c_mid, c_out, 1, 1),
        )

    def forward(self, x):
        return self.block(x)

class ImageNetResNetBackbone(nn.Module):
    """ImageNet classification ResNet used only as a feature extractor, not a detector."""
    _native_channels = {
        'resnet18': [128, 256, 512],
        'resnet34': [128, 256, 512],
        'resnet50': [512, 1024, 2048],
        'resnet101': [512, 1024, 2048],
    }
    _weights_name = {
        'resnet18': 'ResNet18_Weights',
        'resnet34': 'ResNet34_Weights',
        'resnet50': 'ResNet50_Weights',
        'resnet101': 'ResNet101_Weights',
    }

    def __init__(self, name='resnet50', pretrained=True, out_width=0.75, freeze_stem=True, compress_ratio=0.5):
        super().__init__()
        name = str(name).lower()
        if name not in self._native_channels:
            raise ValueError(f'Unsupported ImageNet backbone: {name}')
        self.name = name
        self.pretrained = bool(pretrained)
        self.freeze_stem = bool(freeze_stem)
        self.compress_ratio = float(compress_ratio)

        base = self._build_resnet(name, self.pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)  # /4
        self.layer1 = base.layer1  # /4
        self.layer2 = base.layer2  # /8
        self.layer3 = base.layer3  # /16
        self.layer4 = base.layer4  # /32

        def ch(n): return max(round(n * out_width), 1)
        self.out_channels = [ch(256), ch(512), ch(1024)]
        c3, c4, c5 = self._native_channels[name]
        self.proj3 = FeatureCompressionBlock(c3, self.out_channels[0], self.compress_ratio)
        self.proj4 = FeatureCompressionBlock(c4, self.out_channels[1], self.compress_ratio)
        self.proj5 = FeatureCompressionBlock(c5, self.out_channels[2], self.compress_ratio)

        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        if self.freeze_stem:
            for module in (self.stem, self.layer1):
                for p in module.parameters():
                    p.requires_grad_(False)

    @classmethod
    def _build_resnet(cls, name, pretrained):
        try:
            from torchvision import models as tv_models
        except Exception as e:
            raise ImportError('torchvision is required for ImageNet ResNet feature backbones.') from e

        fn = getattr(tv_models, name)
        if not pretrained:
            try:
                return fn(weights=None)
            except TypeError:
                return fn(pretrained=False)

        try:
            weights_cls = getattr(tv_models, cls._weights_name[name])
            return fn(weights=weights_cls.DEFAULT)
        except Exception as e_new:
            try:
                return fn(pretrained=True)
            except Exception as e_old:
                raise RuntimeError(
                    'Could not load ImageNet pretrained ResNet weights. Enable internet on Kaggle '
                    'or add a cached torchvision weight file. Set cfg.pretrained_backbone=False only '
                    'if you intentionally want a random classification backbone.'
                ) from e_old

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_stem:
            self.stem.eval()
            self.layer1.eval()
        return self

    def forward(self, x):
        x = (x - self.imagenet_mean.to(dtype=x.dtype, device=x.device)) / self.imagenet_std.to(dtype=x.dtype, device=x.device)
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return self.proj3(p3), self.proj4(p4), self.proj5(p5)

class ScratchCSPNeck(nn.Module):
    def __init__(self, backbone_channels, depth: float = 0.33):
        super().__init__()
        c3, c4, c5 = backbone_channels
        def rep(n): return max(round(n * depth), 1)

        # â”€â”€ Top-down (FPN) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # P5 â†’ P4 fusion
        self.up_p5   = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4  = C2f(c5 + c4, c4, n=rep(3), shortcut=False)

        # P4 â†’ P3 fusion
        self.up_p4   = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p3  = C2f(c4 + c3, c3, n=rep(3), shortcut=False)

        # â”€â”€ Bottom-up (PAN) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # N3 â†’ N4
        self.down_n3 = Conv(c3, c3, 3, 2)
        self.c2f_n4  = C2f(c3 + c4, c4, n=rep(3), shortcut=False)

        # N4 â†’ N5
        self.down_n4 = Conv(c4, c4, 3, 2)
        self.c2f_n5  = C2f(c4 + c5, c5, n=rep(3), shortcut=False)

        self.out_channels = [c3, c4, c5]  # N3, N4, N5

    def forward(self, features):
        p3, p4, p5 = features

        # Top-down
        td_p4 = self.c2f_p4(torch.cat([self.up_p5(p5), p4], dim=1))
        td_p3 = self.c2f_p3(torch.cat([self.up_p4(td_p4), p3], dim=1))

        # Bottom-up
        n3 = td_p3
        n4 = self.c2f_n4(torch.cat([self.down_n3(n3), td_p4], dim=1))
        n5 = self.c2f_n5(torch.cat([self.down_n4(n4), p5], dim=1))

        return n3, n4, n5   # 80Ã—80, 40Ã—40, 20Ã—20

class DecoupledHead(nn.Module):
    """
    Separate box, cls, and optional feature branches for each scale.
    box branch: 2xConv -> 4*reg_max (DFL raw distributions)
    cls branch: 2xConv -> num_classes (BCE classifier stays unchanged)
    feature branch: small embedding used only by auxiliary feature loss during train
    """
    def __init__(self, in_ch: int, num_classes: int, reg_max: int = 16, feature_dim: int = 0):
        super().__init__()
        inner = max(in_ch, num_classes, reg_max * 4)
        self.feature_dim = int(feature_dim)

        self.box_conv = nn.Sequential(
            Conv(in_ch, inner, 3, 1),
            Conv(inner, inner, 3, 1),
            nn.Conv2d(inner, 4 * reg_max, 1)
        )
        self.cls_conv = nn.Sequential(
            Conv(in_ch, inner, 3, 1),
            Conv(inner, inner, 3, 1),
            nn.Conv2d(inner, num_classes, 1)
        )
        self.feat_conv = None
        if self.feature_dim > 0:
            self.feat_conv = nn.Sequential(
                Conv(in_ch, inner, 3, 1),
                Conv(inner, inner, 3, 1),
                nn.Conv2d(inner, self.feature_dim, 1)
            )
        self._init_weights()

    def _init_weights(self):
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_conv[-1].bias, bias_val)
        nn.init.constant_(self.box_conv[-1].bias, 1.0)
        if self.feat_conv is not None:
            nn.init.normal_(self.feat_conv[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.feat_conv[-1].bias, 0.0)

    def forward(self, x):
        box = self.box_conv(x)   # [B, 4*reg_max, H, W]
        cls = self.cls_conv(x)   # [B, num_classes, H, W]
        if self.feat_conv is None:
            return torch.cat([box, cls], dim=1)
        feat = self.feat_conv(x) # [B, feature_dim, H, W]
        return torch.cat([box, cls, feat], dim=1)

class ScratchCSPHead(nn.Module):
    def __init__(self, neck_channels, num_classes: int, reg_max: int = 16, feature_dim: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.feature_dim = int(feature_dim)
        self.no          = num_classes + reg_max * 4 + self.feature_dim

        self.heads = nn.ModuleList([
            DecoupledHead(c, num_classes, reg_max, feature_dim=self.feature_dim) for c in neck_channels
        ])

    def forward(self, features):
        return [head(f) for head, f in zip(self.heads, features)]

class ScratchCSP(nn.Module):
    def __init__(self, num_classes: int = 5, width: float = 0.5, depth: float = 0.33,
                 reg_max: int = 16, strides=(8, 16, 32), feature_dim: int = 0,
                 backbone_name: str = 'scratch', pretrained_backbone: bool = False,
                 freeze_backbone_stem: bool = True, backbone_compress_ratio: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.strides     = strides
        self.feature_dim = int(feature_dim)
        self.backbone_name = str(backbone_name).lower()

        if self.backbone_name in ('scratch', 'scratchcsp', 'csp'):
            self.backbone = ScratchCSPBackbone(width=width, depth=depth)
        elif self.backbone_name.startswith('resnet'):
            self.backbone = ImageNetResNetBackbone(
                name=self.backbone_name, pretrained=pretrained_backbone,
                out_width=width, freeze_stem=freeze_backbone_stem,
                compress_ratio=backbone_compress_ratio,
            )
        else:
            raise ValueError(f'Unsupported backbone_name: {backbone_name}')

        self.neck     = ScratchCSPNeck(self.backbone.out_channels, depth=depth)
        self.head     = ScratchCSPHead(self.neck.out_channels, num_classes, reg_max, feature_dim=self.feature_dim)

        # Pre-build anchor-point grids (updated in forward if size changes)
        self._img_size  = None
        self._anchors   = None   # [total_anchors, 2]  (x, y)
        self._strides_t = None   # [total_anchors]  (stride value per anchor)

    # â”€â”€ Grid helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _make_grid(self, img_size: int, device):
        """Pre-compute (cx, cy) grid and stride tensor for all FPN levels."""
        anchors, strides_t = [], []
        for stride in self.strides:
            n = img_size // stride
            gy, gx = torch.meshgrid(
                torch.arange(n, device=device, dtype=torch.float32),
                torch.arange(n, device=device, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack([gx, gy], dim=-1).view(-1, 2) + 0.5  # cell center
            anchors.append(grid)
            strides_t.append(torch.full((n*n,), stride, device=device, dtype=torch.float32))
        return torch.cat(anchors), torch.cat(strides_t)

    # â”€â”€ Box decoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def dist2bbox(dist, anchor_points, xywh=True):
        """
        dist: [B, 4, N]  (lt, rb as DFL outputs, in grid units)
        anchor_points: [N, 2]
        Returns xyxy or xywh (normalized 0-1 after / stride * img_size)
        """
        lt = dist[:, :2, :]          # left-top   [B, 2, N]
        rb = dist[:, 2:, :]          # right-bottom
        ap = anchor_points.T.unsqueeze(0)  # [1, 2, N]
        x1y1 = ap - lt
        x2y2 = ap + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh   = x2y2 - x1y1
            return torch.cat([c_xy, wh], dim=1)  # [B, 4, N]
        return torch.cat([x1y1, x2y2], dim=1)    # [B, 4, N]

    # â”€â”€ Forward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def forward(self, x):
        img_size = x.shape[-1]
        device   = x.device

        # Refresh grids only when img_size changes
        if self._img_size != img_size:
            self._anchors, self._strides_t = self._make_grid(img_size, device)
            self._img_size = img_size

        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        raw = self.head([n3, n4, n5])  # list of [B, no, H, W]

        # Training: return raw feature maps (loss computed outside)
        if self.training:
            return raw

        # Inference: decode to (cx, cy, w, h, cls_scores) in pixel coords
        preds = []
        anchor_ptr = 0
        for feat, stride in zip(raw, self.strides):
            B, _, H, W = feat.shape
            N = H * W
            feat = feat.view(B, -1, N)         # [B, 4*rm+nc, N]

            box_raw  = feat[:, :4*self.reg_max, :]  # [B, 4*rm, N]
            cls_start = 4 * self.reg_max
            cls_end   = cls_start + self.num_classes
            cls_raw   = feat[:, cls_start:cls_end, :]  # [B, nc, N]

            # DFL â†’ distances
            box_raw_4 = box_raw.view(B, 4, self.reg_max, N).softmax(dim=2)
            proj      = torch.arange(self.reg_max, device=device, dtype=torch.float32)
            dist      = (box_raw_4 * proj.view(1, 1, -1, 1)).sum(dim=2)  # [B, 4, N]

            anchors_i = self._anchors[anchor_ptr: anchor_ptr + N]
            xywh_grid = self.dist2bbox(dist, anchors_i, xywh=True)   # [B, 4, N]
            xywh_pix  = xywh_grid * stride                            # pixel coords

            cls_prob = cls_raw.sigmoid()          # [B, nc, N]
            pred = torch.cat([xywh_pix, cls_prob], dim=1).permute(0, 2, 1)  # [B, N, 4+nc]
            preds.append(pred)
            anchor_ptr += N

        return torch.cat(preds, dim=1)  # [B, total_anchors, 4+nc]

def bbox_iou(b1, b2, xywh=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    """TÃ­nh toÃ¡n Complete-IoU (CIoU) hoÃ n toÃ n tá»« cÃ´ng thá»©c toÃ¡n há»c cÆ¡ sá»Ÿ."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = b1.unbind(-1), b2.unbind(-1)
        b1_x1, b1_x2 = x1 - w1/2, x1 + w1/2
        b1_y1, b1_y2 = y1 - h1/2, y1 + h1/2
        b2_x1, b2_x2 = x2 - w2/2, x2 + w2/2
        b2_y1, b2_y2 = y2 - h2/2, y2 + h2/2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = b1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = b2.unbind(-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1*h1 + w2*h2 - inter + eps
    iou   = inter / union

    # TÃ­nh toÃ¡n thÃ nh pháº§n CIoU (Khoáº£ng cÃ¡ch tÃ¢m & Tá»· lá»‡ khung hÃ¬nh)
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw**2 + ch**2 + eps  # ÄÆ°á»ng chÃ©o bao lá»“i
        rho2 = ((b2_x1+b2_x2 - b1_x1-b1_x2)**2 + (b2_y1+b2_y2 - b1_y1-b1_y2)**2) / 4 # Khoáº£ng cÃ¡ch 2 tÃ¢m
        
        if CIoU:
            v = (4 / math.pi**2) * (torch.atan(w2/(h2+eps)) - torch.atan(w1/(h1+eps)))**2
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            return iou - (rho2/c2 + v*alpha)
        return iou - rho2/c2

    return iou

def xywh2xyxy(box):
    """Chuyá»ƒn Ä‘á»•i [..., 4] xywh â†’ xyxy báº±ng tensor."""
    xy, wh = box[..., :2], box[..., 2:]
    return torch.cat([xy - wh/2, xy + wh/2], dim=-1)

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9,
                 per_class_topk=None):
        super().__init__()
        self.topk  = topk
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.per_class_topk = {int(k): int(v) for k, v in (per_class_topk or {}).items()}

    @torch.no_grad()
    def forward(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, mask_gt):
        B, N, nc = pred_scores.shape
        max_gt = gt_boxes.shape[1]

        target_labels  = torch.zeros(B, N,     dtype=torch.long,    device=gt_boxes.device)
        target_boxes   = torch.zeros(B, N, 4,  dtype=torch.float32, device=gt_boxes.device)
        target_scores  = torch.zeros(B, N, nc, dtype=torch.float32, device=gt_boxes.device)
        fg_mask        = torch.zeros(B, N,     dtype=torch.bool,    device=gt_boxes.device)

        if mask_gt.sum() == 0:
            return target_labels, target_boxes, fg_mask, target_scores

        for b in range(B):
            valid = mask_gt[b]          
            if not valid.any(): continue
            gboxes = gt_boxes[b][valid]  
            glabels = gt_labels[b][valid] 
            ng = gboxes.shape[0]

            # â”€â”€ 1. Kiá»ƒm tra tÃ¢m anchor cÃ³ náº±m trong GT Box hay khÃ´ng â”€â”€â”€â”€â”€â”€
            ap = anchor_points.unsqueeze(0).expand(ng, -1, -1)  
            gbox_exp = gboxes.unsqueeze(1).expand(-1, N, -1)   
            in_box = (
                (ap[..., 0] > gbox_exp[..., 0]) & (ap[..., 0] < gbox_exp[..., 2]) &
                (ap[..., 1] > gbox_exp[..., 1]) & (ap[..., 1] < gbox_exp[..., 3])
            )

            # â”€â”€ 2. TÃ­nh Ä‘iá»ƒm Task-aligned metric (Score^alpha * IoU^beta) â”€
            pbox = pred_boxes[b]          
            pscore = pred_scores[b]       

            iou = bbox_iou(
                pbox.unsqueeze(0).expand(ng, -1, -1),
                gboxes.unsqueeze(1).expand(-1, N, -1),
                xywh=False
            ).clamp(0)

            cls_score = pscore[:, glabels].T  
            metric = (cls_score ** self.alpha) * (iou ** self.beta) * in_box.float()

            # â”€â”€ 3. Chá»n Top-k anchors tá»‘t nháº¥t cho má»—i GT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            topk_mask = torch.zeros(ng, N, dtype=torch.bool, device=gt_boxes.device)
            if self.per_class_topk:
                for gi in range(ng):
                    cls_id = int(glabels[gi].item())
                    k = min(int(self.per_class_topk.get(cls_id, self.topk)), N)
                    topk_mask[gi, metric[gi].topk(k).indices] = True
            else:
                k = min(self.topk, N)
                topk_idx = metric.topk(k, dim=1).indices   
                topk_mask.scatter_(1, topk_idx, True)
            assign_mask = topk_mask & in_box

            # Fallback cho object nhá»: náº¿u bbox khÃ´ng chá»©a tÃ¢m anchor nÃ o,
            # gÃ¡n anchor gáº§n tÃ¢m GT nháº¥t Ä‘á»ƒ trÃ¡nh máº¥t hoÃ n toÃ n positive sample.
            no_pos = ~assign_mask.any(dim=1)
            if no_pos.any():
                centers = (gboxes[:, :2] + gboxes[:, 2:]) / 2
                dist2 = ((anchor_points.unsqueeze(0) - centers.unsqueeze(1)) ** 2).sum(-1)
                rows = no_pos.nonzero(as_tuple=True)[0]
                cols = dist2[rows].argmin(dim=1)
                assign_mask[rows, cols] = True

            # Xá»­ lÃ½ xung Ä‘á»™t: 1 anchor Ä‘Æ°á»£c gÃ¡n cho nhiá»u GT -> Chá»n GT cÃ³ metric cao nháº¥t
            if ng > 1:
                best_gt  = metric.argmax(dim=0)        
                best_mask = torch.zeros_like(assign_mask)
                best_mask[best_gt, torch.arange(N, device=gt_boxes.device)] = True
                assign_mask = assign_mask & best_mask

            pos_mask = assign_mask.any(dim=0)          
            gt_idx   = assign_mask.float().argmax(dim=0)  

            fg_mask[b] = pos_mask
            target_labels[b, pos_mask]   = glabels[gt_idx[pos_mask]]
            target_boxes[b, pos_mask]    = gboxes[gt_idx[pos_mask]]
            
            # GÃ¡n Ä‘iá»ƒm Soft IoU cho class tÆ°Æ¡ng á»©ng
            iou_for_score = iou[gt_idx[pos_mask], torch.where(pos_mask)[0]]
            ts = torch.zeros(N, nc, device=gt_boxes.device)
            ts[pos_mask, glabels[gt_idx[pos_mask]]] = iou_for_score.float()
            target_scores[b] = ts

        return target_labels, target_boxes, fg_mask, target_scores

class ScratchCSPLoss(nn.Module):
    def __init__(self, model: ScratchCSP, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5,
                 smooth_l1_gain: float = 0.0, smooth_l1_beta: float = 0.05,
                 class_weights=None, box_class_weights=None, assigner_topk: int = 10, target_score_floor: float = 0.0,
                 cls_focal_gamma: float = 0.0, cls_focal_alpha: float = -1.0,
                 positive_weight_only: bool = True, cls_loss_type: str = 'bce',
                 normalizer_mode: str = 'positive', feature_gain: float = 0.0,
                 feature_temperature: float = 0.20, feature_max_pos: int = 256,
                 chair_class_id: int = 4, chair_assigner_topk: int = 0,
                 hard_negative_mining: bool = False,
                 hard_negative_conf: float = 0.28,
                 hard_negative_topk: int = 256,
                 hard_negative_gain: float = 0.0,
                 hard_negative_class_weights=None,
                 rank_bce_gain: float = 0.0,
                 rank_bce_classes=None,
                 rank_bce_margin: float = 0.15,
                 rank_bce_pos_topk: int = 96,
                 rank_bce_neg_topk: int = 256,
                 rank_bce_min_pos_quality: float = 0.20,
                 chair_hard_negative_mining: bool = False,
                 chair_hard_negative_conf: float = 0.20,
                 chair_hard_negative_topk: int = 256,
                 chair_hard_negative_gain: float = 0.0):
        super().__init__()
        self.model     = model
        self.nc        = model.num_classes
        self.reg_max   = model.reg_max
        self.strides   = model.strides
        self.box_gain  = box_gain
        self.cls_gain  = cls_gain
        self.dfl_gain  = dfl_gain
        self.smooth_l1_gain = float(smooth_l1_gain)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.target_score_floor = float(target_score_floor)
        self.cls_focal_gamma = float(cls_focal_gamma)
        self.cls_focal_alpha = float(cls_focal_alpha)
        self.positive_weight_only = bool(positive_weight_only)
        self.cls_loss_type = str(cls_loss_type).lower()
        self.normalizer_mode = str(normalizer_mode).lower()
        self.feature_dim = int(getattr(model, 'feature_dim', 0))
        self.feature_gain = float(feature_gain)
        self.feature_temperature = float(feature_temperature)
        self.feature_max_pos = int(feature_max_pos)
        self.chair_class_id = max(0, min(int(chair_class_id), self.nc - 1))
        # Backward compatibility: old chair_hard_* config maps into the generic path.
        if hard_negative_gain <= 0 and chair_hard_negative_gain > 0:
            hard_negative_gain = chair_hard_negative_gain
            hard_negative_conf = chair_hard_negative_conf
            hard_negative_topk = chair_hard_negative_topk
            hard_negative_mining = True
        self.hard_negative_mining = bool(hard_negative_mining or chair_hard_negative_mining)
        self.hard_negative_conf = float(hard_negative_conf)
        self.hard_negative_topk = int(hard_negative_topk)
        self.hard_negative_gain = float(hard_negative_gain)
        self.base_hard_negative_gain = float(hard_negative_gain)
        if hard_negative_class_weights is None:
            hard_negative_class_weights = torch.ones(self.nc, dtype=torch.float32)
        self.register_buffer(
            'hard_negative_class_weights',
            torch.as_tensor(hard_negative_class_weights, dtype=torch.float32).view(1, 1, self.nc)
        )
        self.rank_bce_gain = float(rank_bce_gain)
        self.base_rank_bce_gain = float(rank_bce_gain)
        if rank_bce_classes is None:
            rank_bce_classes = []
        self.rank_bce_classes = tuple(sorted({
            int(c) for c in rank_bce_classes if 0 <= int(c) < self.nc
        }))
        self.rank_bce_margin = float(rank_bce_margin)
        self.rank_bce_pos_topk = int(rank_bce_pos_topk)
        self.rank_bce_neg_topk = int(rank_bce_neg_topk)
        self.rank_bce_min_pos_quality = float(rank_bce_min_pos_quality)

        chair_topk = int(chair_assigner_topk)
        per_class_topk = {self.chair_class_id: chair_topk} if chair_topk > int(assigner_topk) else None
        self.assigner  = TaskAlignedAssigner(topk=assigner_topk, per_class_topk=per_class_topk)

        proj = torch.arange(self.reg_max, dtype=torch.float32)
        self.register_buffer('proj', proj)
        if class_weights is None:
            class_weights = torch.ones(self.nc, dtype=torch.float32)
        self.register_buffer('class_weights', torch.as_tensor(class_weights, dtype=torch.float32).view(1, 1, self.nc))
        if box_class_weights is None:
            box_class_weights = torch.ones(self.nc, dtype=torch.float32)
        self.register_buffer('box_class_weights', torch.as_tensor(box_class_weights, dtype=torch.float32).view(self.nc))

    def _decode_preds(self, raw_preds, img_size, device):
        """Giáº£i mÃ£ raw tensor thÃ nh tá»a Ä‘á»™ pixel tháº­t."""
        dist_list, score_list, box_list = [], [], []
        feature_list = []
        anchor_list, stride_list = [], []

        for feat, stride in zip(raw_preds, self.strides):
            B, _, H, W = feat.shape
            N = H * W
            feat = feat.view(B, -1, N)

            base = 4 * self.reg_max
            dist_raw  = feat[:, :base, :]
            cls_raw   = feat[:, base:base + self.nc, :]
            feat_raw  = feat[:, base + self.nc:, :] if self.feature_dim > 0 else None

            dist_4 = dist_raw.view(B, 4, self.reg_max, N).softmax(2)
            dist   = (dist_4 * self.proj.view(1,1,-1,1)).sum(2)

            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            ap = (torch.stack([gx, gy], -1).view(-1, 2) + 0.5) * stride

            lt = dist[:, :2, :] * stride
            rb = dist[:, 2:, :] * stride
            ap_b = ap.T.unsqueeze(0)
            xyxy = torch.cat([ap_b - lt, ap_b + rb], dim=1).permute(0,2,1)

            dist_list.append(dist_raw)
            score_list.append(cls_raw.permute(0,2,1))
            if feat_raw is not None and feat_raw.numel():
                feature_list.append(feat_raw.permute(0,2,1))
            box_list.append(xyxy)
            anchor_list.append(ap)
            stride_list.append(torch.full((N,), stride, device=device))

        pred_dist   = torch.cat(dist_list, dim=2)
        pred_scores = torch.cat(score_list, dim=1)
        pred_boxes  = torch.cat(box_list, dim=1)
        anchor_pts  = torch.cat(anchor_list, dim=0)
        stride_t    = torch.cat(stride_list, dim=0)
        pred_features = torch.cat(feature_list, dim=1) if feature_list else None
        return pred_dist, pred_scores, pred_boxes, anchor_pts, stride_t, pred_features

    def _dfl_loss(self, pred_dist, target_dist):
        """Distribution Focal Loss per-positive-anchor, shape [num_pos]."""
        rm = self.reg_max
        target_dist = target_dist.clamp(0, rm - 1 - 1e-4)
        tl = target_dist.long()
        tr = tl + 1
        wl = tr.float() - target_dist
        wr = 1.0 - wl

        pred = pred_dist.view(-1, 4, rm)
        loss = (
            F.cross_entropy(pred.view(-1, rm), tl.view(-1), reduction='none') * wl.view(-1) +
            F.cross_entropy(pred.view(-1, rm), tr.view(-1), reduction='none') * wr.view(-1)
        ).view(-1, 4).mean(-1)
        return loss

    def _cls_loss(self, pred_scores, target_scores, normalizer):
        """Classification loss on soft quality targets.

        Default is BCE. Optional focal/qfl/varifocal modes are implemented so you can
        switch safely from Config without rewriting the notebook.
        Class weights are applied primarily to positive targets to avoid amplifying
        negative anchors for chair/person, which the confusion matrix showed as the
        main source of false positives.
        """
        loss_type = self.cls_loss_type
        bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
        pred_prob = pred_scores.sigmoid()

        if loss_type in ('focal', 'bce_focal') and self.cls_focal_gamma > 0:
            p_t = target_scores * pred_prob + (1.0 - target_scores) * (1.0 - pred_prob)
            focal_weight = (1.0 - p_t).clamp(min=0.0).pow(self.cls_focal_gamma)
            if self.cls_focal_alpha >= 0:
                alpha = self.cls_focal_alpha
                alpha_weight = target_scores * alpha + (1.0 - target_scores) * (1.0 - alpha)
                focal_weight = focal_weight * alpha_weight
            loss = bce * focal_weight
        elif loss_type in ('qfl', 'quality_focal') and self.cls_focal_gamma > 0:
            # Quality Focal Loss style: focuses on anchors whose score is far from the soft quality target.
            loss = bce * (target_scores - pred_prob).abs().pow(self.cls_focal_gamma)
        elif loss_type in ('varifocal', 'vfl') and self.cls_focal_gamma > 0:
            alpha = self.cls_focal_alpha if self.cls_focal_alpha >= 0 else 0.75
            neg_weight = alpha * pred_prob.pow(self.cls_focal_gamma) * (target_scores <= 0).float()
            pos_weight = target_scores * (target_scores > 0).float()
            loss = bce * (neg_weight + pos_weight)
        else:
            loss = bce

        class_weights = self.class_weights.to(pred_scores.device)
        if self.positive_weight_only:
            pos_strength = target_scores.clamp(0.0, 1.0)
            loss = loss * (1.0 + pos_strength * (class_weights - 1.0))
        else:
            loss = loss * class_weights

        return loss.sum() / normalizer

    def _hard_negative_loss(self, pred_scores, fg_mask):
        if not self.hard_negative_mining or self.hard_negative_gain <= 0:
            return torch.tensor(0.0, device=pred_scores.device)
        probs = pred_scores.sigmoid().detach()
        hard_mask = (~fg_mask).unsqueeze(-1) & (probs > self.hard_negative_conf)
        if not hard_mask.any():
            return torch.tensor(0.0, device=pred_scores.device)

        hard_logits = pred_scores[hard_mask]
        hard_probs = probs[hard_mask]
        class_weights = self.hard_negative_class_weights.to(pred_scores.device).expand_as(pred_scores)
        hard_weights = class_weights[hard_mask].detach().clamp(min=0.05)
        hard_scores_num = hard_probs.numel()
        topk = min(max(self.hard_negative_topk, 1), hard_scores_num)
        if hard_scores_num > topk:
            keep = (hard_probs * hard_weights).topk(topk).indices
            hard_logits = hard_logits[keep]
            hard_weights = hard_weights[keep]

        loss = F.binary_cross_entropy_with_logits(hard_logits, torch.zeros_like(hard_logits), reduction='none')
        return (loss * hard_weights).sum() / hard_weights.sum().clamp_min(1.0)

    def _rank_aware_bce_loss(self, pred_scores, fg_mask, target_scores):
        if self.rank_bce_gain <= 0 or len(self.rank_bce_classes) == 0:
            return torch.tensor(0.0, device=pred_scores.device)

        losses = []
        pos_topk = max(int(self.rank_bce_pos_topk), 1)
        neg_topk = max(int(self.rank_bce_neg_topk), 1)
        for cls_id in self.rank_bce_classes:
            cls_logits = pred_scores[..., cls_id]
            cls_targets = target_scores[..., cls_id]
            pos_mask = cls_targets >= self.rank_bce_min_pos_quality
            if not pos_mask.any():
                pos_mask = cls_targets > 0
            neg_mask = (~fg_mask) & (cls_targets <= 0)
            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_logits = cls_logits[pos_mask]
            pos_weights = cls_targets[pos_mask].detach().clamp(min=0.05)
            if pos_logits.numel() > pos_topk:
                keep = pos_weights.topk(pos_topk).indices
                pos_logits = pos_logits[keep]
                pos_weights = pos_weights[keep]

            neg_logits = cls_logits[neg_mask]
            if neg_logits.numel() > neg_topk:
                keep = neg_logits.detach().topk(neg_topk).indices
                neg_logits = neg_logits[keep]

            pair_loss = F.softplus(neg_logits[:, None] - pos_logits[None, :] + self.rank_bce_margin)
            pair_loss = pair_loss * pos_weights.view(1, -1)
            denom = pos_weights.sum().clamp_min(1.0) * float(max(int(neg_logits.numel()), 1))
            losses.append(pair_loss.sum() / denom)

        if not losses:
            return torch.tensor(0.0, device=pred_scores.device)
        return torch.stack(losses).mean()

    def _feature_contrastive_loss(self, pred_features, fg_mask, target_labels, target_scores):
        if self.feature_gain <= 0 or pred_features is None or not fg_mask.any():
            return torch.tensor(0.0, device=target_scores.device)

        labels = target_labels[fg_mask].long()
        weights = target_scores.sum(-1)[fg_mask].float().clamp(min=0.0)
        feats = pred_features[fg_mask]
        valid = weights > 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=target_scores.device)

        labels = labels[valid]
        weights = weights[valid]
        feats = feats[valid]
        if labels.unique().numel() < 2:
            return torch.tensor(0.0, device=target_scores.device)

        max_pos = max(int(self.feature_max_pos), 16)
        if feats.shape[0] > max_pos:
            idx = weights.topk(max_pos).indices
            feats = feats[idx]
            labels = labels[idx]
            weights = weights[idx]

        feats = F.normalize(feats.float(), dim=-1)
        temp = max(float(self.feature_temperature), 1e-3)
        sim = torch.matmul(feats, feats.t()) / temp
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        same = labels[:, None].eq(labels[None, :]) & ~eye
        valid_rows = same.any(dim=1)
        if not valid_rows.any():
            return torch.tensor(0.0, device=target_scores.device)

        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * (~eye).float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-9))
        pos_count = same.float().sum(dim=1).clamp_min(1.0)
        loss = -(same.float() * log_prob).sum(dim=1) / pos_count
        row_weights = weights[valid_rows].detach().clamp(min=0.05)
        return (loss[valid_rows] * row_weights).sum() / row_weights.sum().clamp_min(1.0)

    def forward(self, raw_preds, targets, img_size):
        device = raw_preds[0].device
        B = raw_preds[0].shape[0]

        pred_dist, pred_scores, pred_boxes, anchor_pts, stride_t, pred_features = \
            self._decode_preds(raw_preds, img_size, device)

        max_gt = int(targets[:, 0].long().bincount(minlength=B).max().item()) if len(targets) else 0
        gt_labels = torch.zeros(B, max_gt, dtype=torch.long, device=device)
        gt_boxes  = torch.zeros(B, max_gt, 4, device=device)
        mask_gt   = torch.zeros(B, max_gt, dtype=torch.bool, device=device)

        if len(targets):
            bi = targets[:, 0].long()
            cls = targets[:, 1].long()
            xywh_pix = targets[:, 2:] * img_size
            xyxy_pix = xywh2xyxy(xywh_pix)

            counts = torch.zeros(B, dtype=torch.long, device=device)
            for idx in range(len(targets)):
                b = bi[idx].item()
                j = counts[b].item()
                gt_labels[b, j] = cls[idx]
                gt_boxes[b, j]  = xyxy_pix[idx]
                mask_gt[b, j]   = True
                counts[b] += 1

        with torch.no_grad():
            tgt_labels, tgt_boxes, fg_mask, tgt_scores = self.assigner(
                pred_scores.sigmoid().detach(), pred_boxes.detach(),
                anchor_pts, gt_labels, gt_boxes, mask_gt
            )

        if self.target_score_floor > 0:
            positive_targets = tgt_scores > 0
            tgt_scores = torch.where(positive_targets, tgt_scores.clamp(min=self.target_score_floor), tgt_scores)

        target_scores_sum = tgt_scores.sum()
        num_positives = fg_mask.sum().float()

        if self.normalizer_mode == 'quality':
            normalizer = target_scores_sum.clamp(min=1.0)
        else:
            normalizer = torch.max(target_scores_sum, num_positives).clamp(min=1.0)

        cls_loss = self._cls_loss(pred_scores, tgt_scores, normalizer)
        feat_loss = self._feature_contrastive_loss(pred_features, fg_mask, tgt_labels, tgt_scores)
        hard_negative_loss = self._hard_negative_loss(pred_scores, fg_mask)
        rank_bce_loss = self._rank_aware_bce_loss(pred_scores, fg_mask, tgt_scores)
        box_loss = torch.tensor(0.0, device=device)
        dfl_loss = torch.tensor(0.0, device=device)
        smooth_l1_loss = torch.tensor(0.0, device=device)

        if fg_mask.any():
            pred_b = pred_boxes[fg_mask]
            tgt_b  = tgt_boxes[fg_mask]
            bbox_weight = tgt_scores.sum(-1)[fg_mask].clamp(min=0.0)
            fg_labels = tgt_labels[fg_mask].long().clamp(0, self.nc - 1)
            bbox_weight = bbox_weight * self.box_class_weights.to(device=bbox_weight.device)[fg_labels]

            iou_val = bbox_iou(pred_b, tgt_b, xywh=False, CIoU=True)
            box_loss = ((1.0 - iou_val) * bbox_weight).sum() / normalizer

            pred_b_norm = pred_b / float(img_size)
            tgt_b_norm = tgt_b / float(img_size)
            smooth_raw = F.smooth_l1_loss(
                pred_b_norm, tgt_b_norm, reduction='none',
                beta=max(float(self.smooth_l1_beta), 1e-6)
            ).mean(-1)
            smooth_l1_loss = (smooth_raw * bbox_weight).sum() / normalizer

            ap_fg   = anchor_pts.unsqueeze(0).expand(B, -1, -1)[fg_mask]
            st_fg   = stride_t.unsqueeze(0).expand(B, -1)[fg_mask]

            tgt_lt  = (ap_fg - tgt_b[:, :2]) / st_fg.unsqueeze(1)
            tgt_rb  = (tgt_b[:, 2:] - ap_fg) / st_fg.unsqueeze(1)
            tgt_d   = torch.cat([tgt_lt, tgt_rb], dim=1).clamp(0, self.reg_max - 1)

            pred_d_fg = pred_dist.permute(0,2,1)[fg_mask]
            dfl_raw = self._dfl_loss(pred_d_fg, tgt_d)
            dfl_loss = (dfl_raw * bbox_weight).sum() / normalizer

        total = (
            self.box_gain * box_loss + self.cls_gain * cls_loss +
            self.dfl_gain * dfl_loss + self.smooth_l1_gain * smooth_l1_loss +
            self.feature_gain * feat_loss + self.hard_negative_gain * hard_negative_loss +
            self.rank_bce_gain * rank_bce_loss
        )
        return total, {
            'box': float(box_loss.detach().item()),
            'cls': float(cls_loss.detach().item()),
            'dfl': float(dfl_loss.detach().item()),
            'smooth_l1': float(smooth_l1_loss.detach().item()),
            'feat': float(feat_loss.detach().item()),
            'hard_neg': float(hard_negative_loss.detach().item()),
            'rank_bce': float(rank_bce_loss.detach().item())
        }

def nms(boxes, scores, iou_thres: float = 0.55):
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    if torchvision_nms is not None:
        return torchvision_nms(boxes, scores, iou_thres)

    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        i = order[0].item()
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        iou = bbox_iou(boxes[i].unsqueeze(0).expand(len(rest), -1),
                       boxes[rest], xywh=False, CIoU=False).squeeze()
        order = rest[iou <= iou_thres]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def postprocess(pred, conf_thres=0.001, iou_thres=0.55, max_det=600, nms_pre_topk=8000,
                class_conf_thres=None, class_iou_thres=None):
    """Class-aware NMS. pred: [B, NA, 4+nc] in pixel xywh + class probabilities."""
    results = []
    for p in pred:
        scores, cls_ids = p[:, 4:].max(dim=1)
        num_classes = p.shape[1] - 4
        if class_conf_thres is not None:
            thresholds = torch.as_tensor(class_conf_thres, dtype=scores.dtype, device=scores.device)
            if thresholds.numel() != num_classes:
                raise ValueError('class_conf_thres length must match num_classes')
            mask = scores > thresholds[cls_ids]
        else:
            mask = scores > conf_thres
        p, scores, cls_ids = p[mask], scores[mask], cls_ids[mask]

        if len(p) == 0:
            results.append(torch.zeros(0, 6, device=pred.device))
            continue

        if nms_pre_topk is not None and len(scores) > nms_pre_topk:
            topk = scores.topk(nms_pre_topk).indices
            p, scores, cls_ids = p[topk], scores[topk], cls_ids[topk]

        boxes_xyxy = xywh2xyxy(p[:, :4])
        class_nms_iou = None
        if class_iou_thres is not None:
            class_nms_iou = torch.as_tensor(class_iou_thres, dtype=scores.dtype, device=scores.device)
            if class_nms_iou.numel() != num_classes:
                raise ValueError('class_iou_thres length must match num_classes')
        dets = []
        for cls in cls_ids.unique():
            m = cls_ids == cls
            cls_iou_thres = float(class_nms_iou[int(cls.item())].item()) if class_nms_iou is not None else iou_thres
            keep = nms(boxes_xyxy[m], scores[m], cls_iou_thres)
            cls_boxes = boxes_xyxy[m][keep]
            cls_scores = scores[m][keep]
            cls_col = torch.full((len(keep), 1), float(cls.item()), device=pred.device)
            dets.append(torch.cat([cls_boxes, cls_scores.unsqueeze(1), cls_col], dim=1))

        det = torch.cat(dets, dim=0) if dets else torch.zeros(0, 6, device=pred.device)
        if len(det):
            det = det[det[:, 4].argsort(descending=True)[:max_det]]
        results.append(det)
    return results

def _detections_to_official_boxes(det, rec, cfg: Config):
    if det is None or len(det) == 0:
        return []
    h = int(rec.get('height', 0))
    w = int(rec.get('width', 0))
    if h <= 0 or w <= 0:
        return []
    scale = min(float(cfg.img_size) / float(h), float(cfg.img_size) / float(w))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x = (int(cfg.img_size) - nw) // 2
    pad_y = (int(cfg.img_size) - nh) // 2
    arr = det.detach().cpu().numpy() if hasattr(det, 'detach') else np.asarray(det)
    converted = []
    for x1, y1, x2, y2, score, cls_id in arr:
        x1 = np.clip((float(x1) - pad_x) / scale, 0, w)
        x2 = np.clip((float(x2) - pad_x) / scale, 0, w)
        y1 = np.clip((float(y1) - pad_y) / scale, 0, h)
        y2 = np.clip((float(y2) - pad_y) / scale, 0, h)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        converted.append([float(x1), float(y1), float(x2), float(y2), float(score), float(cls_id)])
    if not converted:
        return []
    converted = np.asarray(converted, dtype=np.float32)

    boxes = []
    for x1, y1, x2, y2, score, cls_id in converted:
        boxes.append({
            'class': cfg.class_names[int(cls_id)],
            'confidence': float(np.clip(score, 0.0, 1.0)),
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
        })
    return boxes

def run_official_map50_tmp(predictions, cfg: Config):
    eval_script = os.path.join(cfg.data_root, 'tools', 'evaluate_predictions.py')
    gt_json = os.path.join(cfg.data_root, cfg.val_ann)
    if not os.path.exists(eval_script):
        return None
    tmp_dir = tempfile.mkdtemp(prefix='official_map50_')
    pred_json = os.path.join(tmp_dir, 'predictions.json')
    score_json = os.path.join(tmp_dir, 'score.json')
    try:
        with open(pred_json, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False)
        cmd = [
            sys.executable, eval_script,
            '--ground_truth', gt_json,
            '--predictions', pred_json,
            '--output', score_json,
            '--iou_threshold', '0.5',
            '--max_detections_per_image', str(min(int(getattr(cfg, 'max_det', 100)), 100)),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(score_json, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return float(metrics.get('mAP@0.5', -1.0))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def compute_ap(recall, precision):
    if len(recall) == 0:
        return 0.0

    recall = np.asarray(recall, dtype=np.float64).tolist()
    precision = np.asarray(precision, dtype=np.float64).tolist()

    mrec = [0.0] + recall + [1.0]
    mpre = [0.0] + precision + [0.0]

    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]

    return float(ap)

def bbox_iou_eval(box_a, box_b):
    """
    Same IoU as evaluate_predictions.py.
    Input: xyxy [xmin, ymin, xmax, ymax].
    """
    if hasattr(box_a, "detach"):
        box_a = box_a.detach().cpu().tolist()
    if hasattr(box_b, "detach"):
        box_b = box_b.detach().cpu().tolist()

    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0

    return intersection / union

def compute_detection_confusion_matrix(all_preds, all_gts, cfg: Config, iou_thres=0.5, conf_thres=0.0,
                                       class_conf_thres=None):
    nc = cfg.num_classes
    bg = nc
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    gt_by_img, pred_by_img = defaultdict(list), defaultdict(list)

    for gboxes, gcls, gid in all_gts:
        for box, cls_id in zip(gboxes, gcls):
            gt_by_img[int(gid)].append({
                'box': box, 'cls': int(cls_id.item()), 'matched': False
            })

    for pboxes, pscores, pcls, pid in all_preds:
        if class_conf_thres is not None:
            thresholds = torch.as_tensor(class_conf_thres, dtype=pscores.dtype)
            keep = pscores >= thresholds[pcls]
        else:
            keep = pscores >= conf_thres
        for box, score, cls_id in zip(pboxes[keep], pscores[keep], pcls[keep]):
            pred_by_img[int(pid)].append({
                'box': box, 'score': float(score.item()), 'cls': int(cls_id.item())
            })

    for gid in sorted(set(gt_by_img.keys()) | set(pred_by_img.keys())):
        gts = [dict(item) for item in gt_by_img.get(gid, [])]
        preds = sorted(pred_by_img.get(gid, []), key=lambda item: item['score'], reverse=True)

        for pred_item in preds:
            pred_cls = pred_item['cls']
            best_iou, best_gt = 0.0, -1
            for j, gt_item in enumerate(gts):
                if gt_item['matched']:
                    continue
                iou = bbox_iou(
                    pred_item['box'].unsqueeze(0),
                    gt_item['box'].unsqueeze(0),
                    xywh=False, CIoU=False
                ).clamp(0).item()
                if iou > best_iou:
                    best_iou, best_gt = iou, j

            if best_gt >= 0 and best_iou >= iou_thres:
                true_cls = gts[best_gt]['cls']
                matrix[true_cls, pred_cls] += 1
                gts[best_gt]['matched'] = True
            else:
                matrix[bg, pred_cls] += 1

        for gt_item in gts:
            if not gt_item['matched']:
                matrix[gt_item['cls'], bg] += 1

    summary = {}
    for ci, name in enumerate(cfg.class_names):
        tp = int(matrix[ci, ci])
        fp = int(matrix[:, ci].sum() - tp)
        fn = int(matrix[ci, :].sum() - tp)
        summary[name] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'recall': round(tp / max(tp + fn, 1), 6),
            'precision': round(tp / max(tp + fp, 1), 6),
        }
    return matrix, summary

def save_confusion_matrix_artifacts(matrix, summary, cfg: Config, prefix='val_confusion_matrix'):
    labels = list(cfg.class_names) + ['background']
    os.makedirs(cfg.save_dir, exist_ok=True)
    csv_path = os.path.join(cfg.save_dir, f'{prefix}.csv')
    norm_csv_path = os.path.join(cfg.save_dir, f'{prefix}_normalized.csv')
    json_path = os.path.join(cfg.save_dir, f'{prefix}_summary.json')
    png_path = os.path.join(cfg.save_dir, f'{prefix}.png')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true/pred'] + labels)
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label] + row)

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = matrix / np.maximum(row_sums, 1)
    with open(norm_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true/pred'] + labels)
        for label, row in zip(labels, norm.tolist()):
            writer.writerow([label] + [round(float(v), 6) for v in row])

    payload = {
        'labels': labels,
        'matrix': matrix.tolist(),
        'normalized_by_true': [[round(float(v), 6) for v in row] for row in norm.tolist()],
        'iou_threshold': getattr(cfg, 'cm_iou_thres', 0.5),
        'confidence_threshold': (
            getattr(cfg, 'class_conf_thres', None)
            if getattr(cfg, 'cm_use_class_conf_thres', False)
            else getattr(cfg, 'cm_conf_thres', 0.0)
        ),
        'per_class': summary,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix @ IoU {getattr(cfg, "cm_iou_thres", 0.5):.2f}')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value:
                ax.text(j, i, str(int(value)), ha='center', va='center',
                        color='white' if norm[i, j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.show()
    print(f'Saved confusion matrix: {png_path}')
    print(f'Saved confusion matrix CSV: {csv_path}')
    return payload

@torch.no_grad()
def predict_with_tta(model, imgs, cfg: Config, tta: bool = False):
    """Horizontal-flip TTA cho final evaluation. KhÃ´ng dÃ¹ng trong train eval Ä‘á»ƒ tiáº¿t kiá»‡m thá»i gian."""
    pred = model(imgs)
    if not tta:
        return pred

    imgs_flip = torch.flip(imgs, dims=[3])
    pred_flip = model(imgs_flip)
    pred_flip[..., 0] = cfg.img_size - pred_flip[..., 0]  # unflip cx
    return torch.cat([pred, pred_flip], dim=1)

@torch.no_grad()
def evaluate_map(model, loader, cfg: Config, iou_thresholds=None, verbose=True, tta=False, official=False):
    if iou_thresholds is None:
        iou_thresholds = np.array([0.5])

    model.eval()
    nc = cfg.num_classes
    all_preds = []
    all_gts = []
    official_predictions = [] if official and hasattr(loader.dataset, 'records') else None
    cfg._last_offical_map50 = None
    img_id = 0

    for imgs, targets in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        if getattr(cfg, 'channels_last', False) and cfg.device.startswith('cuda'):
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        pred = predict_with_tta(model, imgs, cfg, tta=tta)
        
        eval_class_thresholds = (
            getattr(cfg, "class_conf_thres", None)
            if getattr(cfg, "eval_use_class_conf_thres", False)
            else None
        )
    
        eval_class_ious = (
            getattr(cfg, "class_iou_thres", None)
            if getattr(cfg, "eval_use_class_conf_thres", False)
            else None
        )

        dets = postprocess(
            pred,
            getattr(cfg, "eval_conf_thres", cfg.conf_thres),
            cfg.iou_thres,
            cfg.max_det,
            cfg.nms_pre_topk,
            class_conf_thres=eval_class_thresholds,
            class_iou_thres=eval_class_ious,
        )

        bs = imgs.shape[0]
        for b in range(bs):
            gt_mask = targets[:, 0] == b
            gt = targets[gt_mask]
            if len(gt):
                gt_boxes_pix = xywh2xyxy(gt[:, 2:] * cfg.img_size)
                gt_cls = gt[:, 1].long()
                all_gts.append((gt_boxes_pix.cpu(), gt_cls.cpu(), img_id))

            det = dets[b]
            if len(det):
                all_preds.append((det[:, :4].cpu(), det[:, 4].cpu(), det[:, 5].long().cpu(), img_id))
            if official_predictions is not None:
                rec = loader.dataset.records[img_id]
                official_predictions.append({
                    'image_id': rec['image_id'],
                    'boxes': _detections_to_official_boxes(det, rec, cfg),
                })
            img_id += 1

    if official_predictions is not None:
        cfg._last_offical_map50 = run_official_map50_tmp(official_predictions, cfg)

    ap_matrix = np.zeros((len(iou_thresholds), nc))

    for ci in range(nc):
        gt_by_img = {}
        for gboxes, gcls, gid in all_gts:
            n = (gcls == ci).sum().item()
            gt_by_img[gid] = gt_by_img.get(gid, 0) + n
        total_gt = sum(gt_by_img.values())
        if total_gt == 0:
            continue

        preds_cls = []
        for pboxes, pscores, pcls, pid in all_preds:
            mask = pcls == ci
            if mask.any():
                for pb, ps in zip(pboxes[mask], pscores[mask]):
                    preds_cls.append((ps.item(), pb, pid))
        preds_cls.sort(key=lambda x: -x[0])

        for ti, iou_t in enumerate(iou_thresholds):
            tp = np.zeros(len(preds_cls))
            fp = np.zeros(len(preds_cls))
            detected = {}

            for pi, (score, pbox, pid) in enumerate(preds_cls):
                best_iou, best_j = -1, -1
                for gboxes, gcls, gid in all_gts:
                    if gid != pid:
                        continue
                    gi = (gcls == ci).nonzero(as_tuple=True)[0]
                    if len(gi) == 0:
                        continue
                    for j in gi:
                        j_int = int(j.item()) if hasattr(j, "item") else int(j)

                        iou = bbox_iou_eval(pbox, gboxes[j_int])
                        if iou > best_iou:
                            best_iou = iou
                            best_j = (pid, j_int)

                if best_iou >= iou_t and best_j not in detected.get(pid, set()):
                    tp[pi] = 1
                    detected.setdefault(pid, set()).add(best_j)
                else:
                    fp[pi] = 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / (total_gt + 1e-9)
            pre = cum_tp / (cum_tp + cum_fp + 1e-9)
            ap_matrix[ti, ci] = compute_ap(rec, pre)

    map50 = ap_matrix[0].mean()
    map5095 = ap_matrix.mean()

    if verbose:
        print(f'\n{"Class":<15} {"AP@0.5":>10}')
        print('-' * 27)
        for ci, name in enumerate(cfg.class_names):
            print(f'{name:<15} {ap_matrix[0, ci]:>10.4f}')
        print('-' * 27)
        print(f'{"mAP@0.5":<15} {map50:>10.4f}')
        if getattr(cfg, '_last_offical_map50', None) is not None:
            print(f'{"offical_mAP50":<15} {cfg._last_offical_map50:>10.4f}')

        if getattr(cfg, 'save_confusion_matrix', True):
            cm_class_thresholds = (
                getattr(cfg, "class_conf_thres", None)
                if getattr(cfg, "cm_use_class_conf_thres", False)
                else None
            )
            
            cm, cm_summary = compute_detection_confusion_matrix(
                all_preds, all_gts, cfg,
                iou_thres=getattr(cfg, 'cm_iou_thres', 0.5),
                conf_thres=getattr(cfg, 'cm_conf_thres', 0.0),
                class_conf_thres=cm_class_thresholds,
            )
            save_confusion_matrix_artifacts(cm, cm_summary, cfg)
            print('\nConfusion-matrix precision/recall:')
            for name in cfg.class_names:
                row = cm_summary[name]
                print(f'{name:<15} recall={row["recall"]:.4f} precision={row["precision"]:.4f} FN={row["false_negatives"]}')
    return map50, map5095, ap_matrix

def build_optimizer_scheduler(model, cfg: Config, n_batches: int):
    accum = max(int(getattr(cfg, 'accumulate', 1)), 1)
    steps_per_epoch = max(math.ceil(n_batches / accum), 1)
    schedule_epochs = max(int(getattr(cfg, 'lr_schedule_epochs', cfg.epochs)), int(cfg.epochs))
    total_steps = schedule_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    backbone_lr_mult = float(getattr(cfg, 'backbone_lr_mult', 1.0))
    use_backbone_lr = (
        str(getattr(cfg, 'backbone_name', '')).lower().startswith('resnet')
        and bool(getattr(cfg, 'pretrained_backbone', False))
        and backbone_lr_mult != 1.0
    )
    pretrained_prefixes = (
        'backbone.stem.', 'backbone.layer1.', 'backbone.layer2.',
        'backbone.layer3.', 'backbone.layer4.'
    )
    low_lr_param_ids = set()
    if use_backbone_lr:
        low_lr_param_ids = {
            id(p) for name, p in model.named_parameters()
            if p.requires_grad and name.startswith(pretrained_prefixes)
        }

    if cfg.use_adamw:
        main_params, backbone_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in low_lr_param_ids:
                backbone_params.append(p)
            else:
                main_params.append(p)
        param_groups = []
        if main_params:
            param_groups.append({'params': main_params, 'lr': cfg.lr0, 'weight_decay': cfg.weight_decay, 'is_pretrained_backbone': False})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': cfg.lr0 * backbone_lr_mult, 'weight_decay': cfg.weight_decay, 'is_pretrained_backbone': True})
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr0, weight_decay=cfg.weight_decay)
    else:
        g_wd, g_no_wd, g_bn = [], [], []
        bb_wd, bb_no_wd, bb_bn = [], [], []
        seen = set()

        def _add(main_lst, backbone_lst, p):
            if not p.requires_grad:
                return
            if id(p) in seen:
                return
            seen.add(id(p))
            if id(p) in low_lr_param_ids:
                backbone_lst.append(p)
            else:
                main_lst.append(p)

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in module.parameters():
                    _add(g_bn, bb_bn, p)
            else:
                if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                    _add(g_wd, bb_wd, module.weight)
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    _add(g_no_wd, bb_no_wd, module.bias)

        param_groups = []
        for params, wd, lr, is_bb in (
            (g_wd, cfg.weight_decay, cfg.lr0, False),
            (g_no_wd, 0.0, cfg.lr0, False),
            (g_bn, 0.0, cfg.lr0, False),
            (bb_wd, cfg.weight_decay, cfg.lr0 * backbone_lr_mult, True),
            (bb_no_wd, 0.0, cfg.lr0 * backbone_lr_mult, True),
            (bb_bn, 0.0, cfg.lr0 * backbone_lr_mult, True),
        ):
            if params:
                param_groups.append({'params': params, 'weight_decay': wd, 'lr': lr, 'is_pretrained_backbone': is_bb})
        optimizer = torch.optim.SGD(param_groups, lr=cfg.lr0, momentum=cfg.momentum, nesterov=True)

    if low_lr_param_ids:
        print(
            f'Pretrained backbone LR: full={cfg.lr0 * backbone_lr_mult:.2e} '
            f'(mult={backbone_lr_mult:.2f}, freeze={getattr(cfg, "backbone_freeze_epochs", 0)} epoch, '
            f'ramp={getattr(cfg, "backbone_ramp_epochs", 0)} epoch); head/neck LR: {cfg.lr0:.2e}'
        )

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return cfg.lrf + 0.5 * (1 - cfg.lrf) * (1 + math.cos(math.pi * progress))

    def backbone_lr_lambda(step):
        base = lr_lambda(step)
        freeze_epochs = float(getattr(cfg, 'backbone_freeze_epochs', 0))
        ramp_epochs = max(float(getattr(cfg, 'backbone_ramp_epochs', 1)), 1.0)
        frozen_scale = float(getattr(cfg, 'backbone_frozen_lr_scale', 0.0))
        epoch_pos = float(step) / max(float(steps_per_epoch), 1.0)
        if epoch_pos < freeze_epochs:
            return base * frozen_scale
        if epoch_pos < freeze_epochs + ramp_epochs:
            ramp = (epoch_pos - freeze_epochs) / ramp_epochs
            return base * max(frozen_scale, ramp)
        return base

    lr_lambdas = [
        backbone_lr_lambda if group.get('is_pretrained_backbone', False) else lr_lambda
        for group in optimizer.param_groups
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)
    return optimizer, scheduler

class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000):
        import copy
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

def load_checkpoint_for_resume(path, model, ema, optimizer, scheduler, scaler, cfg):
    """Resume tá»« last.pt. TÆ°Æ¡ng thÃ­ch cáº£ checkpoint v3 cÅ© vÃ  v4 má»›i."""
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    state_for_train = ckpt.get('model_train', ckpt.get('model', ckpt))
    model.load_state_dict(state_for_train, strict=True)

    ema_state = ckpt.get('ema', ckpt.get('model', state_for_train))
    ema.ema.load_state_dict(ema_state, strict=True)

    if getattr(cfg, 'channels_last', False) and cfg.device.startswith('cuda'):
        model.to(memory_format=torch.channels_last)
        ema.ema.to(memory_format=torch.channels_last)

    if getattr(cfg, 'resume_optimizer', True) and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Optimizer state resumed.')
        except Exception as e:
            print(f'Could not resume optimizer, continuing with fresh optimizer: {e}')
    if 'scheduler' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
            print('Scheduler state resumed.')
        except Exception as e:
            print(f'Could not resume scheduler, continuing with fresh scheduler: {e}')
    if 'scaler' in ckpt:
        try:
            scaler.load_state_dict(ckpt['scaler'])
            print('AMP scaler state resumed.')
        except Exception as e:
            print(f'Could not resume scaler: {e}')

    return ckpt

def set_pretrained_backbone_bn_eval(model, cfg):
    """Keep ImageNet backbone BN statistics stable during small-batch detection fine-tuning."""
    if not (
        str(getattr(cfg, 'backbone_name', '')).lower().startswith('resnet')
        and bool(getattr(cfg, 'pretrained_backbone', False))
        and bool(getattr(cfg, 'freeze_pretrained_bn', True))
    ):
        return
    prefixes = (
        'backbone.stem', 'backbone.layer1', 'backbone.layer2',
        'backbone.layer3', 'backbone.layer4'
    )
    for name, module in model.named_modules():
        if name.startswith(prefixes) and isinstance(module, nn.BatchNorm2d):
            module.eval()

def update_epoch_loss_gains(criterion, cfg, epoch):
    base_gain = float(getattr(cfg, 'hard_negative_gain', getattr(criterion, 'base_hard_negative_gain', 0.0)))
    start_epoch = int(getattr(cfg, 'hard_negative_start_epoch', 1))
    warmup_epochs = max(int(getattr(cfg, 'hard_negative_warmup_epochs', 1)), 1)
    if epoch < start_epoch:
        scale = 0.0
    else:
        scale = min(1.0, float(epoch - start_epoch + 1) / float(warmup_epochs))
    criterion.hard_negative_gain = base_gain * scale

    rank_base_gain = float(getattr(cfg, 'rank_bce_gain', getattr(criterion, 'base_rank_bce_gain', 0.0)))
    rank_start_epoch = int(getattr(cfg, 'rank_bce_start_epoch', start_epoch))
    rank_warmup_epochs = max(int(getattr(cfg, 'rank_bce_warmup_epochs', warmup_epochs)), 1)
    if epoch < rank_start_epoch:
        rank_scale = 0.0
    else:
        rank_scale = min(1.0, float(epoch - rank_start_epoch + 1) / float(rank_warmup_epochs))
    if hasattr(criterion, 'rank_bce_gain'):
        criterion.rank_bce_gain = rank_base_gain * rank_scale
    return criterion.hard_negative_gain

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    set_pretrained_backbone_bn_eval(model, cfg)
    update_epoch_loss_gains(criterion, cfg, epoch)
    total_loss = box_l = cls_l = dfl_l = smooth_l1_l = feat_l = hard_neg_l = rank_bce_l = 0.0
    show_bar = False  # prevents Jupyter IOStream.flush timeout on long runs
    pbar = tqdm(loader, desc=f'Epoch {epoch:03d}', leave=False, disable=not show_bar, mininterval=10)
    accum = max(int(getattr(cfg, 'accumulate', 1)), 1)
    optimizer.zero_grad(set_to_none=True)

    for ni, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        if getattr(cfg, 'channels_last', False) and device.startswith('cuda'):
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.startswith('cuda'))):
            raw = model(imgs)
            loss, parts = criterion(raw, targets, cfg.img_size)
            loss_for_backward = loss / accum

        scaler.scale(loss_for_backward).backward()

        is_update_step = ((ni + 1) % accum == 0) or ((ni + 1) == len(loader))
        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg, 'grad_clip', 10.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        box_l += parts['box']
        cls_l += parts['cls']
        dfl_l += parts['dfl']
        smooth_l1_l += parts.get('smooth_l1', 0.0)
        feat_l += parts.get('feat', 0.0)
        hard_neg_l += parts.get('hard_neg', 0.0)
        rank_bce_l += parts.get('rank_bce', 0.0)

    n = max(len(loader), 1)
    return {'total': total_loss/n, 'box': box_l/n, 'cls': cls_l/n, 'dfl': dfl_l/n,
            'smooth_l1': smooth_l1_l/n, 'feat': feat_l/n, 'hard_neg': hard_neg_l/n,
            'rank_bce': rank_bce_l/n}

@torch.no_grad()
def validate(model, loader, criterion, device):
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    total_loss = box_l = cls_l = dfl_l = smooth_l1_l = feat_l = hard_neg_l = rank_bce_l = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        if getattr(cfg, 'channels_last', False) and device.startswith('cuda'):
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        raw = model(imgs)
        loss, parts = criterion(raw, targets, cfg.img_size)
        total_loss += loss.item()
        box_l += parts['box']
        cls_l += parts['cls']
        dfl_l += parts['dfl']
        smooth_l1_l += parts.get('smooth_l1', 0.0)
        feat_l += parts.get('feat', 0.0)
        hard_neg_l += parts.get('hard_neg', 0.0)
        rank_bce_l += parts.get('rank_bce', 0.0)
    model.train(was_training)
    n = max(len(loader), 1)
    return {'total': total_loss/n, 'box': box_l/n, 'cls': cls_l/n, 'dfl': dfl_l/n,
            'smooth_l1': smooth_l1_l/n, 'feat': feat_l/n, 'hard_neg': hard_neg_l/n,
            'rank_bce': rank_bce_l/n}

def save_topk_snapshot(epoch, score, ema_model, cfg, ckpt):
    """Save top-k EMA snapshots by the selected validation score."""
    k = int(getattr(cfg, 'save_top_k', 0))
    if k <= 0 or score is None:
        return
    path = os.path.join(cfg.save_dir, f'topk_epoch{epoch:03d}_map{score:.4f}.pt')
    torch.save({'epoch': epoch, 'model': ema_model.state_dict(), 'cfg': cfg.__dict__, 'map5095': score}, path)
    topk_records.append({'epoch': epoch, 'score': float(score), 'path': path})
    topk_records.sort(key=lambda x: x['score'], reverse=True)
    while len(topk_records) > k:
        old = topk_records.pop(-1)
        try:
            if os.path.exists(old['path']):
                os.remove(old['path'])
        except Exception as e:
            print(f'Warning: could not remove old top-k checkpoint {old["path"]}: {e}')

def load_state_dict_any(path, device):
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    return obj

def resolve_eval_checkpoint_path(cfg):
    candidates = [
        os.path.join(cfg.save_dir, 'best_weights.pt'),
        os.path.join(cfg.save_dir, 'best.pt'),
        os.path.join(cfg.models_dir, 'best.pth'),
        'best.pth',
        os.path.join(cfg.save_dir, 'last_weights.pt'),
        os.path.join(cfg.save_dir, 'last.pt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_checkpoint_into_model_if_available(model, cfg, label='model'):
    path = resolve_eval_checkpoint_path(cfg)
    if path is None:
        print(f'No saved checkpoint found in {cfg.save_dir}; using current {label} in memory.')
        model.eval()
        return None
    model.load_state_dict(load_state_dict_any(path, cfg.device), strict=True)
    model.eval()
    print(f'Loaded {label}: {path}')
    return path

def load_fresh_model_from_weights(path, cfg):
    m = ScratchCSP(
        num_classes=cfg.num_classes, width=cfg.width_mult,
        depth=cfg.depth_mult, reg_max=cfg.reg_max, strides=cfg.strides,
        feature_dim=getattr(cfg, 'feature_dim', 0),
        backbone_name=getattr(cfg, 'backbone_name', 'scratch'),
        pretrained_backbone=getattr(cfg, 'pretrained_backbone', False),
        freeze_backbone_stem=getattr(cfg, 'freeze_backbone_stem', True),
        backbone_compress_ratio=getattr(cfg, 'backbone_compress_ratio', 0.5),
    ).to(cfg.device)
    m.load_state_dict(load_state_dict_any(path, cfg.device), strict=True)
    m.eval()
    return m

@torch.no_grad()
def sweep_postprocess(model, loader, cfg, label='model'):
    """Tune NMS IoU/pre-topk on validation. This often helps dense public-object images."""
    if not getattr(cfg, 'run_post_sweep', False):
        return None

    orig = (cfg.iou_thres, cfg.nms_pre_topk, cfg.max_det)
    candidates_iou = [0.50, 0.55, 0.60, 0.65]
    candidates_topk = [6000, 8000]
    best = {'map5095': -1.0, 'map50': -1.0, 'iou': None, 'topk': None, 'max_det': None}

    print(f'\nPostprocess sweep for {label}:')
    for iou in candidates_iou:
        for topk in candidates_topk:
            cfg.iou_thres = iou
            cfg.nms_pre_topk = topk
            cfg.max_det = 100
            m50, m95, _ = evaluate_map(model, loader, cfg, verbose=False, tta=False)
            print(f'  iou={iou:.2f} pre_topk={topk:5d} -> mAP50={m50:.4f}')
            if m95 > best['map5095']:
                best = {'map5095': float(m95), 'map50': float(m50), 'iou': iou, 'topk': topk, 'max_det': 100}

    cfg.iou_thres, cfg.nms_pre_topk, cfg.max_det = best['iou'], best['topk'], best['max_det']
    print(f'Best postprocess for {label}: iou={cfg.iou_thres}, pre_topk={cfg.nms_pre_topk}, '
          f'mAP50={best["map50"]:.4f}')
    return best

class PredictionEnsemble(nn.Module):
    """Concatenate predictions from several scratch checkpoints, then common NMS handles duplicates."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for m in self.models:
            m.eval()

    def forward(self, x):
        preds = [m(x) for m in self.models]
        return torch.cat(preds, dim=1)

def preprocess_for_inference(img_bgr, img_size=640):
    """Letterbox single BGR image. Returns tensor, scale, pad, original shape."""
    h, w = img_bgr.shape[:2]
    scale = min(img_size / h, img_size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    pad_x = (img_size - nw) // 2
    pad_y = (img_size - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, scale, (pad_x, pad_y), (h, w)

def dets_to_original(dets, scale, pad, orig_shape):
    """Convert detections [x1,y1,x2,y2,score,cls] from square letterbox coords to original image coords."""
    if len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    dets = dets.copy().astype(np.float32)
    pad_x, pad_y = pad
    h, w = orig_shape
    dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
    dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale
    dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w)
    dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h)
    bw = dets[:, 2] - dets[:, 0]
    bh = dets[:, 3] - dets[:, 1]
    keep = (bw >= 1) & (bh >= 1)
    return dets[keep]

def draw_predictions_original(img_rgb, detections, class_names, title='Predictions'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_rgb)
    ax.axis('off')
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)
        color = COLORS[cls_id % len(COLORS)]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 5, 0), f'{class_names[cls_id]} {score:.2f}',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor='none'))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def predict_image(model, img_path: str, cfg: Config, conf=None, visualize=True, mode='infer'):
    """Predict one image.

    mode='infer'  : dÃ¹ng ngÆ°á»¡ng sáº¡ch Ä‘á»ƒ váº½/demo, trÃ¡nh hÃ ng loáº¡t box score 0.00.
    mode='export' : dÃ¹ng ngÆ°á»¡ng export Ä‘á»ƒ táº¡o val_predictions.json cho official AP.
    """
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f'Cannot read {img_path}'
    tensor, scale, pad, orig_shape = preprocess_for_inference(img_bgr, cfg.img_size)
    tensor = tensor.to(cfg.device)
    if getattr(cfg, 'channels_last', False) and cfg.device.startswith('cuda'):
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    model.eval()
    pred = model(tensor)

    if conf is not None:
        conf_thres = conf
        class_thresholds = None
        class_ious = None
    elif str(mode).lower() == 'export':
        conf_thres = getattr(cfg, "export_conf_thres", cfg.conf_thres)
        class_thresholds = (
            getattr(cfg, "class_conf_thres", None)
            if getattr(cfg, "export_use_class_conf_thres", False)
            else None
        )
        class_ious = (
            getattr(cfg, "class_iou_thres", None)
            if getattr(cfg, "export_use_class_conf_thres", False)
            else None
        )
    else:
        conf_thres = getattr(cfg, "infer_conf_thres", 0.25)
        class_thresholds = getattr(cfg, "class_conf_thres", None)
        class_ious = getattr(cfg, "class_iou_thres", None)

    dets = postprocess(
        pred, conf_thres, cfg.iou_thres, cfg.max_det,
        nms_pre_topk=getattr(cfg, 'nms_pre_topk', 1500),
        class_conf_thres=class_thresholds,
        class_iou_thres=class_ious
    )
    dets_np = dets[0].detach().cpu().numpy() if len(dets[0]) else np.zeros((0, 6), dtype=np.float32)
    dets_orig = dets_to_original(dets_np, scale, pad, orig_shape)
    if visualize:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        draw_predictions_original(img_rgb, dets_orig, cfg.class_names, title=os.path.basename(img_path))
    return dets_orig

@torch.no_grad()
def visualize_val_samples(model, dataset, cfg: Config, n=8):
    model.eval()
    indices = np.linspace(0, len(dataset)-1, min(n, len(dataset))).astype(int).tolist()
    cols = 4
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, idx in zip(axes, indices):
        rec = dataset.records[idx]
        img_bgr = cv2.imread(rec['path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dets = predict_image(model, rec['path'], cfg, conf=None, visualize=False, mode='infer')
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title(rec['image_id'])
        for det in dets:
            x1,y1,x2,y2,score,cls_id = det
            color = COLORS[int(cls_id) % len(COLORS)]
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor=color, facecolor='none'))
            ax.text(x1, max(y1-3, 0), f'{cfg.class_names[int(cls_id)]} {score:.2f}',
                    color='white', fontsize=7, bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')
    plt.suptitle('Validation predictions in original image coordinates')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, 'val_predictions_original_coords.png'), dpi=150)
    plt.show()

@torch.no_grad()
def export_predictions_json(model, image_dir: str, output_json: str, cfg: Config, conf=None):
    """Export assignment-format predictions for every image in image_dir."""
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    predictions = []
    model.eval()
    for name in tqdm(image_files, desc='Export predictions'):
        path = os.path.join(image_dir, name)
        dets = predict_image(model, path, cfg, conf=conf, visualize=False, mode='export')
        boxes = []
        for x1, y1, x2, y2, score, cls_id in dets:
            boxes.append({
                'class': cfg.class_names[int(cls_id)],
                'confidence': float(np.clip(score, 0.0, 1.0)),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        predictions.append({'image_id': name, 'boxes': boxes})
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(predictions)} image predictions to: {output_json}')
    return predictions

def run_assignment_evaluator(gt_json, pred_json, out_json, cfg: Config):
    eval_script = os.path.join(cfg.data_root, 'tools', 'evaluate_predictions.py')
    if not os.path.exists(eval_script):
        print('Evaluator script not found:', eval_script)
        return None
    cmd = [
        sys.executable, eval_script,
        '--ground_truth', gt_json,
        '--predictions', pred_json,
        '--output', out_json,
        '--iou_threshold', '0.5',
        '--max_detections_per_image', str(min(int(getattr(cfg, 'max_det', 100)), 100)),
    ]
    try:
        with open(eval_script, 'r', encoding='utf-8') as f:
            evaluator_src = f.read()
        if '--confusion_confidence_threshold' in evaluator_src:
            cmd += ['--confusion_confidence_threshold', str(getattr(cfg, 'cm_conf_thres', 0.0))]
    except Exception:
        pass
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    with open(out_json, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
os.makedirs(cfg.models_dir, exist_ok=True)

ema = None
scaler = None
topk_records = []
COLORS = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#2ecc71']


def create_model(cfg_obj=None):
    cfg_obj = cfg_obj or cfg
    model = ScratchCSP(
        num_classes=cfg_obj.num_classes,
        width=cfg_obj.width_mult,
        depth=cfg_obj.depth_mult,
        reg_max=cfg_obj.reg_max,
        strides=cfg_obj.strides,
        feature_dim=getattr(cfg_obj, 'feature_dim', 0),
        backbone_name=getattr(cfg_obj, 'backbone_name', 'scratch'),
        pretrained_backbone=getattr(cfg_obj, 'pretrained_backbone', False),
        freeze_backbone_stem=getattr(cfg_obj, 'freeze_backbone_stem', True),
        backbone_compress_ratio=getattr(cfg_obj, 'backbone_compress_ratio', 0.5),
    ).to(cfg_obj.device)
    if getattr(cfg_obj, 'channels_last', False) and str(cfg_obj.device).startswith('cuda'):
        model = model.to(memory_format=torch.channels_last)
    return model


def create_criterion(model, cfg_obj=None):
    cfg_obj = cfg_obj or cfg
    criterion = ScratchCSPLoss(
        model,
        box_gain=cfg_obj.box_gain,
        cls_gain=cfg_obj.cls_gain,
        dfl_gain=cfg_obj.dfl_gain,
        smooth_l1_gain=getattr(cfg_obj, 'smooth_l1_gain', 0.0),
        smooth_l1_beta=getattr(cfg_obj, 'smooth_l1_beta', 0.05),
        class_weights=getattr(cfg_obj, 'class_weights', None),
        assigner_topk=getattr(cfg_obj, 'assigner_topk', 10),
        target_score_floor=getattr(cfg_obj, 'target_score_floor', 0.0),
        cls_focal_gamma=getattr(cfg_obj, 'cls_focal_gamma', 0.0),
        cls_focal_alpha=getattr(cfg_obj, 'cls_focal_alpha', -1.0),
        positive_weight_only=getattr(cfg_obj, 'positive_weight_only', True),
        cls_loss_type=getattr(cfg_obj, 'cls_loss_type', 'bce'),
        normalizer_mode=getattr(cfg_obj, 'normalizer_mode', 'positive'),
        feature_gain=getattr(cfg_obj, 'feature_gain', 0.0),
        feature_temperature=getattr(cfg_obj, 'feature_temperature', 0.20),
        feature_max_pos=getattr(cfg_obj, 'feature_max_pos', 256),
        chair_class_id=getattr(cfg_obj, 'focus_class_id', 4),
        chair_assigner_topk=getattr(cfg_obj, 'chair_assigner_topk', 0),
        hard_negative_mining=getattr(cfg_obj, 'hard_negative_mining', False),
        hard_negative_conf=getattr(cfg_obj, 'hard_negative_conf', 0.28),
        hard_negative_topk=getattr(cfg_obj, 'hard_negative_topk', 256),
        hard_negative_gain=getattr(cfg_obj, 'hard_negative_gain', 0.0),
        hard_negative_class_weights=getattr(cfg_obj, 'hard_negative_class_weights', None),
        chair_hard_negative_mining=getattr(cfg_obj, 'chair_hard_negative_mining', False),
        chair_hard_negative_conf=getattr(cfg_obj, 'chair_hard_negative_conf', 0.20),
        chair_hard_negative_topk=getattr(cfg_obj, 'chair_hard_negative_topk', 256),
        chair_hard_negative_gain=getattr(cfg_obj, 'chair_hard_negative_gain', 0.0),
        rank_bce_gain=getattr(cfg_obj, 'rank_bce_gain', 0.0),
        rank_bce_classes=getattr(cfg_obj, 'rank_bce_classes', None),
        rank_bce_margin=getattr(cfg_obj, 'rank_bce_margin', 0.15),
        rank_bce_pos_topk=getattr(cfg_obj, 'rank_bce_pos_topk', 96),
        rank_bce_neg_topk=getattr(cfg_obj, 'rank_bce_neg_topk', 256),
        rank_bce_min_pos_quality=getattr(cfg_obj, 'rank_bce_min_pos_quality', 0.20),
    )
    return criterion.to(cfg_obj.device)


def configure_paths(cfg_obj, train_data=None, val_data=None, image_dir=None, val_image_dir=None):
    # CLI arguments already point to assignment-relative paths, so keep data_root as current directory.
    cfg_obj.data_root = '.'
    if train_data is not None:
        cfg_obj.train_ann = train_data
    if val_data is not None:
        cfg_obj.val_ann = val_data
    if image_dir is not None:
        cfg_obj.train_img = image_dir
    if val_image_dir is not None:
        cfg_obj.val_img = val_image_dir
    return cfg_obj


def load_model_for_inference(checkpoint_path, cfg_obj=None):
    cfg_obj = cfg_obj or cfg
    model = create_model(cfg_obj)
    state = load_state_dict_any(checkpoint_path, cfg_obj.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if getattr(cfg_obj, 'channels_last', False) and str(cfg_obj.device).startswith('cuda'):
        model = model.to(memory_format=torch.channels_last)
    return model


