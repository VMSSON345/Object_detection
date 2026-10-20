import argparse
import os
import shutil
import time

import torch

from utils import detector as det


def parse_args():
    parser = argparse.ArgumentParser(description='Train the ScratchCSP object detector.')
    parser.add_argument('--train_data', default='./public/annotations/train.json')
    parser.add_argument('--val_data', default='./public/annotations/val.json')
    parser.add_argument('--image_dir', default='./public/train/images')
    parser.add_argument('--val_image_dir', default='./public/val/images')
    parser.add_argument('--checkpoint_dir', default='./models/')
    parser.add_argument('--save_dir', default='./runs')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', default='')
    parser.add_argument('--lr0', type=float, default=None)
    parser.add_argument('--no_pretrained_backbone', action='store_true')
    return parser.parse_args()


def build_config(args):
    cfg = det.Config()
    det.configure_paths(
        cfg,
        train_data=args.train_data,
        val_data=args.val_data,
        image_dir=args.image_dir,
        val_image_dir=args.val_image_dir,
    )
    cfg.models_dir = args.checkpoint_dir
    cfg.save_dir = args.save_dir
    cfg.resume_ckpt = args.resume

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.val_batch_size is not None:
        cfg.val_batch_size = args.val_batch_size
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.lr0 is not None:
        cfg.lr0 = args.lr0
    if args.device != 'auto':
        cfg.device = args.device
    if args.no_pretrained_backbone:
        cfg.pretrained_backbone = False

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)
    return cfg


def save_training_curves(history, cfg):
    if det.plt is None:
        return
    try:
        fig, (ax1, ax2, ax3) = det.plt.subplots(1, 3, figsize=(18, 4))
        epochs_x = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs_x, history['train_loss'], label='Train Loss', color='#e74c3c')
        ax1.plot(epochs_x, history['val_loss'], label='Val Loss', color='#2ecc71')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_x, history['lr'], color='#3498db')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        if history['map50']:
            mx = [x[0] for x in history['map50']]
            my50 = [x[1] for x in history['map50']]
            my95 = [x[1] for x in history['map5095']]
            ax3.plot(mx, my50, marker='o', label='mAP@0.5')
            ax3.plot(mx, my95, marker='o', label='mAP@0.5:0.95')
        ax3.set_title('mAP')
        ax3.set_xlabel('Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        det.plt.tight_layout()
        det.plt.savefig(os.path.join(cfg.save_dir, 'training_curves.png'), dpi=150)
        det.plt.close(fig)
    except Exception as exc:
        print(f'Could not save training curves: {exc}')


def main():
    args = parse_args()
    cfg = build_config(args)
    det.cfg = cfg
    det.seed_everything(cfg.seed)

    train_loader, val_loader = det.build_dataloaders(cfg)
    counts, weights = det.estimate_class_weights_from_records(
        train_loader.dataset.records,
        cfg.num_classes,
        power=cfg.class_balance_power,
        clip=cfg.class_weight_clip,
        manual_multipliers=getattr(cfg, 'manual_class_weight_multipliers', None),
    )
    cfg.class_weights = weights
    print('Class counts:', {name: int(c) for name, c in zip(cfg.class_names, counts)})
    print('Class weights:', {name: round(float(w), 3) for name, w in zip(cfg.class_names, weights)})

    model = det.create_model(cfg)
    criterion = det.create_criterion(model, cfg)
    optimizer, scheduler = det.build_optimizer_scheduler(model, cfg, len(train_loader))
    ema = det.ModelEMA(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and str(cfg.device).startswith('cuda')))

    det.ema = ema
    det.scaler = scaler
    det.topk_records = []

    history = {
        'train_loss': [], 'val_loss': [], 'box': [], 'cls': [], 'dfl': [], 'feat': [],
        'map50': [], 'map5095': [], 'lr': [],
    }
    best_score = -1.0
    best_map = -1.0
    best_val = float('inf')
    best_epoch = 0
    no_improve_evals = 0
    start_epoch = 1

    if cfg.resume_ckpt:
        if os.path.exists(cfg.resume_ckpt):
            ckpt = det.load_checkpoint_for_resume(cfg.resume_ckpt, model, ema, optimizer, scheduler, scaler, cfg)
            history = ckpt.get('history', history)
            best_score = float(ckpt.get('best_score', best_score))
            best_map = float(ckpt.get('best_map', best_map))
            best_val = float(ckpt.get('best_val', best_val))
            best_epoch = int(ckpt.get('best_epoch', ckpt.get('epoch', 0)))
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            ema.updates = ckpt.get('ema_updates', (start_epoch - 1) * len(train_loader))
            print(f'Resumed from {cfg.resume_ckpt}')
        else:
            print(f'WARNING: resume checkpoint not found: {cfg.resume_ckpt}')

    start = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        mosaic_mid_epoch = max(1, int(round(cfg.epochs * getattr(cfg, 'mosaic_mid_epoch_frac', 0.60))))
        close_mosaic_epoch = max(1, cfg.epochs - cfg.close_mosaic_epochs + 1)
        if epoch == mosaic_mid_epoch and epoch < close_mosaic_epoch:
            train_loader.dataset.mosaic_prob = getattr(cfg, 'mosaic_mid_prob', train_loader.dataset.mosaic_prob)
        if epoch == close_mosaic_epoch and hasattr(train_loader.dataset, 'close_mosaic'):
            train_loader.dataset.close_mosaic()

        train_stats = det.train_one_epoch(model, train_loader, optimizer, scheduler, criterion, cfg.device, epoch)
        val_stats = det.validate(ema.ema, val_loader, criterion, cfg.device)

        history['train_loss'].append(train_stats['total'])
        history['val_loss'].append(val_stats['total'])
        history['box'].append(val_stats['box'])
        history['cls'].append(val_stats['cls'])
        history['dfl'].append(val_stats['dfl'])
        history['feat'].append(val_stats.get('feat', 0.0))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        map50 = map5095 = None
        run_map = (
            epoch == 1
            or epoch % cfg.eval_interval == 0
            or epoch >= getattr(cfg, 'full_eval_start_epoch', cfg.epochs + 1)
            or epoch == cfg.epochs
        )
        if run_map:
            map50, map5095, _ = det.evaluate_map(ema.ema, val_loader, cfg, verbose=False)
            map50, map5095 = float(map50), float(map5095)
            history['map50'].append((epoch, map50))
            history['map5095'].append((epoch, map5095))

        ckpt = {
            'epoch': epoch,
            'model_train': model.state_dict(),
            'ema': ema.ema.state_dict(),
            'ema_updates': ema.updates,
            'model': ema.ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'history': history,
            'cfg': cfg.__dict__,
            'best_score': best_score,
            'best_map': best_map,
            'best_val': best_val,
            'best_epoch': best_epoch,
        }
        torch.save(ckpt, os.path.join(cfg.save_dir, 'last.pt'))
        torch.save(ema.ema.state_dict(), os.path.join(cfg.save_dir, 'last_weights.pt'))

        if map5095 is not None:
            metric_mode = getattr(cfg, 'best_metric', 'map5095').lower()
            if metric_mode == 'map5095':
                current_score = map5095
            elif metric_mode == 'combo':
                current_score = 0.65 * map50 + 0.35 * map5095
            else:
                current_score = map50
            det.save_topk_snapshot(epoch, current_score, ema.ema, cfg, ckpt)
        else:
            current_score = None

        improved = False
        if current_score is not None and current_score > best_score:
            best_score = current_score
            best_map = map5095
            improved = True
        elif current_score is None and val_stats['total'] < best_val and best_score < 0:
            improved = True

        if val_stats['total'] < best_val:
            best_val = val_stats['total']

        if improved:
            best_epoch = epoch
            no_improve_evals = 0
            ckpt['best_score'] = best_score
            ckpt['best_map'] = best_map
            ckpt['best_val'] = best_val
            ckpt['best_epoch'] = best_epoch
            torch.save(ckpt, os.path.join(cfg.save_dir, 'best.pt'))
            torch.save(ema.ema.state_dict(), os.path.join(cfg.save_dir, 'best_weights.pt'))
            torch.save(ema.ema.state_dict(), os.path.join(cfg.models_dir, 'best.pth'))
            mark = ' * saved best.pth'
        else:
            mark = ''
            if map5095 is not None:
                no_improve_evals += 1

        map_text = f' | mAP50={map50:.4f} mAP50-95={map5095:.4f}' if map50 is not None else ''
        elapsed = (time.time() - start) / 60
        print(
            f'Epoch {epoch:03d}/{cfg.epochs} | train={train_stats["total"]:.4f} '
            f'val={val_stats["total"]:.4f} lr={optimizer.param_groups[0]["lr"]:.2e}'
            f'{map_text} | {elapsed:.1f} min{mark}',
            flush=True,
        )

        if map5095 is not None and no_improve_evals >= cfg.patience_evals:
            print(f'Early stopping. Best epoch={best_epoch}, best_score={best_score:.4f}')
            break

    best_weights = os.path.join(cfg.save_dir, 'best_weights.pt')
    final_weight = os.path.join(cfg.models_dir, 'best.pth')
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, final_weight)
    elif not os.path.exists(final_weight):
        torch.save(ema.ema.state_dict(), final_weight)
    save_training_curves(history, cfg)
    print(f'Training done. Best score={best_score:.4f}, best mAP@0.5:0.95={best_map:.4f}')
    print(f'Saved submission checkpoint: {final_weight}')


if __name__ == '__main__':
    main()

