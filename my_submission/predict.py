import argparse
import os
import urllib.request

from utils import detector as det

WEIGHT_URL = "https://huggingface.co/VMSON/my_submission_weights/resolve/main/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(description='Run object detection inference.')
    parser.add_argument('--image_dir', required=True, help='Directory containing images to predict.')
    parser.add_argument('--output', required=True, help='Path to write predictions.json.')
    parser.add_argument('--checkpoint', default='./models/best.pth', help='Model checkpoint path.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--iou', type=float, default=None)
    parser.add_argument('--max_det', type=int, default=None)
    parser.add_argument('--nms_pre_topk', type=int, default=None)
    return parser.parse_args()


def download_checkpoint():
    checkpoint_path = './models/best.pth'

    os.makedirs('./models', exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        print("Downloading weights from Hugging Face...")

        urllib.request.urlretrieve(
            WEIGHT_URL,
            checkpoint_path
        )

        print("Download complete.")

    return checkpoint_path


def resolve_checkpoint(path):
    candidates = [
        path,
        './models/best.pth',
        './runs/best_weights.pt',
        './runs/best.pt',
        './runs/last_weights.pt',
        './runs/last.pt',
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            print(f'Using checkpoint: {candidate}')
            return candidate

    return download_checkpoint()


def main():
    args = parse_args()

    cfg = det.Config()
    cfg.pretrained_backbone = False

    if args.device != 'auto':
        cfg.device = args.device

    if args.img_size is not None:
        cfg.img_size = args.img_size

    if args.iou is not None:
        cfg.iou_thres = args.iou

    if args.max_det is not None:
        cfg.max_det = args.max_det

    if args.nms_pre_topk is not None:
        cfg.nms_pre_topk = args.nms_pre_topk

    det.cfg = cfg

    checkpoint = resolve_checkpoint(args.checkpoint)

    model = det.load_model_for_inference(
        checkpoint,
        cfg
    )

    det.export_predictions_json(
        model=model,
        image_dir=args.image_dir,
        output_json=args.output,
        cfg=cfg,
        conf=args.conf,
    )


if __name__ == '__main__':
    main()