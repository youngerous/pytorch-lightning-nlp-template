import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--seed", type=int, default=42)

    ## WANDB
    parser.add_argument("--wandb_project", type=str, default="lightning-template")
    parser.add_argument("--wandb_entity", type=str, default="youngerous")

    ## DATASET
    parser.add_argument("--dataset_name", type=str, default="imdb")

    ## GPU
    parser.add_argument(
        "--gpu_accelerator", type=str, default="gpu", help="Set 'cpu' for debugging"
    )
    parser.add_argument(
        "--gpu_strategy", type=str, default="ddp", help="Options: null, ddp, dp, ..."
    )
    parser.add_argument("--gpu_devices", type=int, default=-1)

    ## MODEL
    parser.add_argument("--model_pretrained", type=str, default="bert-base-uncased")
    parser.add_argument("--model_load_ckpt_pth", help="Default: none")

    ## CHECKPOINT
    parser.add_argument("--checkpoint_dirpath", type=str, default="src/checkpoints")
    parser.add_argument(
        "--checkpoint_filename", type=str, default="ckpt-{epoch:03d}-{val_loss:.5f}"
    )
    parser.add_argument("--checkpoint_save_top_k", type=int, default=2)
    parser.add_argument("--checkpoint_monitor", type=str, default="val_loss")

    ## MODE
    parser.add_argument("--do_train_only", type=str2bool, default=False)
    parser.add_argument("--do_test_only", type=str2bool, default=False)

    ## TRAIN
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="native")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=-1, help="Default: disabled")
    parser.add_argument(
        "--gradient_clip_val", type=float, default=0.0, help="Default: not clipping"
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    ## VALIDATION
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--earlystop_patience", type=int, default=3)

    ## LOG
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    return args
