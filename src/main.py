import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from loader import BaseDataModule
from model import BaseModel


def set_experiment_name(cfg):
    # set customized experiment name (useful when tuning hparams)
    name = f"""
    {cfg['EXP_NAME']}-ep{cfg['TRAIN']['epoch']}
    """.strip()
    return name


def run(cfg):
    pl.seed_everything(cfg["SEED"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL"]["pretrained"])
    datamodule = BaseDataModule(config=cfg, tokenizer=tokenizer)
    model = BaseModel(config=cfg, tokenizer=tokenizer)

    exp_name = set_experiment_name(cfg)
    wandb_logger = WandbLogger(
        name=exp_name,
        project=cfg["WANDB"]["project"],
        entity=cfg["WANDB"]["entity"],
    )
    ckpt_pth = os.path.join(cfg["MODEL"]["CHECKPOINT"]["dirpath"], exp_name)
    callbacks = (
        [
            ModelCheckpoint(
                dirpath=ckpt_pth,
                monitor=cfg["MODEL"]["CHECKPOINT"]["monitor"],
                save_top_k=cfg["MODEL"]["CHECKPOINT"]["save_top_k"],
                filename=cfg["MODEL"]["CHECKPOINT"]["filename"],
            ),
            EarlyStopping(
                monitor=cfg["MODEL"]["CHECKPOINT"]["monitor"],
                patience=cfg["VALIDATION"]["earlystop_patience"],
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        if not cfg["MODE"]["do_test_only"]
        else None
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=os.getcwd(),
        devices=cfg["GPU"]["devices"],
        accelerator=cfg["GPU"]["accelerator"],
        strategy=cfg["GPU"]["strategy"],
        amp_backend=cfg["TRAIN"]["backend"],
        gradient_clip_val=cfg["TRAIN"]["gradient_clip_val"],
        max_epochs=cfg["TRAIN"]["epoch"],
        max_steps=cfg["TRAIN"]["max_steps"],
        precision=cfg["TRAIN"]["precision"],
        accumulate_grad_batches=cfg["TRAIN"]["accumulate_grad_batches"],
        check_val_every_n_epoch=cfg["VALIDATION"]["interval"],
        log_every_n_steps=cfg["LOG"]["interval"],
    )

    if not cfg["MODE"]["do_test_only"]:
        trainer.fit(model, datamodule=datamodule)

    if not cfg["MODE"]["do_train_only"]:
        if cfg["MODEL"]["load_ckpt_pth"] is not None:
            model = BaseModel.load_from_checkpoint(
                cfg["MODEL"]["load_ckpt_pth"], config=cfg, tokenizer=tokenizer
            )
            trainer.test(model, datamodule=datamodule)
        else:
            trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)
    run(cfg)
