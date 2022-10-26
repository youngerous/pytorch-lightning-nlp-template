import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from config.base import load_config
from loader import BaseDataModule
from model import BaseModel


def set_experiment_name(cfg):
    # set customized experiment name (useful when tuning hparams)
    name = f"""
    {cfg.exp_name}-ep{cfg.epoch}-lr{cfg.lr}-bsz{cfg.batch_size}
    """.strip()
    return name


def run(cfg):
    pl.seed_everything(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_pretrained)
    datamodule = BaseDataModule(config=cfg, tokenizer=tokenizer)
    model = BaseModel(config=cfg, tokenizer=tokenizer)

    exp_name = set_experiment_name(cfg)
    wandb_logger = WandbLogger(
        name=exp_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
    )
    ckpt_pth = os.path.join(cfg.checkpoint_dirpath, exp_name)
    callbacks = (
        [
            ModelCheckpoint(
                dirpath=ckpt_pth,
                monitor=cfg.checkpoint_monitor,
                save_top_k=cfg.checkpoint_save_top_k,
                filename=cfg.checkpoint_filename,
            ),
            EarlyStopping(
                monitor=cfg.checkpoint_monitor,
                patience=cfg.earlystop_patience,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        if not cfg.do_test_only
        else None
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=os.getcwd(),
        devices=cfg.gpu_devices,
        accelerator=cfg.gpu_accelerator,
        strategy=cfg.gpu_strategy,
        amp_backend=cfg.backend,
        gradient_clip_val=cfg.gradient_clip_val,
        max_epochs=cfg.epoch,
        max_steps=cfg.max_steps,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.validation_interval,
        log_every_n_steps=cfg.log_interval,
    )

    if not cfg.do_test_only:
        trainer.fit(model, datamodule=datamodule)

    if not cfg.do_train_only:
        if cfg.model_load_ckpt_pth is not None:
            model = BaseModel.load_from_checkpoint(
                cfg.model_load_ckpt_pth, config=cfg, tokenizer=tokenizer
            )
            trainer.test(model, datamodule=datamodule)
        else:
            trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
