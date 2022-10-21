import torch
from datasets import load_metric
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup


class BaseModel(LightningModule):
    def __init__(self, config, tokenizer):
        """method used to define our model parameters"""
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_pretrained
        )
        self.accuracy = load_metric("accuracy")
        # self.save_hyperparameters()

    def configure_optimizers(self):
        # optimizer
        optimizer = AdamW(self.parameters(), lr=float(self.cfg.lr))

        # scheduler with warmup
        num_devices = (
            torch.cuda.device_count()
            if self.trainer.devices == -1
            else int(self.trainer.devices)
        )
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.cfg.accumulate_grad_batches
            // num_devices
            * self.cfg.epoch
        )
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            ),
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
            labels=batch["labels"],
        )
        preds = torch.argmax(output.logits, dim=1)
        loss = output.loss.mean()
        acc = self.accuracy.compute(
            references=batch["labels"].data, predictions=preds.data
        )
        return preds, loss, acc
