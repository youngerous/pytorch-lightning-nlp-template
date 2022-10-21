import pytorch_lightning as pl
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class IMDB(Dataset):
    def __init__(self, tok, text, label):
        self.tok = tok
        self.text = text
        self.label = label

        assert len(self.text) == len(self.label)
        print(f"Load {len(self.label)} data.")

    def __getitem__(self, idx):
        src = self.tok(
            self.text[idx], truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": src["input_ids"],
            "token_type_ids": src["token_type_ids"],
            "attention_mask": src["attention_mask"],
            "labels": torch.tensor(self.label[idx]),
        }

    def __len__(self):
        return len(self.label)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer

    def prepare_data(self):
        """Only called from the main process for downloading dataset"""
        load_dataset(self.cfg.dataset_name, split="train")
        load_dataset(self.cfg.dataset_name, split="test")

    def setup(self, stage: str):
        if stage == "fit":
            dset = load_dataset(self.cfg.dataset_name, split="train")
            trn_text, val_text, trn_label, val_label = train_test_split(
                dset["text"], dset["label"], test_size=0.1
            )
            self.trn_dset = IMDB(self.tokenizer, trn_text, trn_label)
            self.val_dset = IMDB(self.tokenizer, val_text, val_label)

        if stage == "test":
            dset = load_dataset(self.cfg.dataset_name, split="test")
            self.tst_dset = IMDB(self.tokenizer, dset["text"], dset["label"])

    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
