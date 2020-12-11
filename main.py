import os
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
import hydra

from dataset import RIIDDataset, collate_fn
from model import RIIDDTransformerModel
from preprocessing import get_users, generate_h5
from utils import get_wd

SEED = 69
seed_everything(SEED)


@hydra.main(config_name="config")
def train(cfg) -> None:

    learning_rate = cfg["learning_rate"]
    emb_dim = cfg["emb_dim"]
    dropout = cfg["dropout"]
    n_heads = cfg["n_heads"]
    n_encoder_layers = cfg["n_encoder_layers"]
    n_decoder_layers = cfg["n_decoder_layers"]
    dim_feedforward = cfg["dim_feedforward"]
    batch_size = cfg["batch_size"]
    num_user_train = cfg["num_user_train"]
    num_user_val = cfg["num_user_val"]
    max_window_size = cfg["max_window_size"]
    num_workers = cfg["num_workers"]

    user_ids, user_weights = get_users()

    generate_h5()

    dataset = RIIDDataset(
        user_mapping=user_ids,
        hdf5_file="feats.h5",
        window_size=max_window_size,
    )

    # create split
    train_dataset, val_dataset, _ = random_split(
        dataset,
        [
            num_user_train,
            num_user_val,
            len(dataset) - num_user_train - num_user_val,
        ],
    )

    # Init DataLoader from RIIID Dataset subset
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # Weighted sampler
        sampler=torch.utils.data.WeightedRandomSampler(
            weights=user_weights[train_dataset.indices],
            num_samples=len(train_dataset),
        ),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # if GPU then pin memory for perf
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=512,
        collate_fn=collate_fn,
        sampler=torch.utils.data.WeightedRandomSampler(
            weights=user_weights[val_dataset.indices],
            num_samples=len(val_dataset),
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Init our model
    model = RIIDDTransformerModel(
        learning_rate=learning_rate,
        emb_dim=emb_dim,  # embedding dimension - this is for everything
        dropout=dropout,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        dim_feedforward=dim_feedforward,
        batch_size=batch_size,
        num_user_train=num_user_train,
        num_user_val=num_user_val,
        max_window_size=max_window_size,
    )

    logger = TensorBoardLogger(
        f"{get_wd()}lightning_logs",
        name=f"e{emb_dim}_h{n_heads}_d{dropout}_el{n_decoder_layers}_dl{n_decoder_layers}_f{dim_feedforward}_b{batch_size}_n{num_user_train}_w{max_window_size}",
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=200,
        progress_bar_refresh_rate=1,
        callbacks=[
            EarlyStopping(monitor="avg_val_auc", patience=20, mode="max"),
            ModelCheckpoint(
                monitor="avg_val_auc",
                filename="{epoch}-{val_loss_step:.2f}-{avg_val_auc:.2f}",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
    )

    # Train the model ⚡
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=[val_loader],
    )


if __name__ == "__main__":
    train()