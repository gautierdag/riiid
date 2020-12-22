import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
import hydra

from dataset import get_dataloaders
from model import RIIDDTransformerModel

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
    validation_batch_size = cfg["validation_batch_size"]
    max_window_size = cfg["max_window_size"]
    num_workers = cfg["num_workers"]
    use_lectures = cfg["use_lectures"]
    use_prior_q_times = cfg["use_prior_q_times"]
    val_step_frequency = cfg["val_step_frequency"]
    val_size = cfg["val_size"]
    accumulate_grad_batches = cfg["accumulate_grad_batches"]
    use_agg_feats = cfg["use_agg_feats"]
    use_exercise_feats = cfg["use_exercise_feats"]

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        validation_batch_size=validation_batch_size,
        max_window_size=max_window_size,
        use_lectures=use_lectures,
        num_workers=num_workers,
        use_agg_feats=use_agg_feats,
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
        max_window_size=max_window_size,
        use_prior_q_times=use_prior_q_times,
        lr_step_frequency=val_step_frequency,
        use_agg_feats=use_agg_feats,
        use_exercise_feats=use_exercise_feats,
    )

    experiment_name = (
        f"base_e{emb_dim}_h{n_heads}_d{dropout}_lr{learning_rate}"
        + f"_el{n_decoder_layers}_dl{n_decoder_layers}"
        + f"_f{dim_feedforward}_b{batch_size}_w{max_window_size}"
        + f"_lec_{use_lectures}_qtimes_{use_prior_q_times}_use_agg_{use_agg_feats}_use_ex{use_exercise_feats}"
    )
    logger = TensorBoardLogger(f"{get_wd()}lightning_logs", name=experiment_name)

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
        progress_bar_refresh_rate=accumulate_grad_batches,
        callbacks=[
            EarlyStopping(monitor="avg_val_auc", patience=10, mode="max"),
            ModelCheckpoint(
                monitor="avg_val_auc",
                filename="{epoch}-{val_loss_step:.2f}-{avg_val_auc:.2f}",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        val_check_interval=val_step_frequency,  # check validation every validation_step
        limit_val_batches=val_size,  # run through only 10% of val every time
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Train the model ⚡
    trainer.fit(
        model, train_dataloader=train_loader, val_dataloaders=[val_loader],
    )

    # Test on Final Full validation set
    trainer.test(test_dataloaders=[val_loader])


if __name__ == "__main__":
    train()
