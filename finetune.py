from model import RIIDDTransformerModel
from dataset import get_dataloaders
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from utils import get_wd
import hydra

SEED = 69
seed_everything(SEED)


@hydra.main(config_name="config_finetune")
def finetune(cfg) -> None:

    val_step_frequency = cfg["val_step_frequency"]
    learning_rate = cfg["learning_rate"]
    model_path = cfg["model_path"]
    batch_size = cfg["batch_size"]
    validation_batch_size = cfg["validation_batch_size"]
    max_window_size = cfg["max_window_size"]
    num_workers = cfg["num_workers"]
    use_lectures = cfg["use_lectures"]

    model = RIIDDTransformerModel.load_from_checkpoint(f"{get_wd()}{model_path}")
    model.learning_rate = learning_rate
    model.lr_step_frequency = val_step_frequency

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        validation_batch_size=validation_batch_size,
        max_window_size=max_window_size,
        use_lectures=use_lectures,
        num_workers=num_workers,
    )

    logger = TensorBoardLogger(
        f"{get_wd()}lightning_logs",
        name="fine_tune_lg",
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=2,
        progress_bar_refresh_rate=1,
        callbacks=[
            EarlyStopping(monitor="avg_val_auc", patience=5, mode="max"),
            ModelCheckpoint(
                monitor="avg_val_auc",
                filename="{epoch}-{val_loss_step:.2f}-{avg_val_auc:.2f}",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        val_check_interval=val_step_frequency,  # check validation every 1000 step
        limit_val_batches=0.05,  # run through only 5% of val every time
    )

    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=[val_loader],
    )


if __name__ == "__main__":
    finetune()
