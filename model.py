import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional.classification import auroc
from transformer import TransformerEncDec


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class RIIDDTransformerModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        n_content_id=13943,  # number of different contents = 13942 + 1 (for padding)
        n_part=8,  # number of different parts = 7 + 1 (for padding)
        n_tags=189,  # number of different tags = 188 + 1 (for padding)
        n_correct=5,  # 0,1 (false, true), 2 (start token), 3 (padding), 4 (lecture)
        n_agg_feats=12,  # number of agg feats
        n_exercise_feats=4,  # number of exercise feats
        emb_dim=64,  # embedding dimension
        dropout=0.1,
        n_heads: int = 1,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        activation: str = "relu",
        max_window_size=100,
        start_window_size=10,
        use_prior_q_times=False,
        use_agg_feats=False,
        use_exercise_feats=False,
        lr_step_frequency=2000,
        transformer_type="base",
    ):
        super(RIIDDTransformerModel, self).__init__()
        self.model_type = "RiiidTransformer"
        self.learning_rate = learning_rate
        self.lr_step_frequency = lr_step_frequency
        self.max_window_size = max_window_size
        self.window_size = start_window_size
        self.n_heads = n_heads
        self.transformer_type = transformer_type

        self.use_prior_q_times = use_prior_q_times
        self.use_agg_feats = use_agg_feats
        self.use_exercise_feats = use_exercise_feats

        # save params of models to yml
        self.save_hyperparameters()

        #### EXERCISE SEQUENCE
        self.embed_content_id = nn.Embedding(n_content_id, emb_dim, padding_idx=13942)
        self.embed_parts = nn.Embedding(n_part, emb_dim, padding_idx=0)
        self.embed_tags = nn.Embedding(n_tags, emb_dim, padding_idx=188)

        # exercise weights to weight the mean embeded excercise embeddings
        e_w = [0.35, 0.55, 0.1]
        if self.use_exercise_feats:
            self.embed_exercise_features = nn.Linear(n_exercise_feats, emb_dim)
            e_w.append(0.1)

        self.exercise_weights = nn.Parameter(torch.tensor(e_w), requires_grad=True)
        self.register_parameter("exercise_weights", self.exercise_weights)

        ### RESPONSE SEQUENCE (1st time stamp of sequence is useless)
        self.embed_answered_correctly = nn.Embedding(
            n_correct, emb_dim, padding_idx=3
        )  # 2 + 1 for start token + 1 for padding_idn_inputs
        self.embed_timestamps = nn.Linear(1, emb_dim)
        # response weights to weight the mean embeded response embeddings
        r_w = [0.5, 0.5]

        if use_prior_q_times:
            # embed prior q time
            self.embed_prior_q_time = nn.Linear(1, emb_dim)
            r_w.append(0.5)
        if use_agg_feats:
            self.embed_agg_feats = nn.Linear(n_agg_feats, emb_dim)
            r_w.append(0.5)

        self.response_weights = nn.Parameter(torch.tensor(r_w), requires_grad=True)
        self.register_parameter("response_weights", self.response_weights)  ###

        self.transformer = TransformerEncDec(
            d_model=emb_dim,
            transformer_type=transformer_type,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )
        self.out_linear = nn.Linear(emb_dim, 2)
        init_weights(self)

    def get_random_steps(self, lengths, max_steps=10):
        """
        for x return integer between 1 - 10 or
        between 1 - x if x < 10
        """
        m = torch.distributions.uniform.Uniform(
            0,
            (
                torch.minimum(
                    torch.ones(lengths.shape, device=self.device) * max_steps, lengths
                )
            ).float(),
        )
        return torch.floor(m.sample()).long() + 1

    def select_last_window_subset(
        self, lengths, window_size=100,
    ):
        """
        Returns the idxs for the last window_size subset of lengths
        Inputs:
            lengths (Long Tensor, [batch_dim]): the length at which each sequence ends for each element in batch
            window_size (int): the length from which to select
        Returns:
            idxs (Tensor, [window_size, batch_dim, 1])
        """
        sequence_length = lengths.max()
        batch_dim = len(lengths)

        if window_size > sequence_length:
            window_size = sequence_length

        # gather_idxs should be of size (window_size, batch_dim, 1)
        gather_idxs = (
            torch.arange(window_size, device=self.device)
            .expand(1, window_size)
            .transpose(1, 0)
            .unsqueeze(1)
            .repeat(1, batch_dim, 1)
        )

        # select lengths > window_size - where the indexes will be funky
        # namely where the new indexes won't start at 0
        outside_start_window = torch.where(lengths > window_size)[0]
        if len(outside_start_window) > 0:
            possible_idxs = torch.arange(sequence_length, device=self.device).expand(
                len(outside_start_window), sequence_length
            )
            # select the correct indexes for sequence_elements of length > window_size
            outside_start_window_idxs = torch.nonzero(
                (
                    (possible_idxs < lengths[outside_start_window].unsqueeze(1))
                    & (
                        possible_idxs
                        >= lengths[outside_start_window].unsqueeze(1) - window_size
                    )
                )
            )[:, 1].reshape(len(outside_start_window), window_size)
            outside_start_window_idxs = outside_start_window_idxs.transpose(
                1, 0
            ).unsqueeze(2)
            gather_idxs[:, outside_start_window, :] = outside_start_window_idxs

        return gather_idxs

    def forward(
        self,
        content_ids,
        parts,
        answers,
        tags,
        timestamps,
        prior_q_times,
        last_window_subset_idxs,
        agg_feats=None,
        e_feats=None,
        **kwargs,
    ):
        # content_ids: (Source Sequence Length, Number of samples, Embedding)
        # tgt: (Target Sequence Length,Number of samples, Embedding)

        # sequence that will go into encoder
        embeded_content = self.embed_content_id(content_ids)
        embeded_parts = self.embed_parts(parts)
        embeded_tags = self.embed_tags(tags).sum(dim=2)

        exercise_sequence_components = [
            embeded_content,
            embeded_parts,
            embeded_tags,
        ]
        if self.use_exercise_feats:
            embeded_exercise_feats = self.embed_exercise_features(e_feats)
            exercise_sequence_components.append(embeded_exercise_feats)

        e_w = F.softmax(self.exercise_weights, dim=0)
        embeded_exercises = (
            torch.stack(exercise_sequence_components, dim=3) * e_w
        ).sum(dim=3)

        # sequence that will go into decoder
        embeded_answered_correctly = self.embed_answered_correctly(answers)
        embeded_timestamps = self.embed_timestamps(timestamps.unsqueeze(2))

        response_sequence_components = [embeded_answered_correctly, embeded_timestamps]
        if self.use_prior_q_times:
            embeded_q_times = self.embed_prior_q_time(prior_q_times.unsqueeze(2))
            # zero embedding - if start token
            embeded_q_times[0, torch.where(answers[0, :] == 2)[0], :] = 0
            response_sequence_components.append(embeded_q_times)

        if self.use_agg_feats:
            embeded_agg_feats = self.embed_agg_feats(agg_feats)
            embeded_agg_feats[0, torch.where(answers[0, :] == 2)[0], :] = 0
            response_sequence_components.append(embeded_agg_feats)

        r_w = F.softmax(self.response_weights, dim=0)
        embeded_responses = (
            torch.stack(response_sequence_components, dim=3) * r_w
        ).sum(dim=3)

        shorter_embeded_exercises = embeded_exercises.gather(
            0, last_window_subset_idxs.repeat(1, 1, embeded_exercises.shape[2])
        )
        shorter_embeded_responses = embeded_responses.gather(
            0, last_window_subset_idxs.repeat(1, 1, embeded_responses.shape[2])
        )

        output = self.transformer(shorter_embeded_exercises, shorter_embeded_responses)
        output = self.out_linear(output)
        return F.softmax(output, dim=2)[:, :, 1]

    @auto_move_data
    def predict_n_steps(self, batch, steps, return_all_preds=False):
        """
        Predicts n steps for all items in batch and return predictions
        only for those steps (flattened)
        steps: tensor of length B where each item is the number of steps that need to be taken
        """
        seq_length, n_users = batch["content_ids"].shape
        lengths = batch["length"]

        users = torch.arange(n_users)

        user_indexes = []
        sequence_indexes = []

        for i in range(steps.max().int(), 0, -1):
            preds = self(**batch)

            sequence_indexes_at_i = lengths[steps >= i] - i
            user_indexes_at_i = users[steps >= i]

            # get index for which to update the answers
            # since answers is shifted we want to map preds 0..98 -> answers 1:99
            answers_idx = torch.where(sequence_indexes_at_i + 1 != seq_length)
            a_seq_idx = sequence_indexes_at_i[answers_idx] + 1
            u_seq_idx = user_indexes_at_i[answers_idx]

            # set answer to either 0 or 1 if not lecture
            batch["answers"][a_seq_idx, u_seq_idx] = torch.where(
                batch["answers"][a_seq_idx, u_seq_idx] != 4,
                (preds[sequence_indexes_at_i[answers_idx], u_seq_idx] > 0.5).long(),
                batch["answers"][a_seq_idx, u_seq_idx],
            )

            user_indexes.append(user_indexes_at_i)
            sequence_indexes.append(sequence_indexes_at_i)

        if return_all_preds:
            return preds

        user_indexes = torch.cat(user_indexes)
        sequence_indexes = torch.cat(sequence_indexes)

        return (
            preds[sequence_indexes, user_indexes],
            batch["row_ids"][sequence_indexes, user_indexes],
        )

    def training_step(self, batch, batch_nb):
        # select the window size appropriate for transformer from larger sequence

        last_window_subset_idxs = self.select_last_window_subset(
            batch["length"], window_size=self.window_size
        )

        result = self(**batch, last_window_subset_idxs=last_window_subset_idxs)

        answers = batch["answered_correctly"].gather(
            0, last_window_subset_idxs.squeeze(2)
        )
        loss_mask = batch["loss_mask"].gather(0, last_window_subset_idxs.squeeze(2))

        loss = F.binary_cross_entropy(result, answers, weight=loss_mask)
        self.log("train_loss", loss.cpu())

        # update window size during training
        if (
            self.global_step % 500 == 0
            and self.global_step != 0
            and self.window_size < self.max_window_size
        ):
            self.window_size += 1

        self.log("window_size", self.window_size)

        return loss

    def validate_n_steps(self, batch, last_window_subset_idxs):
        """
        Predicts max_steps steps for all items in batch and return predictions
        only for those steps (flattened)
        steps: tensor of length B where each item is the number of steps that need to be taken
        """
        n_users = batch["content_ids"].shape[1]
        seq_length = batch["answers"].shape[0]
        lengths = batch["length"]
        steps = batch["steps"]
        prediction_window_size = last_window_subset_idxs.shape[0]
        users = torch.arange(n_users)
        user_indexes = []
        sequence_indexes = []
        for i in range(steps.max().int(), 0, -1):
            preds = self(**batch, last_window_subset_idxs=last_window_subset_idxs)
            sequence_indexes_at_i = lengths[steps >= i] - i
            preds_shifted_sequence_indexes_at_i = sequence_indexes_at_i - (
                lengths[steps >= i] - prediction_window_size
            ).clamp(min=0)
            user_indexes_at_i = users[steps >= i]

            # get index for which to update the answers
            # since answers is shifted we want to map preds 0..98 -> answers 1:99
            answers_idx = torch.where(sequence_indexes_at_i + 1 != seq_length)[0]
            a_seq_idx = sequence_indexes_at_i[answers_idx] + 1
            u_seq_idx = user_indexes_at_i[answers_idx]

            # set answer to either 0 or 1 if not lecture
            batch["answers"][a_seq_idx, u_seq_idx] = torch.where(
                batch["answers"][a_seq_idx, u_seq_idx] != 4,
                (
                    preds[preds_shifted_sequence_indexes_at_i[answers_idx], u_seq_idx]
                    > 0.5
                ).long(),
                batch["answers"][a_seq_idx, u_seq_idx],
            )

            user_indexes.append(user_indexes_at_i)
            sequence_indexes.append(sequence_indexes_at_i)

        user_indexes = torch.cat(user_indexes)
        sequence_indexes = torch.cat(sequence_indexes)
        return (preds, sequence_indexes, user_indexes)

    def val_test_step(self, batch, log_as="val"):
        batch["steps"] = self.get_random_steps(batch["length"], max_steps=10)
        # evaluate on max window size
        last_window_subset_idxs = self.select_last_window_subset(
            batch["length"], window_size=self.max_window_size
        )
        answers = batch["answered_correctly"].gather(
            0, last_window_subset_idxs.squeeze(2)
        )
        loss_mask = batch["loss_mask"].gather(0, last_window_subset_idxs.squeeze(2))

        result, sequence_indexes, user_indexes = self.validate_n_steps(
            batch, last_window_subset_idxs
        )
        step_mask = torch.zeros(batch["loss_mask"].shape, device=self.device)
        step_mask[sequence_indexes, user_indexes] = 1
        step_mask = step_mask.gather(0, last_window_subset_idxs.squeeze(2))

        loss_mask *= step_mask

        loss = F.binary_cross_entropy(result, answers, weight=loss_mask)
        self.log(f"{log_as}_loss_step", loss.cpu())

        select_mask = loss_mask > 0
        return (
            torch.masked_select(result, select_mask).cpu(),
            torch.masked_select(answers, select_mask).cpu(),
        )

    def val_test_epoch_end(self, outputs, log_as="val"):
        y_pred = torch.cat([out[0] for out in outputs], dim=0)
        y = torch.cat([out[1] for out in outputs], dim=0)
        auc = auroc(y_pred, y)
        if log_as == "val":
            self.log(f"avg_{log_as}_auc", auc, prog_bar=True)
        else:
            self.log(f"avg_{log_as}_auc", auc)

    def validation_step(self, batch, batch_nb, dataset_nb=None):
        return self.val_test_step(batch, log_as="val")

    def validation_epoch_end(self, outputs):
        self.val_test_epoch_end(outputs, log_as="val")

    def test_step(self, batch, batch_nb, dataset_nb=None):
        return self.val_test_step(batch, log_as="test")

    def test_epoch_end(self, outputs):
        self.val_test_epoch_end(outputs, log_as="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=2, factor=0.5
            ),
            "monitor": "avg_val_auc",
            "interval": "epoch",
            "strict": True,
        }

        return [optimizer], [scheduler]
