import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("fivethirtyeight")
import seaborn as sns

import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional.classification import auroc


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, sequence_length):
        # returns embeds (sequence_length, 1, d_model)
        return self.pe[:sequence_length, :]


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, sequence_length):
        # returns embeds (sequence_length, 1, d_model)
        return self.pe[:sequence_length, :]


class RIIDDTransformerModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        n_content_id=13943,  # number of different contents = 13942 + 1 (for padding)
        n_part=8,  # number of different parts = 7 + 1 (for padding)
        n_tags=189,  # number of different tags = 188 + 1 (for padding)
        n_correct=5,  # 0,1 (false, true), 2 (start token), 3 (padding), 4 (lecture)
        emb_dim=64,  # embedding dimension
        dropout=0.1,
        n_heads: int = 1,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        activation: str = "relu",
        max_window_size=100,
        use_prior_q_times=False,
        use_prior_q_explanation=False,
    ):
        super(RIIDDTransformerModel, self).__init__()
        self.model_type = "RiiidTransformer"
        self.learning_rate = learning_rate
        self.max_window_size = max_window_size

        self.use_prior_q_times = use_prior_q_times
        self.use_prior_q_explanation = use_prior_q_explanation

        # save params of models to yml
        self.save_hyperparameters()

        #### EXERCISE SEQUENCE
        self.embed_content_id = nn.Embedding(n_content_id, emb_dim, padding_idx=13942)
        self.embed_parts = nn.Embedding(n_part, emb_dim, padding_idx=0)
        self.embed_tags = nn.Embedding(n_tags, emb_dim, padding_idx=188)

        # exercise weights to weight the mean embeded excercise embeddings
        self.exercise_weights = torch.nn.Parameter(torch.tensor([0.35, 0.55, 0.1]))

        ### RESPONSE SEQUENCE (1st time stamp of sequence is useless)
        self.embed_answered_correctly = nn.Embedding(
            n_correct, emb_dim, padding_idx=3
        )  # 2 + 1 for start token + 1 for padding_idn_inputs

        self.embed_timestamps = nn.Linear(1, emb_dim)

        # embed prior q time and q explanation
        self.embed_prior_q_time = nn.Linear(1, emb_dim)
        self.embed_prior_q_explanation = nn.Embedding(2, emb_dim)

        # response weights to weight the mean embeded response embeddings
        w = [0.5, 0.5]
        if use_prior_q_times:
            w.append(0.5)
        if use_prior_q_explanation:
            w.append(0.5)

        self.response_weights = torch.nn.Parameter(torch.tensor([w]))

        self.pos_encoder = PositionalEncoding(emb_dim)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )

        self.out_linear = nn.Linear(emb_dim, 2)
        init_weights(self)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def get_random_steps(self, lengths, max_steps=10):
        """
        for x return integer between 1 - 10 or
        between 1 - x if x < 10
        """
        m = torch.distributions.uniform.Uniform(
            0,
            (
                torch.minimum(
                    torch.ones(lengths.shape, device=self.device) * 10, lengths
                )
            ).float(),
        )
        return torch.floor(m.sample()).long() + 1

    def get_random_lengths(self, lengths):
        # gets random new lengths
        m = torch.distributions.uniform.Uniform(0, lengths.float())
        return torch.floor(m.sample()).long() + 1

    def randomize_evaluation_step(self, batch, max_steps=10):
        # randomize new lengths (for where start token is present)
        # batch["length"] = torch.where(
        #     batch["answers"][0, :] == 2,
        #     self.get_random_lengths(batch["length"]),
        #     batch["length"],
        # )
        # randomize number of steps based on new random lengths
        batch["steps"] = self.get_random_steps(batch["length"], max_steps=max_steps)
        return batch

    @auto_move_data
    def forward(
        self,
        content_ids,
        parts,
        answers,
        tags,
        timestamps,
        prior_q_times,
        prior_q_explanation,
    ):
        # content_ids: (Source Sequence Length, Number of samples, Embedding)
        # tgt: (Target Sequence Length,Number of samples, Embedding)

        # if data is flat then expand to get Batch dim
        if len(content_ids.shape) == 1:
            content_ids = content_ids.unsqueeze(1)
            parts = parts.unsqueeze(1)
            answers = answers.unsqueeze(1)
            tags = tags.unsqueeze(1)
            timestamps = timestamps.unsqueeze(1)
            prior_q_times = prior_q_times.unsqueeze(1)
            prior_q_explanation = prior_q_explanation.unsqueeze(1)

        sequence_length = content_ids.shape[0]

        # sequence that will go into encoder
        embeded_content = self.embed_content_id(content_ids)
        embeded_parts = self.embed_parts(parts)
        embeded_tags = self.embed_tags(tags).sum(dim=2)
        e_w = F.softmax(self.exercise_weights, dim=0)

        embeded_exercise_sequence = (
            torch.stack([embeded_content, embeded_parts, embeded_tags], dim=3) * e_w
        ).sum(dim=3)

        # sequence that will go into decoder
        embeded_answered_correctly = self.embed_answered_correctly(answers)
        embeded_timestamps = self.embed_timestamps(timestamps.unsqueeze(2))
        r_w = F.softmax(self.response_weights, dim=0)

        exercise_sequence_components = [embeded_answered_correctly, embeded_timestamps]
        if self.use_prior_q_times:
            embeded_q_times = self.embed_prior_q_time(prior_q_times.unsqueeze(2))
            # zero embedding - if start token
            embeded_q_times[0, torch.where(answers[0, :] == 2)[0], :] = 0
            exercise_sequence_components.append(embeded_q_times)
        if self.use_prior_q_explanation:
            embeded_q_explanation = self.embed_prior_q_explanation(prior_q_explanation)
            # zero embedding - if start token
            embeded_q_explanation[0, torch.where(answers[0, :] == 2)[0], :] = 0
            exercise_sequence_components.append(embeded_q_explanation)

        embeded_responses = (
            torch.stack(exercise_sequence_components, dim=3) * r_w
        ).sum(dim=3)

        # adding positional vector
        embedded_positions = self.pos_encoder(sequence_length + 1)

        # add shifted position embedding ( start token is first position)
        embeded_responses = embeded_responses + embedded_positions[:-1, :, :]

        embeded_exercise_sequence = (
            embeded_exercise_sequence + embedded_positions[1:, :, :]
        )

        # mask of shape S x S -> prevents attention looking forward
        top_right_attention_mask = self.generate_square_subsequent_mask(
            sequence_length
        ).type_as(embeded_exercise_sequence)

        output = self.transformer(
            embeded_exercise_sequence,
            embeded_responses,
            tgt_mask=top_right_attention_mask,  # (S,S)
            src_mask=top_right_attention_mask,  # (T,T)
        )

        output = self.out_linear(output)
        return F.softmax(output, dim=2)[:, :, 1]

    def process_batch_step(self, batch):
        # return result
        return self(
            batch["content_ids"],
            batch["parts"],
            batch["answers"],
            batch["tags"],
            batch["timestamps"],
            batch["prior_q_times"],
            batch["prior_q_explanation"],
        )

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
            preds = self.process_batch_step(batch)

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

    @auto_move_data
    def predict_fast_single_user(
        self, content_ids, parts, answers, tags, timestamps, n=1
    ):
        """
        Predicts n steps for a single user in batch and return predictions
        only for those steps (flattened)
        """
        length = len(content_ids)
        out_predictions = torch.zeros(n, device=self.device)
        for i in range(n, 0, -1):
            preds = self(content_ids, parts, answers, tags, timestamps)
            out_predictions[n - i] = preds[length - i, 0]

            # answers are shifted (start token) so need + 1
            answer_idx = length - i + 1
            # don't update if at end of answers
            if answer_idx + 1 < len(answers):
                # don't update if true is lecture
                if answers[answer_idx] != 4:
                    answers[answer_idx] = (preds[length - i, 0] > 0.5).long()

        return out_predictions

    def training_step(self, batch, batch_nb):
        result = self.process_batch_step(batch)
        loss = F.binary_cross_entropy(
            result, batch["answered_correctly"], weight=batch["loss_mask"]
        )
        self.log("train_loss", loss)
        return loss

    def validate_n_steps(self, batch, max_steps=10):
        """
        Predicts max_steps steps for all items in batch and return predictions
        only for those steps (flattened)
        steps: tensor of length B where each item is the number of steps that need to be taken
        """
        n_users = batch["content_ids"].shape[1]
        seq_length = batch["answers"].shape[0]
        lengths = batch["length"]
        steps = batch["steps"]
        users = torch.arange(n_users)
        user_indexes = []
        sequence_indexes = []
        for i in range(steps.max().int(), 0, -1):
            preds = self.process_batch_step(batch)
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

        user_indexes = torch.cat(user_indexes)
        sequence_indexes = torch.cat(sequence_indexes)
        return (preds, sequence_indexes, user_indexes)

    def val_test_step(self, batch, log_as="val"):
        batch = self.randomize_evaluation_step(batch)

        result, sequence_indexes, user_indexes = self.validate_n_steps(batch)

        step_mask = torch.zeros(batch["loss_mask"].shape, device=self.device)
        step_mask[sequence_indexes, user_indexes] = 1

        batch["loss_mask"] *= step_mask

        loss = F.binary_cross_entropy(
            result, batch["answered_correctly"], weight=batch["loss_mask"]
        )
        self.log(f"{log_as}_loss_step", loss)
        select_mask = batch["loss_mask"] > 0
        positions = torch.cat(
            result.shape[1] * [torch.arange(result.shape[0]).unsqueeze(1)], dim=1
        )
        return (
            torch.masked_select(result, select_mask),
            torch.masked_select(batch["answered_correctly"], select_mask),
            torch.masked_select(positions, select_mask),
        )

    def val_test_epoch_end(self, outputs, log_as="val", plot_acc=False):
        y_pred = torch.cat([out[0] for out in outputs], dim=0)
        y = torch.cat([out[1] for out in outputs], dim=0)
        pos = torch.cat([out[2] for out in outputs], dim=0)
        auc = auroc(y_pred, y)

        if plot_acc:
            # Calculate accuracy per position
            M = torch.zeros(pos.max() + 1, len(y), device=self.device)
            M[pos, torch.arange(len(y))] = 1
            M = torch.nn.functional.normalize(M, p=1, dim=1)
            acc_per_position = torch.mm(
                M, ((y_pred > 0.5) == y).float().unsqueeze(1)
            ).flatten()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.regplot(
                y=acc_per_position.cpu().numpy(),
                x=torch.arange(len(acc_per_position)).cpu().numpy(),
                ax=ax,
            )
            ax.set_ylim(0.5, 1)
            ax.set_xlim(0, len(acc_per_position) - 1)
            ax.set_ylabel("acc")
            ax.set_xlabel("position")

        if log_as == "val":
            self.log(f"avg_{log_as}_auc", auc, prog_bar=True)
            if plot_acc:
                self.logger.experiment.add_figure(
                    f"{log_as}_acc_per_pos", fig, global_step=self.current_epoch
                )
        else:
            self.log(f"avg_{log_as}_auc", auc)
            if plot_acc:
                self.logger.experiment.add_figure(
                    f"{log_as}_acc_per_pos", fig, global_step=1
                )

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
                optimizer, mode="max", patience=5
            ),
            "monitor": "avg_val_auc",
            "interval": "step",
            "frequency": 1000,
            "strict": True,
        }

        return [optimizer], [scheduler]
