from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional.classification import auroc
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        batch_size=256,  # will get saved as hyperparam
        num_user_train=300000,  # will get saved as hyperparam
        num_user_val=30000,  # will get saved as hyperparam
        max_window_size=100,
    ):
        super(RIIDDTransformerModel, self).__init__()
        self.model_type = "RiiidTransformer"
        self.learning_rate = learning_rate
        self.max_window_size = max_window_size

        # save params of models to yml
        self.save_hyperparameters()

        self.embed_content_id = nn.Embedding(n_content_id, emb_dim, padding_idx=13942)
        self.embed_parts = nn.Embedding(n_part, emb_dim, padding_idx=0)
        self.embed_tags = nn.Embedding(n_tags, emb_dim, padding_idx=188)
        # exercise weights to weight the mean embeded excercise embeddings
        self.exercise_weights = torch.nn.Parameter(torch.tensor([0.35, 0.55, 0.1]))

        self.embed_answered_correctly = nn.Embedding(
            n_correct, emb_dim, padding_idx=3
        )  # 2 + 1 for start token + 1 for padding_idn_inputs

        self.embed_timestamps = nn.Linear(1, emb_dim)

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
        for a length x
            if x >= 10:
                returns a random integer between 1 - 10
            else:
                returns a random integer between 1 - x
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
        batch["length"] = torch.where(
            batch["answers"][0, :] == 2,
            self.get_random_lengths(batch["length"]),
            batch["length"],
        )
        # randomize number of steps based on new random lengths
        batch["steps"] = self.get_random_steps(batch["length"], max_steps=max_steps)
        return batch

    @auto_move_data
    def forward(self, content_ids, parts, answers, tags, timestamps):
        # content_ids: (Source Sequence Length, Number of samples, Embedding)
        # tgt: (Target Sequence Length,Number of samples, Embedding)

        # if data is flat then expand to get Batch dim
        if len(content_ids.shape) == 1:
            content_ids = content_ids.unsqueeze(1)
            parts = parts.unsqueeze(1)
            answers = answers.unsqueeze(1)
            tags = tags.unsqueeze(1)
            timestamps = timestamps.unsqueeze(1)

        sequence_length = content_ids.shape[0]

        # sequence that will go into encoder
        embeded_content = self.embed_content_id(content_ids)
        embeded_parts = self.embed_parts(parts)
        embeded_tags = self.embed_tags(tags).sum(dim=2)
        e_w = F.softmax(self.exercise_weights, dim=0)

        embeded_exercise_sequence = (
            (embeded_content * e_w[0])
            + (embeded_parts * e_w[1])
            + (embeded_tags * e_w[2])
        )

        # sequence that will go into decoder
        embeded_responses = self.embed_answered_correctly(answers)
        embeded_timestamps = self.embed_timestamps(timestamps.unsqueeze(2))
        embeded_responses = (embeded_responses + embeded_timestamps) * 0.5

        # adding positional vector
        embedded_positions = self.pos_encoder(sequence_length)
        embeded_responses = embeded_responses + embedded_positions
        embeded_exercise_sequence = embeded_exercise_sequence + embedded_positions

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
        )

    @auto_move_data
    def predict_n_steps(self, batch, steps, return_all_preds=False):
        """
        Predicts n steps for all items in batch and return predictions
        only for those steps (flattened)
        steps: tensor of length B where each item is the number of steps that need to be taken
        """
        n_users = batch["content_ids"].shape[1]
        lengths = batch["length"]

        users = torch.arange(n_users)

        user_indexes = []
        sequence_indexes = []

        for i in range(steps.max().int(), 0, -1):
            preds = model.process_batch_step(batch)

            sequence_indexes_at_i = lengths[steps >= i] - i
            user_indexes_at_i = users[steps >= i]

            # set answer to either 0 or 1 if not lecture
            batch["answers"][sequence_indexes_at_i, user_indexes_at_i] = torch.where(
                batch["answers"][sequence_indexes_at_i, user_indexes_at_i] != 4,
                (preds[sequence_indexes_at_i, user_indexes_at_i] > 0.5).long(),
                4,
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
            answers[length - i] = torch.where(
                answers[length - i] != 4,
                (preds[length - i, 0] > 0.5).long(),
                4,
            )
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
        lengths = batch["length"]
        steps = batch["steps"]
        users = torch.arange(n_users)
        user_indexes = []
        sequence_indexes = []
        for i in range(steps.max().int(), 0, -1):
            preds = model.process_batch_step(batch)
            sequence_indexes_at_i = lengths[steps >= i] - i
            user_indexes_at_i = users[steps >= i]
            # set answer to either 0 or 1 if not lecture
            batch["answers"][sequence_indexes_at_i, user_indexes_at_i] = torch.where(
                batch["answers"][sequence_indexes_at_i, user_indexes_at_i] != 4,
                (preds[sequence_indexes_at_i, user_indexes_at_i] > 0.5).long(),
                4,
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
            torch.masked_select(result, batch["loss_mask"] > 0),
            torch.masked_select(batch["answered_correctly"], batch["loss_mask"] > 0),
            torch.masked_select(positions, select_mask),
        )

    def val_test_epoch_end(self, outputs, log_as="val"):
        y_pred = torch.cat([out[0] for out in outputs], dim=0)
        y = torch.cat([out[1] for out in outputs], dim=0)
        pos = torch.cat([out[2] for out in outputs], dim=0)
        auc = auroc(y_pred, y)

        # Calculate accuracy per position
        M = torch.zeros(pos.max() + 1, len(y), device=self.device)
        M[pos, torch.arange(len(y))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        acc_per_position = torch.mm(
            M, ((y_pred > 0.5) == y).float().unsqueeze(1)
        ).flatten()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.regplot(
            y=acc_per_position.cpu(), x=torch.arange(len(acc_per_position)).cpu(), ax=ax
        )
        ax.set_ylim(0.5, 1)
        ax.set_xlim(0, len(acc_per_position) - 1)
        ax.set_ylabel("acc")
        ax.set_xlabel("position")
        if log_as == "val":
            self.log(f"avg_{log_as}_auc", auc, prog_bar=True)
            self.logger.experiment.add_figure(
                f"{log_as}_acc_per_pos", fig, global_step=self.current_epoch
            )
        else:
            self.log(f"avg_{log_as}_auc", auc)
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
                optimizer, mode="max", patience=10
            ),
            "monitor": "avg_val_auc",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [scheduler]