import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import h5py

from preprocessing import (
    get_time_elapsed_from_timestamp,
    questions_lectures_tags,
    questions_lectures_parts,
)
from utils import get_wd


class RIIDDataset(Dataset):
    """RIID dataset."""

    def __init__(
        self,
        user_mapping,
        user_history,
        hdf5_file="feats.h5",
        window_size=100,
    ):
        """
        Args:
            user_mapping (np.array): array of user_ids per row
            user_history (np.array): array of length of history up till this row
            hdf5_file (string): location of hf5 feats file
        """
        # np array where index maps to a user id
        self.user_mapping = user_mapping
        self.user_history = user_history
        self.hdf5_file = f"{get_wd()}{hdf5_file}"
        self.max_window_size = window_size
        

    def open_hdf5(self):
        # opens the h5py file
        self.f = h5py.File(self.hdf5_file, "r")

    def __len__(self):
        return len(self.user_mapping)

    def __getitem__(self, idx):

        # open the hdf5 file in the iterator to allow multiple workers
        # https://github.com/pytorch/pytorch/issues/11929
        if not hasattr(self, "f"):
            self.open_hdf5()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        user_id = self.user_mapping[idx]
        length = self.user_history[idx]
        # length = self.f[f"{user_id}/answered_correctly"].len()

        window_size = min(self.max_window_size, length)

        content_ids = np.zeros(window_size, dtype=np.int64).copy()
        answered_correctly = np.zeros(window_size, dtype=np.int64).copy()
        timestamps = np.zeros(window_size, dtype=np.float32).copy()
        prior_q_times = np.zeros(window_size, dtype=np.float32).copy()
        prior_q_explanation = np.zeros(window_size, dtype=np.float32).copy()

        # index for loading larger than window size
        start_index = 0
        if length > window_size:
            # randomly select window size subset instead of trying to cram in everything
            start_index = length - window_size

        self.f[f"{user_id}/content_ids"].read_direct(
            content_ids,
            source_sel=np.s_[start_index : start_index + window_size],
            dest_sel=np.s_[0:window_size],
        )
        self.f[f"{user_id}/answered_correctly"].read_direct(
            answered_correctly,
            source_sel=np.s_[start_index : start_index + window_size],
            dest_sel=np.s_[0:window_size],
        )
        self.f[f"{user_id}/timestamps"].read_direct(
            timestamps,
            source_sel=np.s_[start_index : start_index + window_size],
            dest_sel=np.s_[0:window_size],
        )
        self.f[f"{user_id}/prior_question_elapsed_time"].read_direct(
            prior_q_times,
            source_sel=np.s_[start_index : start_index + window_size],
            dest_sel=np.s_[0:window_size],
        )
        self.f[f"{user_id}/prior_question_had_explanation"].read_direct(
            prior_q_explanation,
            source_sel=np.s_[start_index : start_index + window_size],
            dest_sel=np.s_[0:window_size],
        )

        # convert timestamps to time elapsed
        time_elapsed_timestamps = get_time_elapsed_from_timestamp(timestamps)

        # get question tags
        tags = questions_lectures_tags[content_ids, :]

        # get question parts
        parts = questions_lectures_parts[content_ids]

        # shift by one the answered_correctly sequence
        answers = np.roll(answered_correctly, 1)

        # set start token if start_index is actually first element
        if start_index == 0:
            answers[0] = 2
        # else replace first element of sequence with actual previous element
        else:
            self.f[f"{user_id}/answered_correctly"].read_direct(
                answers,
                source_sel=np.s_[start_index - 1],
                dest_sel=np.s_[0],
            )

        return {
            "parts": torch.from_numpy(parts).long(),
            "tags": torch.from_numpy(tags).long(),
            "content_ids": torch.from_numpy(content_ids),
            "answered_correctly": torch.from_numpy(answered_correctly),
            "answers": torch.from_numpy(answers),
            "timestamps": torch.from_numpy(time_elapsed_timestamps),
            "prior_q_times": torch.from_numpy(prior_q_times),
            "prior_q_explanation": torch.from_numpy(prior_q_explanation),
            "length": window_size,
        }


def collate_fn(batch):
    """
    The collate function is used to merge individual data samples into a batch
    It handles the padding aspect
    """

    # collate lenghts into 1D tensor
    items = {"length": torch.tensor([batch_item["length"] for batch_item in batch])}

    # find shape that the batch will have
    max_length = items["length"].max()
    num_items = len(batch)

    # padding list
    for (key, padding) in [
        ("parts", 0),
        ("content_ids", 13942),
        ("answered_correctly", 3),
        ("answers", 3),
        ("timestamps", 0.0),  # note timestamps isnt an embedding
        ("tags", 188),
        ("prior_q_times", 0),
        ("prior_q_explanation", 0),
    ]:
        items[key] = pad_sequence(
            [batch_item[key] for batch_item in batch],
            batch_first=False,
            padding_value=padding,
        )

    # mask to weight loss by (S, N)
    items["loss_mask"] = (
        (
            torch.arange(max_length).expand(num_items, max_length)
            < items["length"].unsqueeze(1)
        )
        .transpose(1, 0)
        .float()
    )
    items["loss_mask"] *= items["answered_correctly"] != 4  # mask the lectures
    items["answered_correctly"] = items["answered_correctly"].float()

    return items
