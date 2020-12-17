import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import h5py

from preprocessing import (
    get_time_elapsed_from_timestamp,
    questions_lectures_tags,
    questions_lectures_parts,
    generate_h5,
)
from utils import get_wd, DATA_FOLDER_PATH


import math
import torch.nn.functional as F


def pad_to_multiple(tensor, multiple=128, pad_value=0):
    m = tensor.shape[0] / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - tensor.shape[0]
    pad_offset = (0,) * 2 * (len(tensor.shape) - 1)
    return F.pad(tensor, (*pad_offset, 0, remainder), value=pad_value)


class RIIDDataset(Dataset):
    """RIID dataset."""

    def __init__(
        self,
        user_mapping,
        user_history,
        hdf5_file="feats.h5",
        window_size=100,
        use_cache=False,
    ):
        """
        Args:
            user_mapping (np.array): array of user_ids per row
            user_history (np.array): array of length of history up till this row
            hdf5_file (string): location of hf5 feats file
            window_size (int): size of window to lookback to max
            use_cache (opt, bool): whether to cache reads
        """
        # np array where index maps to a user id
        self.user_mapping = user_mapping
        self.user_history = user_history
        self.hdf5_file = f"{hdf5_file}"
        self.max_window_size = window_size
        self.use_cache = use_cache
        self.cache = {}

    def open_hdf5(self):
        # opens the h5py file
        self.f = h5py.File(self.hdf5_file, "r")

    def __len__(self):
        return len(self.user_mapping)

    def load_user_into_cache(self, user_id):
        """
        add a user to self.cache
        """

        self.cache[user_id] = {
            "content_ids": np.array(self.f[f"{user_id}/content_ids"], dtype=np.int64),
            "answered_correctly": np.array(
                self.f[f"{user_id}/answered_correctly"], dtype=np.int64
            ),
            "timestamps": np.array(self.f[f"{user_id}/timestamps"], dtype=np.float32),
            "prior_question_elapsed_time": np.array(
                self.f[f"{user_id}/prior_question_elapsed_time"], dtype=np.float32
            ),
        }

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

        # index for loading larger than window size
        start_index = 0
        if length > window_size:
            # randomly select window size subset instead of trying to cram in everything
            start_index = length - window_size

        if not self.use_cache:
            content_ids = np.zeros(window_size, dtype=np.int64).copy()
            answered_correctly = np.zeros(window_size, dtype=np.int64).copy()
            timestamps = np.zeros(window_size, dtype=np.float32).copy()
            prior_q_times = np.zeros(window_size, dtype=np.float32).copy()

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
        else:
            if user_id not in self.cache:
                self.load_user_into_cache(user_id)

            content_ids = self.cache[user_id]["content_ids"][
                start_index : start_index + window_size
            ].copy()
            answered_correctly = self.cache[user_id]["answered_correctly"][
                start_index : start_index + window_size
            ].copy()
            timestamps = self.cache[user_id]["timestamps"][
                start_index : start_index + window_size
            ].copy()
            prior_q_times = self.cache[user_id]["prior_question_elapsed_time"][
                start_index : start_index + window_size
            ].copy()

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
            "content_ids": torch.from_numpy(content_ids).long(),
            "answered_correctly": torch.from_numpy(answered_correctly).float(),
            "answers": torch.from_numpy(answers).long(),
            "timestamps": torch.from_numpy(time_elapsed_timestamps).float(),
            "prior_q_times": torch.from_numpy(prior_q_times).float(),
            "length": window_size,
        }


def get_collate_fn(min_multiple=None):
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
        ]:
            items[key] = pad_sequence(
                [batch_item[key] for batch_item in batch],
                batch_first=False,
                padding_value=padding,
            )

            if min_multiple is not None:
                # apply special padding
                items[key] = pad_to_multiple(
                    items[key], multiple=min_multiple, pad_value=padding
                )

        new_max_length = items["answered_correctly"].shape[0]
        # mask to weight loss by (S, N)
        items["loss_mask"] = (
            (
                torch.arange(new_max_length).expand(num_items, new_max_length)
                < items["length"].unsqueeze(1)
            )
            .transpose(1, 0)
            .float()
        )
        items["loss_mask"] *= items["answered_correctly"] != 4  # mask the lectures
        items["answered_correctly"] = items["answered_correctly"].float()

        return items

    return collate_fn


def get_train_val_idxs(
    df,
    train_size=90000000,
    validation_size=2500000,
    new_user_prob=0.25,
    use_lectures=True,
):
    try:
        train_idxs = np.load(f"{get_wd()}{DATA_FOLDER_PATH}/train_{train_size}.npy")
        val_idxs = np.load(f"{get_wd()}{DATA_FOLDER_PATH}/val_{validation_size}.npy")

    except FileNotFoundError:
        train_idxs = []
        val_idxs = []

        # create df with user_ids and indices
        tmp_df = df[~df.content_type_id][["user_id"]].copy()
        if not use_lectures:
            tmp_df.reset_index(drop=True, inplace=True)

        tmp_df["index"] = tmp_df.index.values.astype(np.uint32)
        user_id_index = tmp_df.groupby("user_id")["index"].apply(np.array)

        # iterate over users in random order
        for indices in user_id_index.sample(user_id_index.size, random_state=69):
            if len(train_idxs) > train_size:
                break
            # fill validation data
            if len(val_idxs) < validation_size:
                # add new user
                if np.random.rand() < new_user_prob:
                    val_idxs += list(indices)

                # randomly split user between train and val otherwise
                else:
                    offset = np.random.randint(0, indices.size)
                    train_idxs += list(indices[:offset])
                    val_idxs += list(indices[offset:])
            else:
                train_idxs += list(indices)

        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/train_{train_size}.npy",
            train_idxs,
        )
        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/val_{validation_size}.npy",
            val_idxs,
        )
    return train_idxs, val_idxs


def get_dataloaders(
    batch_size=1024,
    max_window_size=100,
    use_lectures=True,
    num_workers=0,
    min_multiple=None,
):

    print("Reading pickle")
    df = pd.read_pickle(f"{get_wd()}riiid_train.pkl.gzip")

    if not use_lectures:
        print("Removing lectures")
        df = df[~df.content_type_id]
        h5_file_name = f"{get_wd()}feats_no_lec.h5"
    else:
        h5_file_name = f"{get_wd()}feats.h5"

    generate_h5(df, file_name=h5_file_name)

    print("Creating Dataset")
    user_mapping = df.user_id.values
    user_history = df.groupby("user_id").cumcount().values + 1
    dataset = RIIDDataset(
        user_mapping,
        user_history,
        hdf5_file=h5_file_name,
        window_size=max_window_size,
        use_cache=True,
    )
    print(f"len(dataset): {len(dataset)}")

    print("Creating Train/Split")
    q_train_indices, q_valid_indices = get_train_val_idxs(df, use_lectures=use_lectures)

    print(f"len(q_train_indices): {len(q_train_indices)}")
    print(f"len(q_valid_indices): {len(q_valid_indices)}")

    # Init DataLoader from RIIID Dataset subset
    train_loader = DataLoader(
        dataset=Subset(dataset, q_train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(min_multiple=min_multiple),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # if GPU then pin memory for perf
    )
    val_loader = DataLoader(
        dataset=Subset(dataset, q_valid_indices),
        batch_size=1024,
        collate_fn=get_collate_fn(min_multiple=min_multiple),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    del df

    return train_loader, val_loader
