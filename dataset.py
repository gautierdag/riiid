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
    questions_lectures_mean,
    questions_lectures_std,
    questions_lectures_wass,
    generate_h5,
    DATA_FOLDER_PATH,
)
from utils import get_wd


import math
import torch.nn.functional as F


def pad_to_multiple(tensor, multiple=128, pad_value=0):
    m = tensor.shape[0] / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - tensor.shape[0]
    pad_offset = (0,) * 2 * (len(tensor.shape) - 1)
    return F.pad(tensor, (*pad_offset, 0, remainder), value=pad_value)


def get_parts_agg_feats(q_idx, parts, answered_correctly):
    parts_aggs = np.zeros((len(parts), 7))

    # select parts for only questions
    p = parts[q_idx]

    # get running counts of parts answered
    count_parts = np.zeros((p.size, 7))
    count_parts[np.arange(p.size), p - 1] = 1
    count_parts = count_parts.cumsum(axis=0)

    # get running sum of answered correctly for each part
    answered_correctly_parts = np.zeros((p.size, 7))
    answered_correctly_parts[np.arange(p.size), p - 1] = answered_correctly[q_idx]
    answered_correctly_parts = answered_correctly_parts.cumsum(axis=0)

    parts_aggs[q_idx, :] = answered_correctly_parts / (count_parts + 1)

    return parts_aggs


def dfill(a):
    n = a.size
    b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
    return np.arange(n)[b[:-1]].repeat(np.diff(b))


def argunsort(s):
    n = s.size
    u = np.empty(n, dtype=np.int64)
    u[s] = np.arange(n)
    return u


def cumcount(a):
    n = a.size
    s = a.argsort(kind="mergesort")
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]


def get_agg_feats(content_ids, answered_correctly, parts):

    # Calculate agg feats
    # question idx
    q_idx = np.where(answered_correctly != 4)[0]

    # attempts of question id
    attempts = np.zeros(len(content_ids))
    attempts[q_idx] = cumcount(content_ids[q_idx]).clip(max=5, min=0) / 5

    # content mean
    m_content_mean = np.zeros(len(content_ids))
    m_content_mean[q_idx] = questions_lectures_mean[content_ids][q_idx].cumsum() / (
        np.arange(len(q_idx)) + 1
    )

    # user mean
    m_user_mean = np.zeros(len(content_ids))
    m_user_mean[q_idx] = answered_correctly[q_idx].cumsum() / (
        np.arange(len(q_idx)) + 1
    )

    # prevent leak
    m_user_mean = np.roll(m_user_mean, 1)
    m_user_mean[0] = 0

    # part mean
    parts_mean = get_parts_agg_feats(q_idx, parts, answered_correctly)

    # prevent leak
    parts_mean = np.roll(parts_mean, 1, axis=0)
    parts_mean[0] = 0

    return np.concatenate(
        [
            attempts[..., np.newaxis],
            m_content_mean[..., np.newaxis],
            m_user_mean[..., np.newaxis],
            parts_mean,
        ],
        axis=1,
    )


def get_exercises_feats(content_ids):
    e_feats = np.zeros((len(content_ids), 3))
    e_feats[:, 0] = questions_lectures_mean[content_ids]
    e_feats[:, 1] = questions_lectures_std[content_ids]
    e_feats[:, 2] = questions_lectures_wass[content_ids]
    return e_feats


class RIIDDataset(Dataset):
    """RIID dataset."""

    def __init__(
        self,
        user_mapping,
        user_history,
        hdf5_file="feats.h5",
        window_size=100,
        use_agg_feats=True,
    ):
        """
        Args:
            user_mapping (np.array): array of user_ids per row
            user_history (np.array): array of length of history up till this row
            hdf5_file (string): location of hf5 feats file
            window_size (int): size of window to lookback to max
        """
        # np array where index maps to a user id
        self.user_mapping = user_mapping
        self.user_history = user_history
        self.hdf5_file = f"{hdf5_file}"
        self.max_window_size = window_size
        self.use_agg_feats = use_agg_feats
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

        if user_id not in self.cache:
            self.load_user_into_cache(user_id)

        content_ids = self.cache[user_id]["content_ids"][
            : start_index + window_size
        ].copy()
        answered_correctly = self.cache[user_id]["answered_correctly"][
            : start_index + window_size
        ].copy()

        # get question parts
        parts = questions_lectures_parts[content_ids]

        agg_feats = None
        if self.use_agg_feats:
            agg_feats = get_agg_feats(content_ids, answered_correctly, parts)
            agg_feats = agg_feats[start_index:]

        # select to proper length
        parts = parts[start_index:]
        content_ids = content_ids[start_index:]
        answered_correctly = answered_correctly[start_index:]

        # exercise feats
        e_feats = get_exercises_feats(content_ids)

        # load in time stuff
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

        # shift by one the answered_correctly sequence
        answers = np.roll(answered_correctly, 1)

        # set start token if start_index is actually first element
        if start_index == 0:
            answers[0] = 2
        # else replace first element of sequence with actual previous element
        else:
            answers[0] = self.cache[user_id]["answered_correctly"][start_index - 1]

        return {
            "parts": torch.from_numpy(parts).long(),
            "tags": torch.from_numpy(tags).long(),
            "content_ids": torch.from_numpy(content_ids).long(),
            "answered_correctly": torch.from_numpy(answered_correctly).float(),
            "answers": torch.from_numpy(answers).long(),
            "timestamps": torch.from_numpy(time_elapsed_timestamps).float(),
            "prior_q_times": torch.from_numpy(prior_q_times).float(),
            "agg_feats": torch.from_numpy(agg_feats).float()
            if agg_feats is not None
            else agg_feats,
            "e_feats": e_feats,
            "length": window_size,
        }


def get_collate_fn(min_multiple=None, use_agg_feats=True):
    def collate_fn(batch):
        """
        The collate function is used to merge individual data samples into a batch
        It handles the padding aspect
        """

        # collate lenghts into 1D tensor
        items = {"length": torch.tensor([batch_item["length"] for batch_item in batch])}

        # find shape that the batch will have
        num_items = len(batch)

        PADDING_LIST = [
            ("parts", 0),
            ("content_ids", 13942),
            ("answered_correctly", 3),
            ("answers", 3),
            ("timestamps", 0.0),  # note timestamps isnt an embedding
            ("tags", 188),
            ("prior_q_times", 0),
            ("e_feats", 0),
        ]

        if use_agg_feats:
            PADDING_LIST.append(("agg_feats", 0))

        # padding list
        for (key, padding) in PADDING_LIST:
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
        train_idxs = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/train_{train_size}_lec_{use_lectures}.npy"
        )
        val_idxs = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/val_{validation_size}_lec_{use_lectures}.npy"
        )

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
            f"{get_wd()}{DATA_FOLDER_PATH}/train_{train_size}_lec_{use_lectures}.npy",
            train_idxs,
        )
        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/val_{validation_size}_lec_{use_lectures}.npy",
            val_idxs,
        )
    return train_idxs, val_idxs


def get_dataloaders(
    batch_size=1024,
    validation_batch_size=1024,
    max_window_size=100,
    use_lectures=True,
    num_workers=0,
    min_multiple=None,
    use_agg_feats=True,
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
        use_agg_feats=use_agg_feats,
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
        collate_fn=get_collate_fn(
            min_multiple=min_multiple, use_agg_feats=use_agg_feats
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # if GPU then pin memory for perf
    )
    val_loader = DataLoader(
        dataset=Subset(dataset, q_valid_indices),
        batch_size=validation_batch_size,
        collate_fn=get_collate_fn(
            min_multiple=min_multiple, use_agg_feats=use_agg_feats
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    del df

    return train_loader, val_loader
