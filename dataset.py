import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import h5py

from preprocessing import (
    questions_lectures_tags,
    questions_lectures_parts,
    questions_lectures_mean,
    questions_lectures_std,
    questions_lectures_wass,
    questions_lectures_pct,
    generate_h5,
)
from utils import get_wd

from preprocessing import preprocess_df


two_hours = 2 * 60 * 60 * 1000
eps = 0.0000001


def get_time_elapsed_from_timestamp(arr):
    arr_seconds = np.diff(arr, prepend=0) / 1000
    return (np.log(arr_seconds + eps).astype(np.float32) - 3.5) / 20


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


def rolling_mean(a, n=500):
    """
    Does mean over a rolling window size (n)
    """
    if n == 0:
        return np.zeros(a.size)
    ret = np.cumsum(np.pad(a, (n - 1, 0), "constant"), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def rolling_mean_2d(a, n=500):
    """
    Does mean over a rolling window size (n)
    a is size (N, E) - where N is the axis to cumsum over
    """
    if n == 0:
        return np.zeros(a.shape)
    ret = np.cumsum(np.pad(a, ((n - 1, 0), (0, 0)), "constant"), dtype=float, axis=0)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    return ret[n - 1 :, :] / n


def rolling_mean_over_time(timestamps, arr, time_step=two_hours):
    """
    Calculates mean over array using timestamp and timestep
    Uses a diff of time_step to start new section to average over
    """
    start_idxs = np.diff(timestamps, prepend=0) > time_step
    start_idxs[0] = True
    idxs = np.arange(len(start_idxs))
    idxs[~start_idxs] = 0
    start_idxs = np.maximum.accumulate(idxs)

    sums = arr.cumsum()
    sums = np.where(start_idxs == 0, sums, sums - sums[start_idxs - 1])

    counts = np.arange(len(timestamps))
    counts = counts + 1 - start_idxs[counts]

    return sums / counts


def get_arr_mean_feats(q_idx, arr, windows=[]):

    m_arr_mean = np.zeros((len(arr), len(windows) + 1))

    # content mean
    m_arr_mean[:, 0][q_idx] = arr[q_idx].cumsum() / (np.arange(len(q_idx)) + 1)

    # calculate rolling mean for window sizes 500, 1k and 2.5k
    for i in range(1, len(windows) + 1):
        m_arr_mean[:, i][q_idx] = rolling_mean([q_idx], n=windows[i - 1])

    return m_arr_mean


def get_parts_agg_feats(q_idx, parts, answered_correctly):
    # select parts for only questions
    p = parts[q_idx].astype(int)

    # get running counts of parts answered
    count_parts = np.zeros((p.size, 7), dtype=np.int)
    count_parts[np.arange(p.size, dtype=np.int), p - 1] = 1
    count_parts = count_parts.cumsum(axis=0)

    # get running sum of answered correctly for each part
    answered_correctly_parts = np.zeros((p.size, 7))
    answered_correctly_parts[np.arange(p.size), p - 1] = answered_correctly[q_idx]

    # answered_correctly_parts average
    parts_aggs = np.zeros((len(parts), 7), dtype=np.float)
    parts_aggs[q_idx] = answered_correctly_parts.cumsum(axis=0) / ((count_parts + 1))
    return parts_aggs

def get_agg_feats(content_ids, answered_correctly, parts, timestamps):

    # Calculate mean agg feats
    # question idx
    q_idx = np.where(answered_correctly != 4)[0]

    # user mean
    m_user_mean = get_arr_mean_feats(q_idx, answered_correctly)

    # prevent leak
    m_user_mean = np.roll(m_user_mean, 1, axis=0)
    m_user_mean[0] = 0

    # part mean
    parts_mean = get_parts_agg_feats(q_idx, parts, answered_correctly)

    # prevent leak
    parts_mean = np.roll(parts_mean, 1, axis=0)
    parts_mean[0] = 0

    return np.concatenate(
        [
            m_user_mean,
            parts_mean
        ],
        axis=1,
    )


def get_exercises_feats(content_ids):
    e_feats = np.zeros((len(content_ids), 4))
    e_feats[:, 0] = questions_lectures_mean[content_ids]
    e_feats[:, 1] = questions_lectures_std[content_ids]
    e_feats[:, 2] = questions_lectures_wass[content_ids]
    e_feats[:, 3] = questions_lectures_pct[content_ids]
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
        use_e_feats=True,
        content_ids=None,
        answered_correctly=None,
        timestamps=None,
        prior_question_elapsed_time=None,
        read_file=True,
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
        self.user_history = user_history.astype(np.int16)
        self.read_file = read_file
        if not read_file and (content_ids is None):
            raise ValueError(
                "If not reading file, must pass in values for content_ids, answers, timestamps, and prior_question_elapsed"
            )

        self.content_ids = content_ids.astype(np.int16)
        self.answered_correctly = answered_correctly.astype(np.int8)
        self.timestamps = timestamps.astype(np.float32)
        self.prior_question_elapsed_time = prior_question_elapsed_time.astype(
            np.float32
        )

        self.hdf5_file = f"{hdf5_file}"
        self.max_window_size = window_size
        self.use_agg_feats = use_agg_feats
        self.use_e_feats = use_e_feats

    def open_hdf5(self):
        # opens the h5py file
        self.f = h5py.File(self.hdf5_file, "r")

    def __len__(self):
        return len(self.user_mapping)

    def get_user_raw(self, idx):
        """
        reads a user's series until that
        """
        user_id = self.user_mapping[idx]
        length = self.user_history[idx]

        if self.read_file:
            if not hasattr(self, "f"):
                self.open_hdf5()
            content_ids = np.array(self.f[f"{user_id}/content_ids"], dtype=np.int64)
            answered_correctly = np.array(
                self.f[f"{user_id}/answered_correctly"], dtype=np.int64
            )
            timestamps = np.array(self.f[f"{user_id}/timestamps"], dtype=np.float32)
            prior_question_elapsed_time = np.array(
                self.f[f"{user_id}/prior_question_elapsed_time"], dtype=np.float32
            )
        else:
            content_ids = self.content_ids[idx - length + 1 : idx + 1]
            answered_correctly = self.answered_correctly[idx - length + 1 : idx + 1]
            timestamps = self.timestamps[idx - length + 1 : idx + 1]
            prior_question_elapsed_time = self.prior_question_elapsed_time[
                idx - length + 1 : idx + 1
            ]
        return (
            content_ids,
            answered_correctly,
            timestamps,
            prior_question_elapsed_time,
            length,
        )

    def __getitem__(self, idx):

        # open the hdf5 file in the iterator to allow multiple workers
        # https://github.com/pytorch/pytorch/issues/11929

        if torch.is_tensor(idx):
            idx = idx.tolist()

        (
            content_ids,
            answered_correctly,
            timestamps,
            prior_question_elapsed_time,
            length,
        ) = self.get_user_raw(idx)
        window_size = min(self.max_window_size, length)

        # index for loading larger than window size
        start_index = 0
        if length > window_size:
            # randomly select window size subset instead of trying to cram in everything
            start_index = length - window_size

        content_ids = content_ids[: start_index + window_size].copy()
        answered_correctly = answered_correctly[: start_index + window_size].copy()
        timestamps = timestamps[: start_index + window_size].copy()

        # get question parts
        parts = questions_lectures_parts[content_ids]

        # attempts of question id
        attempts = np.ones(len(content_ids)) * 5
        q_idx = np.where(answered_correctly != 4)[0]
        attempts[q_idx] = cumcount(content_ids[q_idx]).clip(max=4, min=0)
        attempts = attempts.astype(np.int8)
        attempts = attempts[start_index:]

        agg_feats = None
        if self.use_agg_feats:
            agg_feats = get_agg_feats(
                content_ids, answered_correctly, parts, timestamps
            )
            agg_feats = agg_feats[start_index:]

        e_feats = None
        if self.use_e_feats:
            # exercise feats
            e_feats = get_exercises_feats(content_ids)
            e_feats = e_feats[start_index:]

        # select to proper length
        parts = parts[start_index:]
        content_ids = content_ids[start_index:]

        # will replace first element of sequence with actual previous element
        if start_index > 0:
            start_token = answered_correctly[start_index - 1]
        else:
            # set start token if start_index is actually first element
            start_token = 2

        answered_correctly = answered_correctly[start_index:]

        timestamps = timestamps[start_index:]

        # load in time stuff
        prior_q_times = prior_question_elapsed_time[
            start_index : start_index + window_size
        ].copy()

        # convert timestamps to time elapsed
        time_elapsed_timestamps = get_time_elapsed_from_timestamp(timestamps)

        # get question tags
        tags = questions_lectures_tags[content_ids, :]

        # shift by one the answered_correctly sequence
        answers = np.roll(answered_correctly, 1)
        answers[0] = start_token

        return {
            "parts": torch.from_numpy(parts).long(),
            "tags": torch.from_numpy(tags).long(),
            "content_ids": torch.from_numpy(content_ids).long(),
            "answered_correctly": torch.from_numpy(answered_correctly).float(),
            "answers": torch.from_numpy(answers).long(),
            "timestamps": torch.from_numpy(time_elapsed_timestamps).float(),
            "prior_q_times": torch.from_numpy(prior_q_times).float(),
            "attempts": torch.from_numpy(attempts).long(),
            "agg_feats": torch.from_numpy(agg_feats).float()
            if agg_feats is not None
            else agg_feats,
            "e_feats": torch.from_numpy(e_feats).float()
            if e_feats is not None
            else e_feats,
            "length": window_size,
        }


def get_collate_fn(use_agg_feats=True, use_e_feats=True):
    def collate_fn(batch):
        """
        The collate function is used to merge individual data samples into a batch
        It handles the padding aspect
        """

        # collate lenghts into 1D tensor
        items = {
            "length": torch.tensor(
                [batch_item["length"] for batch_item in batch], dtype=torch.long
            )
        }

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
            ("attempts", 5),
        ]

        if use_agg_feats:
            PADDING_LIST.append(("agg_feats", 0))
        if use_e_feats:
            PADDING_LIST.append(("e_feats", 0))

        # padding list
        for (key, padding) in PADDING_LIST:
            items[key] = pad_sequence(
                [batch_item[key] for batch_item in batch],
                batch_first=False,
                padding_value=padding,
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
    validation_size=2500000,
    new_user_prob=0.25,
    use_lectures=True,
):
    # except FileNotFoundError:
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

    return train_idxs, val_idxs


def get_dataloaders(
    batch_size=1024,
    validation_batch_size=1024,
    max_window_size=100,
    use_lectures=True,
    use_agg_feats=True,
    use_e_feats=True,
    read_file=False,
    num_workers=0,
):

    print("Reading pickle")
    df = pd.read_pickle(f"{get_wd()}riiid_train.pkl.gzip")

    if read_file:
        if not use_lectures:
            print("Removing lectures")
            df = df[~df.content_type_id]
            h5_file_name = f"{get_wd()}feats_no_lec.h5"
        else:
            h5_file_name = f"{get_wd()}feats.h5"
        generate_h5(df, file_name=h5_file_name)
    else:
        print("Preprocessing df")
        h5_file_name = ""
        df = preprocess_df(df)
        df.answered_correctly.replace(
            -1, 4, inplace=True
        )  # set lecture to token 4 for answered correctly

    print("Creating Dataset")
    user_mapping = df.user_id.values
    user_history = df.groupby("user_id").cumcount().values + 1

    dataset = RIIDDataset(
        user_mapping,
        user_history,
        hdf5_file=h5_file_name,
        window_size=max_window_size,
        use_agg_feats=use_agg_feats,
        use_e_feats=use_e_feats,
        read_file=read_file,
        content_ids=df.content_id.values,
        answered_correctly=df.answered_correctly.values,
        timestamps=df.timestamp.values,
        prior_question_elapsed_time=df.prior_question_elapsed_time.values,
    )
    print(f"len(dataset): {len(dataset)}")

    # print("Select user history with less than 256 rows")
    # df = df[user_history <= 256]

    print("Creating Train/Split")
    q_train_indices, q_valid_indices = get_train_val_idxs(df, use_lectures=use_lectures)

    print(f"len(q_train_indices): {len(q_train_indices)}")
    print(f"len(q_valid_indices): {len(q_valid_indices)}")

    # Init DataLoader from RIIID Dataset subset
    train_loader = DataLoader(
        dataset=Subset(dataset, q_train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(use_agg_feats=use_agg_feats, use_e_feats=use_e_feats),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # if GPU then pin memory for perf
    )
    val_loader = DataLoader(
        dataset=Subset(dataset, q_valid_indices),
        batch_size=validation_batch_size,
        collate_fn=get_collate_fn(use_agg_feats=use_agg_feats, use_e_feats=use_e_feats),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    del df

    return train_loader, val_loader
