import numpy as np
import pandas as pd
import seaborn as sns

import math
import os
import sys

import h5py

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import seed_everything

from tqdm.auto import tqdm

eps = 0.0000001


folder_path = "data"
print("Loading lectures")
lectures_df = pd.read_csv("lectures.csv")
lectures_parts = np.load(f"{folder_path}/lectures_parts.npy")
lectures_types = np.load(f"{folder_path}/lectures_types.npy")
print("Loading questions")
questions_parts = np.load(f"{folder_path}/questions_parts.npy")
questions_df = pd.read_csv("questions.csv")

# process tags
def split_tags(t):
    try:
        return [int(i) for i in t.split(" ")]
    except AttributeError:
        return list()


questions_lectures_parts = np.concatenate([questions_parts, lectures_parts])

# Get tags to be 2D array of shape (Q, T), where Q is question_idx, and T is the max number of tag possible (6)
questions_df["tags"] = questions_df.tags.apply(split_tags)
questions_tags = pd.DataFrame(questions_df["tags"].tolist(), index=questions_df.index)

# map lecture id to new id
lectures_mapping = dict(
    zip(lectures_df.lecture_id.values, (lectures_df.index + 13523).values)
)
lectures_df.lecture_id = lectures_df.index + 13523
lectures_tags = pd.DataFrame(
    lectures_df.tag.values, index=lectures_df.lecture_id.values
)

questions_lectures_tags = pd.concat([questions_tags, lectures_tags])
# pad with max tag + 1
questions_lectures_tags = (
    questions_lectures_tags.fillna(questions_lectures_tags.max().max() + 1)
    .astype(np.int)
    .values
)


def preprocess_df(df):
    """
    Converts the lecture ids to proper content_ids
    Adds the answered_correctly column if not exists
    """
    df.content_type_id = df.content_type_id.astype(bool)

    # prior information
    df.prior_question_had_explanation = df.prior_question_had_explanation.astype(
        np.uint8
    )
    df.prior_question_elapsed_time = (
        df.prior_question_elapsed_time.fillna(0).clip(upper=300000) / 300000
    )  # normalizes to 0-1

    # map lecture ids to new content_ids
    df.loc[df.content_type_id, "content_id"] = df[df.content_type_id].content_id.map(
        lectures_mapping
    )
    # if not answered correctly then add column with
    # y = 3 (padding) for all questions and y = 4 for lectures
    if "answered_correctly" not in df.columns:
        df["answered_correctly"] = df.content_type_id.map({False: 3, True: 4})

    return df


def get_time_elapsed_from_timestamp(arr):
    arr_seconds = np.diff(arr, prepend=0) / 1000
    return (np.log(arr_seconds + eps).astype(np.float32) - 3.5) / 20


def generate_h5(df, file_name="train_feats.h5"):
    df = preprocess_df(df)
    df.answered_correctly.replace(
        -1, 4, inplace=True
    )  # set lecture to token 4 for answered correctly

    hf = h5py.File(file_name, "w")

    for user_id, data in tqdm(df.groupby("user_id")):
        processed_feats = data[
            [
                "content_id",
                "answered_correctly",
                "timestamp",
                "prior_question_elapsed_time",
                "prior_question_had_explanation",
            ]
        ].values

        hf.create_dataset(
            f"{user_id}/content_ids", data=processed_feats[:, 0], maxshape=(None,)
        )
        hf.create_dataset(
            f"{user_id}/answered_correctly",
            data=processed_feats[:, 1],
            maxshape=(None,),
        )
        hf.create_dataset(
            f"{user_id}/timestamps", data=processed_feats[:, 2], maxshape=(None,)
        )
        hf.create_dataset(
            f"{user_id}/prior_question_elapsed_time",
            data=processed_feats[:, 3],
            maxshape=(None,),
        )
        hf.create_dataset(
            f"{user_id}/prior_question_had_explanation",
            data=processed_feats[:, 4],
            maxshape=(None,),
        )

    hf.close()