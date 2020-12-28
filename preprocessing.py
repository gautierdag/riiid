import os.path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

from scipy.stats import wasserstein_distance
from utils import get_wd

DATA_FOLDER_PATH = "data"


def get_questions_lectures_parts():
    try:
        questions_lectures_parts = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_parts.npy"
        )
    except FileNotFoundError:
        questions_parts = np.load(f"{get_wd()}{DATA_FOLDER_PATH}/questions_parts.npy")
        lectures_parts = np.load(f"{get_wd()}{DATA_FOLDER_PATH}/lectures_parts.npy")
        questions_lectures_parts = np.concatenate([questions_parts, lectures_parts])
        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_parts.npy",
            questions_lectures_parts,
        )
    return questions_lectures_parts


def get_questions_lectures_tags():
    try:
        questions_lectures_tags = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_tags.npy"
        )
    except FileNotFoundError:
        lectures_df = pd.read_csv(f"{get_wd()}{DATA_FOLDER_PATH}/lectures.csv")
        questions_df = pd.read_csv(f"{get_wd()}{DATA_FOLDER_PATH}/questions.csv")

        # process tags
        def split_tags(t):
            try:
                return [int(i) for i in t.split(" ")]
            except AttributeError:
                return list()

        # Get tags to be 2D array of shape (Q, T), where Q is question_idx, and T is the max number of tag possible (6)
        questions_df["tags"] = questions_df.tags.apply(split_tags)
        questions_tags = pd.DataFrame(
            questions_df["tags"].tolist(), index=questions_df.index
        )
        lectures_tags = pd.DataFrame(
            lectures_df.tag.values, index=lectures_df.index.values + 13523
        )

        questions_lectures_tags = pd.concat([questions_tags, lectures_tags])
        # pad with max tag + 1
        questions_lectures_tags = (
            questions_lectures_tags.fillna(questions_lectures_tags.max().max() + 1)
            .astype(np.int)
            .values
        )

        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_tags.npy",
            questions_lectures_tags,
        )
    return questions_lectures_tags


def get_lectures_mapping():
    try:
        lectures_mapping = pickle.load(
            open(f"{get_wd()}{DATA_FOLDER_PATH}/lectures_mapping.p", "rb")
        )
    except FileNotFoundError:
        lectures_df = pd.read_csv(f"{get_wd()}{DATA_FOLDER_PATH}/lectures.csv")
        # map lecture id to new id
        lectures_mapping = dict(
            zip(lectures_df.lecture_id.values, (lectures_df.index + 13523).values)
        )
        pickle.dump(
            lectures_mapping,
            open(f"{get_wd()}{DATA_FOLDER_PATH}/lectures_mapping.p", "wb"),
        )
    return lectures_mapping


def get_questions_lectures_mean():
    """
    Generates the mean accuracy obtained for each content id (0 for lectures)
    """
    try:
        content_mean = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_mean.npy"
        )
    except FileNotFoundError:
        print("Generating questions lectures mean")
        df = pd.read_pickle(f"{get_wd()}riiid_train.pkl.gzip")
        content_mean = (
            df[~df.content_type_id]
            .groupby("content_id")
            .answered_correctly.mean()
            .reset_index()
        )
        content_mean = np.concatenate(
            [content_mean.answered_correctly.values, np.zeros(418)]
        )
        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_mean.npy",
            content_mean,
        )
        del df

    return content_mean


def get_questions_lectures_std_wass():
    """
    Generates the std and wass distance between user_answers and actual answer on a question
    """
    try:
        questions_lectures_wass = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_wass.npy"
        )
        questions_lectures_std = np.load(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_std.npy"
        )
    except FileNotFoundError:
        print("Generating questions lectures std/wass")
        df = pd.read_pickle(f"{get_wd()}riiid_train.pkl.gzip")
        questions_df = pd.read_csv(f"{get_wd()}{DATA_FOLDER_PATH}/questions.csv")

        user_answer_counts = (
            df[~df.content_type_id][["content_id", "user_answer"]]
            .groupby(["content_id", "user_answer"])
            .user_answer.count()
        )

        user_answer_counts.name = "user_answer_count"
        user_answer_counts = user_answer_counts.reset_index()

        answer_counts = (
            user_answer_counts.groupby("content_id")
            .user_answer_count.sum()
            .reset_index()
            .rename(columns={"user_answer_count": "total_answers"})
        )
        user_answer_counts = pd.merge(
            answer_counts, user_answer_counts, on="content_id", how="inner"
        )
        user_answer_counts["user_answer_count"] = (
            user_answer_counts["user_answer_count"]
            / user_answer_counts["total_answers"]
        )
        user_answer_counts = pd.merge(
            questions_df[["question_id", "correct_answer"]],
            user_answer_counts,
            right_on="content_id",
            left_on="question_id",
            how="inner",
        ).drop(columns=["question_id"])

        user_answer_counts["correct"] = (
            user_answer_counts["correct_answer"] == user_answer_counts["user_answer"]
        ).astype(int)

        def earth_move_dist_with_norm(rows):
            d = wasserstein_distance(rows.user_answer_count.values, rows.correct.values)
            return d

        questions_lectures_wass = (
            user_answer_counts.groupby("content_id")
            .apply(lambda x: earth_move_dist_with_norm(x))
            .values
        )
        questions_lectures_wass = np.concatenate(
            [questions_lectures_wass, np.zeros(418)]
        )

        questions_lectures_std = (
            user_answer_counts.groupby("content_id").user_answer_count.apply(
                lambda x: np.std(x.values)
            )
        ) * 2  # times two so that max is close to 1

        questions_lectures_std = np.concatenate([questions_lectures_std, np.zeros(418)])

        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_wass.npy",
            questions_lectures_wass,
        )
        np.save(
            f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_std.npy",
            questions_lectures_std,
        )

        del df

    return questions_lectures_wass, questions_lectures_std


def get_questions_lectures_pct():
    questions_lectures_pct = np.load(
        f"{get_wd()}{DATA_FOLDER_PATH}/questions_lectures_pct.npy"
    )
    return questions_lectures_pct



lectures_mapping = get_lectures_mapping()
questions_lectures_parts = get_questions_lectures_parts()
questions_lectures_tags = get_questions_lectures_tags()
questions_lectures_mean = get_questions_lectures_mean()
questions_lectures_wass, questions_lectures_std = get_questions_lectures_std_wass()
questions_lectures_pct = get_questions_lectures_pct()


def preprocess_df(df):
    """
    Converts the lecture ids to proper content_ids
    Adds the answered_correctly column if not exists
    """
    df.content_type_id = df.content_type_id.astype(bool)

    # prior explanation
    df.prior_question_had_explanation = df.prior_question_had_explanation.fillna(False).astype(bool)

    # prior time
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


def generate_h5(df, file_name="feats.h5"):
    if os.path.isfile(file_name):
        return

    print("Generating feats h5")
    print("Preprocessing")
    df = preprocess_df(df)
    df.answered_correctly.replace(
        -1, 4, inplace=True
    )  # set lecture to token 4 for answered correctly

    print("Creating h5")
    hf = h5py.File(file_name, "w")
    for user_id, data in tqdm(df.groupby("user_id")):
        processed_feats = data[
            [
                "content_id",
                "answered_correctly",
                "timestamp",
                "prior_question_elapsed_time"
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
            data=data['prior_question_had_explanation'],
            maxshape=(None,),
        )

    hf.close()
