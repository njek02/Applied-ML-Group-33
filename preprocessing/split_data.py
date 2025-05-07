import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training, validation, and test sets using stratified sampling.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'label' column used for stratification.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training,
        validation, and test DataFrames respectively.
    """
    train_set, temp_set = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=1)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, stratify=temp_set['label'], random_state=1)

    return train_set, val_set, test_set


def save_data(train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
    """
    Saves the training, validation, and test DataFrames to CSV files.

    Args:
        train_set (pd.DataFrame): The training dataset.
        val_set (pd.DataFrame): The validation dataset.
        test_set (pd.DataFrame): The test dataset.
    """
    train_set.to_csv("data\\training_data.csv", index=False)
    val_set.to_csv("data\\validation_data.csv", index=False)
    test_set.to_csv("data\\test_data.csv", index=False)

    return


def move_subset(subset, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    files_moved = 0
    for file_name in subset["clip_name"]:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            files_moved += 1
            print(f"File {file_name} moved successfully")
        else:
            print(f"File {file_name} does not exist")

    print(f"Moved {files_moved} files")
    return


if __name__ == "__main__":
    df = pd.read_csv("data\\train.csv")
    train_set, val_set, test_set = split_data(df)

    save_data(train_set, val_set, test_set)

    source_dir = "data\\dataset"
    training_dir = "data\\training_data"
    validation_dir = "data\\validation_data"
    test_dir = "data\\test_data"

    move_subset(train_set, source_dir, training_dir)
    move_subset(val_set, source_dir, validation_dir)
    move_subset(test_set, source_dir, test_dir)
