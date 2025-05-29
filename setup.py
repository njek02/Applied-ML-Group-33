from whale_call_project.data_loading.split_data import split_data_folder

# Setup file to move .aiff data into training, validation and test sets

if __name__ == "__main__":
    csv_source_path = "data/whale_data/data/train.csv"  # Replace this path with your location of train.csv
    dataset_source_path = "data/whale_data/data/train"  # Replace this path with your location of the training data
    split_data_folder(csv_source_path, dataset_source_path)
