from preprocessing.split_data import split_data_folder

csv_source_path = "data/whale_data/data/train.csv"
dataset_source_path = "data/whale_data/data/train"
split_data_folder(csv_source_path, dataset_source_path)