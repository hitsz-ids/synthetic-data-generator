import argparse

import pandas as pd


def split(data, num_list):
    start = 0
    data_list = []
    for i in range(len(num_list)):
        if i == len(num_list) - 1:
            data_list.append(data.iloc[start:, :])
        else:
            data_list.append(data.iloc[start : start + num_list[i], :])
        start += num_list[i]
    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitting the dataset.")
    parser.add_argument("-data_file", type=str, default="red_train.csv")
    parser.add_argument("-dataset_name", type=str, default="red")
    args = parser.parse_args()

    train = pd.read_csv(f"./real_data/{args.data_file}")
    balance_data_list = split(train.copy(), [int(len(train) / 3)] * 3)
    imbalance_data_list = split(train.copy(), [100, 200, len(train) - 100 - 200])
    for i in range(len(balance_data_list)):
        balance_data_list[i].to_csv(
            f"./real_data/{args.dataset_name}_balance_" + str(i) + ".csv", index=False
        )
    for i in range(len(imbalance_data_list)):
        imbalance_data_list[i].to_csv(
            f"./real_data/{args.dataset_name}_imbalance_" + str(i) + ".csv", index=False
        )
