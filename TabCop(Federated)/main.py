import argparse
import time

import federated
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated TabCop.")
    parser.add_argument("-dataset_name", type=str, default="red")
    parser.add_argument("-balance", action="store_true")
    parser.add_argument("-imbalance", action="store_true")
    parser.add_argument("-cft_intervals", type=int, default=1000)
    parser.add_argument("-icft_intervals", type=int, default=1000)
    parser.add_argument("-epsilon", type=float, default=0)
    args = parser.parse_args()

    if args.balance:
        balance_data_list = []
        for i in range(3):
            balance_data_list.append(pd.read_csv(f"real_data/{args.dataset_name}_balance_{i}.csv"))
        start = time.time()
        sys_data_balance = federated.federated_synthesize(
            balance_data_list, args.cft_intervals, args.icft_intervals, args.epsilon
        )
        if args.epsilon == 0:
            sys_data_balance.to_csv(f"./sys_data/{args.dataset_name}_balance.csv", index=False)
            end = time.time()
            time_file = open("time.txt", "a")
            print(f"{args.dataset_name}_balance_time: %f" % (end - start), file=time_file)
            time_file.close()
        else:
            sys_data_balance.to_csv(
                f"./sys_data/{args.dataset_name}_balance_privacy.csv", index=False
            )
            end = time.time()
            time_file = open("time.txt", "a")
            print(f"{args.dataset_name}_balance_privacy_time: %f" % (end - start), file=time_file)
            time_file.close()
    if args.imbalance:
        imbalance_data_list = []
        for i in range(3):
            imbalance_data_list.append(
                pd.read_csv(f"real_data/{args.dataset_name}_imbalance_{i}.csv")
            )
        start = time.time()
        sys_data_balance = federated.federated_synthesize(
            imbalance_data_list, args.cft_intervals, args.icft_intervals, args.epsilon
        )
        if args.epsilon == 0:
            sys_data_balance.to_csv(f"./sys_data/{args.dataset_name}_imbalance.csv", index=False)
            end = time.time()
            time_file = open("time.txt", "a")
            print(f"{args.dataset_name}_imbalance_time: %f" % (end - start), file=time_file)
            time_file.close()
        else:
            sys_data_balance.to_csv(
                f"./sys_data/{args.dataset_name}_imbalance_privacy.csv", index=False
            )
            end = time.time()
            time_file = open("time.txt", "a")
            print(f"{args.dataset_name}_imbalance_privacy_time: %f" % (end - start), file=time_file)
            time_file.close()
