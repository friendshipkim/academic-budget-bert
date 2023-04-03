import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

# default args
data_path = "/n/tata_ddos_ceph/woojeong/data/output"
split = 4
n_shards = 256


def get_file_count(path):
    file_list = os.listdir(path)

    # check duplicates
    index_list = sorted(
        [int(filename.replace(".hdf5", "").split("_")[2]) for filename in file_list]
    )
    duplicates = [index for index in index_list if index_list.count(index) > 1]
    assert len(duplicates) == 0, f"Duplicate files exist: {duplicates}"
    assert list(range(len(index_list))) == index_list, "Filenames are not sequential"

    return len(index_list)


def parse_args():
    parser = argparse.ArgumentParser()
    # path to the whole data
    parser.add_argument("--data_path", default=data_path, type=str)
    # the number of splits
    parser.add_argument("--split", default=split, type=int)
    # the number of training shards
    parser.add_argument("--n_shards", default=n_shards, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"The number of training shards: {args.n_shards}")

    # all files
    src_path = os.path.join(args.data_path, "train_128_20_total")
    train_count = get_file_count(src_path)
    print(
        f"Total training hdf5 files: {train_count} -> Splitting into {args.split} sets"
    )

    # sampling is repeated every n_shards (0/256/512... are sampled from the same txt file)
    # to split hdf5 files into disjoint sets, n_shards // split
    shard_splits = np.array_split(range(n_shards), split)

    # make dirs and copy files
    for i in range(args.split):
        # choose the files sampled from disjoint shards
        train_split = []
        for split_start in shard_splits[i]:
            train_split.append(np.arange(split_start, train_count, n_shards))
        train_split = np.concatenate(train_split)

        # make dirs
        dst_path = os.path.join(args.data_path, f"train_128_20_set{i}")
        os.makedirs(dst_path, exist_ok=True)
        print(f"created {dst_path} directory")

        # copy train shards
        for train_file in tqdm(train_split):
            shutil.copy(
                os.path.join(src_path, f"train_shard_{train_file}.hdf5"), dst_path
            )
    print("Completed")
