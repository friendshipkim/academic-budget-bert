import argparse
import glob
import os
import h5py
import pickle
import numpy as np
from tqdm import tqdm


def get_partial_numpy(lst_of_npy, num_rows):
    result = []
    current_rows = 0
    prev_shape = sum([x.shape[0] for x in lst_of_npy])
    n = len(lst_of_npy)
    # we parse the list in reverse order, since we want the most recently inserted array first
    for i, arr in enumerate(reversed(lst_of_npy)):
        if current_rows < num_rows and arr.shape[0] > 0:
            if arr.shape[0] + current_rows <= num_rows:
                result.append(arr)
                current_rows += arr.shape[0]
                lst_of_npy[n - i - 1] = np.zeros(0)
            else:
                result.append(arr[0 : (num_rows - current_rows)])
                lst_of_npy[n - i - 1] = lst_of_npy[n - i - 1][(num_rows - current_rows):]
                break
        elif current_rows > num_rows:
            raise ValueError("current rows cannot be greater than num_rows")

    curr_shape = sum([x.shape[0] for x in lst_of_npy])
    # print(prev_shape, curr_shape, sum([x.shape[0] for x in result]), num_rows)

    assert prev_shape - curr_shape == num_rows, f"num_rows: {num_rows}, prev_shape - curr_shape: {prev_shape - curr_shape} "
    assert sum([x.shape[0] for x in result]) == num_rows, f"num_rows: {num_rows}, result: {sum([x.shape[0] for x in result])}"
    return np.vstack(result)


def find_num_samples_in_shard(files):
    lengths = []
    print("Finding optimal length of each shard")
    for file in tqdm((files)):
        f = h5py.File(file, "r")
        lengths.append((np.asarray(f[list(f.keys())[0]][:]).shape[0], file))
        f.close()
    lengths = sorted(lengths, key=lambda x: -x[0])
    return lengths


def balance(files, out_dir, lengths_file=None):
    if lengths_file is None:
        lengths = find_num_samples_in_shard(files)
        # # Save as pickle file (optional)
        # with open('total_lengths.pkl', 'wb') as f:
        #     pickle.dump(lengths, f)
    else:
        with open(lengths_file, 'rb') as f:
            lengths = pickle.load(f)
        
    if len(lengths) == 0:
        return
    average_length = sum([x[0] for x in lengths]) // len(lengths)
    print("*" * 72)
    print("total length:", sum([x[0] for x in lengths]))
    print("average length:", average_length)
    print("file count:", len(lengths))
    
    ## We have to re-distribute the shards so that each shard contains equal number of samples.
    ## Since we are operating on tokenised data, re-distributing should not alter the perfornance (IID assumption)
    
    file_content = {}
    print("Re sharding data into equal blocks")
    for length, file in tqdm(lengths):
        f = h5py.File(file, "r")
        keys = list(f.keys())
        data_type = {}
        for key in keys:
            data_type[key] = f[key].dtype
            if key not in file_content:
                file_content[key] = [np.asarray(f[key][:])]
            else:
                file_content[key].append(np.asarray(f[key][:]))
                # file_content acts like a stack containing the most recent data that has been read
        f.close()
        fout = h5py.File(
            os.path.join(out_dir, os.path.basename(file)), "w", libver="latest"
        )
        
        for key in keys:
            fout.create_dataset(
                key,
                data=get_partial_numpy(file_content[key], average_length),
                dtype=data_type[key],
                compression="gzip",
            )
            # remove the #average_length data points from `file_content`
            # since they have already been writtent into shards
        fout.close()


def main(files, out_dir, train_lengths, test_lengths):
    train_files = [file for file in files if "train_shard" in file]
    test_files = [file for file in files if "test_shard" in file]
    balance(train_files, out_dir, train_lengths)
    balance(test_files, out_dir, test_lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--train-lengths", type=str, required=False, default=None)
    parser.add_argument("--test-lengths", type=str, required=False, default=None)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.dir, "*"))
    main(files, args.out_dir, args.train_lengths, args.test_lengths)
