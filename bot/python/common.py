import pickle
import argparse
import subprocess
import re
import gzip
import os
import random
import torch
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameters(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name}: {p.numel()}")


def split_into_batches(l, batch_size):
    """
    :param l:           list
    :param batch_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), batch_size):
        yield l[i:min(len(l), i + batch_size)]


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_cached_tensors(caching_filename, memory, test_memory):
    with gzip.open(caching_filename, 'rb') as f:
        print("Loading cached tensors")
        chunks = pickle.load(f)
        for i in range(chunks):
            items1 = pickle.load(f)
            for item in items1:
                memory.push(item)

        print("Loading cached test tensors")
        items2 = pickle.load(f)
        for item in items2:
            test_memory.push(item)


def cache_tenors(data_paths, cache_dir, small_input, load_state_fn, version):
    os.makedirs(cache_dir, exist_ok=True)

    def save_path(path):
        name = path.split("/")[-1].split(".")[0]
        final_path = os.path.join(cache_dir, f"{name}.{version}.pt")
        return final_path

    def should_process(path):
        return not os.path.exists(save_path(path))

    def clear_old_tensors():
        for path in os.listdir(cache_dir):
            if not path.endswith(f".{version}.pt"):
                print(f"Removing {path}")
                os.remove(os.path.join(cache_dir, path))

    load_all(data_paths, small_input, lambda s: load_state_fn(s, save_path(s["data_path"])), should_process)
    clear_old_tensors()


def load_all(data_paths, small_input, load_fn, filter_fn):
    print("Loading training data...")
    for data_path in data_paths:
        fs = os.listdir(data_path)
        # fs = natural_sort(fs)
        if small_input:
            fs = fs[:20]
        random.shuffle(fs)
        for i in range(len(fs)):
            print(f"\r{i}/{len(fs)}", end="")
            path = data_path + "/" + fs[i]
            if not filter_fn(path):
                continue

            try:
                with gzip.open(path, "rb") as f:
                    s = pickle.load(f)
            except Exception:
                print("Failed to load and deserialize", path)
                continue

            s["data_path"] = path
            load_fn(s)
        print("Done")


def save_cache(caching_filename, memory, test_memory):
    with gzip.open(caching_filename, 'wb') as f:
        batches = list(split_into_batches(memory.get_all(), 64))
        pickle.dump(len(batches), f, protocol=pickle.HIGHEST_PROTOCOL)
        for chunk in batches:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_memory.get_all(), f, protocol=pickle.HIGHEST_PROTOCOL)


def save_git(comment):
    orig_hash = subprocess.check_output(["git", "describe", "--always"]).decode('utf-8').strip()

    if subprocess.call(["git", "commit", "--allow-empty", "-a", "-m", comment]) != 0:
        print("Git commit failed")
        exit(1)

    hash = subprocess.check_output(["git", "describe", "--always"]).decode('utf-8').strip()

    if subprocess.call(["git", "reset", orig_hash]) != 0:
        print("Git reset failed")
        exit(1)

    comment = comment + " " + hash
    print(comment)
    return comment


def train_interface(cache_fn, train_fn, visualize_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--epoch", default=None, type=int)
    parser.add_argument("--comment", default=None, type=str)
    args = parser.parse_args()

    if args.save_cache:
        cache_fn()

    if args.train:
        args.comment = save_git(args.comment)
        if args.comment is None or len(args.comment) == 0:
            print("You need to supply a comment for the training run (--comment)")
            exit(1)
        train_fn(args.comment)

    if args.visualize:
        visualize_fn(args.epoch)


def pack_sequences(sequences):
    if len(sequences) == 0:
        return None

    res = torch.cat(sequences, dim=0)
    indices = []
    offset = 0
    for i in range(len(sequences)):
        indices.append((offset, offset + sequences[i].size()[0]))
        offset += sequences[i].size()[0]

    return res, indices


def unpack_sequences(packed_sequence, device=None, non_blocking=False):
    if packed_sequence is None:
        return None

    data = packed_sequence[0]
    if device is not None:
        data = data.to(device=device, non_blocking=non_blocking)
    indices = packed_sequence[1]
    sequences = [data[a:b] for a, b in indices]
    return sequences


class PadSequence:
    def __init__(self, is_sequence):
        self.is_sequence = is_sequence

        valid_values = {None, 'pad-timewise', 'stack-timewise', 'stack', 'pack-stack-timewise'}
        for x in is_sequence:
            assert x in valid_values, f"Expected is_sequence member to be one of {valid_values}, found {x}"

    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        # Get each sequence and pad it
        num_elements = len(sorted_batch[0])
        assert num_elements == len(self.is_sequence), f"Number of elements in sample tuple did not match \
            the length of the is_sequence parameter to PadSequence ({num_elements} != {len(self.is_sequence)})"

        result = []

        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(b[0]) for b in sorted_batch])

        result = []
        for i in range(num_elements):
            mode = self.is_sequence[i]
            if mode is None:
                result.append([x[i] for x in sorted_batch])
            elif mode == 'stack':
                result.append([x[i] for x in sorted_batch])
                result[-1] = torch.stack(result[-1])
            elif mode == 'stack-timewise' or mode == 'pad-timewise' or mode == 'pack-stack-timewise':
                result.append([])
            else:
                assert False

        for timestep in range(lengths[0]):
            in_progress_threshold = 0
            while in_progress_threshold < len(sorted_batch) and timestep < lengths[in_progress_threshold]:
                in_progress_threshold += 1

            active_batch = sorted_batch[:in_progress_threshold]

            for i in range(num_elements):
                mode = self.is_sequence[i]
                if mode == 'stack-timewise' or mode == 'pack-stack-timewise':
                    if active_batch[0][i] is not None:
                        s = [x[i][timestep] for x in active_batch]
                        s = torch.stack(s)
                        result[i].append(s)
                elif mode == 'pad-timewise':
                    s = [x[i][timestep] for x in active_batch if x[i] is not None]
                    if len(s) > 0:
                        s = torch.nn.utils.rnn.pad_sequence(s, batch_first=True)
                        result[i].append(s)

        for i in range(num_elements):
            mode = self.is_sequence[i]
            if mode == 'pack-stack-timewise':
                result[i] = pack_sequences(result[i])

        combined = type(sorted_batch[0])(*result)
        return combined, lengths


class TensorBoardWrapper:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None
        self.scalars = []

    def add_scalar(self, message, loss, step):
        if self.writer is not None:
            self.writer.add_scalar(message, loss, step)
        else:
            self.scalars.append((message, loss, step))

    def add_embedding(self, *args, **kwargs):
        self.writer.add_embedding(*args, **kwargs)

    def init(self):
        if self.writer is not None:
            return

        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for s in self.scalars:
            self.writer.add_scalar(*s)
        self.scalars = None

    def set_epoch(self, epoch):
        if epoch >= 2:
            self.init()


def training_loop(training_generator, testing_generator, trainer, tensorboard_writer):
    step = 0
    epoch = 0
    current_step = 0

    while True:
        epoch += 1
        if hasattr(tensorboard_writer, "set_epoch"):
            tensorboard_writer.set_epoch(epoch)

        print(f"\rTraining Epoch {epoch}", end="")
        yield epoch, current_step

        last_loss = 0
        num_batches = len(training_generator)
        for i, (batch_tuple, lengths) in enumerate(training_generator):
            with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
                batch = trainer.sampleClass(*batch_tuple)
                print(f"\rTraining Epoch {epoch} [{i+1}/{num_batches}] loss={last_loss}...", end="")
                loss = trainer.train(batch)
                print(f"\rTraining Epoch {epoch} [{i+1}/{num_batches}] loss={loss}", end="")
                last_loss = loss
                step += 1
                current_step = step
                tensorboard_writer.add_scalar("training loss", loss, step)

            # prof.export_chrome_trace("trace.prof")
            # print(prof.key_averages().table(sort_by="cpu_time_total"))

        print()

        total_loss = 0
        total_weight = 0
        for i, (batch_tuple, lengths) in enumerate(testing_generator):
            batch = trainer.sampleClass(*batch_tuple)
            loss = trainer.test(batch)
            # Use the number of timesteps as the weight. TODO: More accurate would be to use the total number of units in all the timesteps
            weight = np.array(lengths).sum()
            total_weight += weight
            total_loss += loss * weight

        total_loss /= total_weight

        print("Test loss:", total_loss)
        tensorboard_writer.add_scalar("test loss", total_loss, step)
