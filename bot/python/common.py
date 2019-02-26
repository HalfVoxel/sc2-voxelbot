import pickle
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


def load_all(data_paths, small_input, load_fn):
    print("Loading training data...")
    for data_path in data_paths:
        fs = os.listdir(data_path)
        fs = natural_sort(fs)
        if small_input:
            fs = fs[:20]
        random.shuffle(fs)
        for i in range(len(fs)):
            print(f"\r{i}/{len(fs)}", end="")
            path = data_path + "/" + fs[i]
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


class PadSequence:
    def __init__(self, is_sequence=[True, False]):
        self.is_sequence = is_sequence

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
            if self.is_sequence[i]:
                result.append([])
            else:
                result.append([x[i] for x in sorted_batch])
                if isinstance(result[-1][0], torch.Tensor):
                    result[-1] = torch.stack(result[-1])

        for timestep in range(lengths[0]):
            in_progress_threshold = 0
            while in_progress_threshold < len(sorted_batch) and timestep < lengths[in_progress_threshold]:
                in_progress_threshold += 1

            active_batch = sorted_batch[:in_progress_threshold]

            for i in range(num_elements):
                if self.is_sequence[i]:
                    s = [x[i][timestep] for x in active_batch]
                    s = torch.stack(s)
                    result[i].append(s)

        # for i in range(num_elements):
        #     items = [x[i] for x in sorted_batch]
        #     if self.is_sequence[i] and False:
        #         sequences_padded = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)
        #         result.append(sequences_padded)
        #     else:
        #         result.append(items)
        #         # result.append(torch.stack(items))

        # combined = type(sorted_batch[0])(*result)
        combined = type(sorted_batch[0])(*result)
        return combined, lengths


def training_loop(training_generator, testing_generator, trainer, tensorboard_writer):
    step = 0
    epoch = 0
    current_step = 0

    while True:
        epoch += 1
        print(f"\rTraining Epoch {epoch}", end="")
        yield epoch, current_step

        last_loss = 0
        num_batches = len(training_generator)
        for i, (batch_tuple, lengths) in enumerate(training_generator):
            batch = trainer.sampleClass(*batch_tuple)
            print(f"\rTraining Epoch {epoch} [{i+1}/{num_batches}] loss={last_loss}...", end="")
            loss = trainer.train(batch)
            print(f"\rTraining Epoch {epoch} [{i+1}/{num_batches}] loss={loss}", end="")
            last_loss = loss
            step += 1
            current_step = step
            tensorboard_writer.add_scalar("training loss", loss, step)

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
