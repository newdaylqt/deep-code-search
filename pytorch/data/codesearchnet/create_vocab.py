from collections import defaultdict
import tables
import pickle
import numpy as np

vocab_name = dict()
vocab_api = dict()
vocab_token = dict()
vocab_desc = dict()

SE_IDX = 0
UNK_IDX = 1

def get_vocab(file_name, part_name):
    vocab = defaultdict(lambda: 0)

    with open(file_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    # count vocab
    for line in lines:
        line = line[:-1]
        tokens = line.split(' ')
        for token in tokens:
            vocab[token] += 1
    
    # sort vocab
    vocab = sorted(vocab.items(), key=lambda v: v[1], reverse=True)
    vocab_with_rank = [("<s>", 0, -1), ("</s>", 0, -1), ("UNK", 1, -1)]
    vocab_indices = dict()
    vocab_indices["<s>"] = 0
    vocab_indices["</s>"] = 0
    vocab_indices["UNK"] = 1
    vocab_indices_10000 = dict()
    vocab_indices_10000["<s>"] = 0
    vocab_indices_10000["</s>"] = 0
    vocab_indices_10000["UNK"] = 1
    for idx, v in enumerate(vocab):
        vocab_with_rank.append((v[0], idx+2, v[1]))
        vocab_indices[v[0]] = idx+2
    for idx, v in enumerate(vocab[:9999]):
        vocab_with_rank.append((v[0], idx+2, v[1]))
        vocab_indices_10000[v[0]] = idx+2

    # save vocab: 1) vocab with frequency; 2) vocab for use
    with open("vocab_with_freq.txt", 'w') as f:
        for v in vocab_with_rank:
            f.write("{:<20s}{:<10d}{}\n".format(v[0], v[1], v[2]))
    with open("vocab.{}.pkl".format(part_name), 'wb') as f:
        pickle.dump(vocab_indices_10000, f)
    
    return vocab_indices_10000


class Index(tables.IsDescription):
    length = tables.Int64Col()
    pos = tables.Int64Col()


def get_sequence_and_indices(file_name, part_name, partition, vocab):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # get token index sequence
    sequence = []
    for line in lines:
        tokens = line.split(' ')
        for token in tokens:
            if token in vocab:
                sequence.append(vocab[token])
            else:
                sequence.append(UNK_IDX)
            # idx = vocab[token]
            # sequence.append(UNK_IDX if idx == -1 else idx)

    # get sample start/end index in the whole sequence
    indices = [(0, 0)]
    for line in lines:
        tokens = line.split(' ')
        last_ind = indices[-1][0] + indices[-1][1]
        indices.append((len(tokens), last_ind))

    return sequence, indices[1:]


def save_to_h5(file_name, part_name, partition, sequence, indices):
    h5file = tables.open_file(file_name, mode='w')
    # seq_group = h5file.create_group(h5file.root, "phrases", "Code Token Sequence")
    # ind_group = h5file.create_group(h5file.root, "indices", "Sample indices")
    sequence_earray = h5file.create_earray(h5file.root, "phrases", obj=np.asarray(sequence))
    indices_table = h5file.create_table(h5file.root, "indices", Index)
    particle = indices_table.row
    for index in indices:
        particle['length'] = index[0]
        particle['pos'] = index[1]
        particle.append()
    h5file.flush()
    h5file.close()


def main():
    for part_name in ["name", "apiseq", "tokens", "desc"]:
        vocab = get_vocab("train.{}.txt".format(part_name), part_name)
        for partition in ["train", "valid", "test"]:
            print ("{}-{}".format(part_name, partition))
            file_name = "{}.{}.txt".format(partition, part_name)
            h5_file_name = "{}.{}.h5".format(partition, part_name)
            sequence, indices = get_sequence_and_indices(file_name, part_name, partition, vocab)
            save_to_h5(h5_file_name, part_name, partition, sequence, indices)


if __name__ == "__main__":
    main()
