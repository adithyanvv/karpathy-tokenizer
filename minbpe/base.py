from collections import Counter


def get_stats(ids, counts=None):
    """
    Count how often each consecutive pair appears.

    Example:
    [1, 2, 2, 3]

    pairs:
    (1, 2)
    (2, 2)
    (2, 3)
    """
    counts = {} if counts is None else counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(ids, pair, new_id):
    """
    Replace every occurrence of a pair with a new token ID.

    Example:
    ids = [1, 2, 2, 3]
    pair = (2, 2)
    new_id = 256

    result:
    [1, 256, 3]
    """
    new_ids = []
    i = 0

    while i < len(ids):
        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


class Tokenizer:
    """
    Base class for our tokenizers.

    A tokenizer has:
    - merges: which pairs were merged
    - vocab: what bytes each token represents
    """

    def __init__(self):
        self.merges = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """
        Start with the 256 possible byte values.

        Token IDs 0-255 represent individual bytes.

        Then add our learned tokens starting at 256.
        """
        vocab = {
            idx: bytes([idx])
            for idx in range(256)
        }

        for (idx1, idx2), new_id in self.merges.items():
            vocab[new_id] = vocab[idx1] + vocab[idx2]

        return vocab

    def train(self, text, vocab_size):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        """
        Convert token IDs back into text.
        """
        text_bytes = b"".join(
            self.vocab[idx]
            for idx in ids
        )

        return text_bytes.decode(
            "utf-8",
            errors="replace"
        )

    def save(self, file_prefix):
        """
        Save the tokenizer's learned merges.

        Creates:
            file_prefix.model
        """
        model_file = file_prefix + ".model"

        with open(model_file, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(str(256 + len(self.merges)) + "\n")

            for (idx1, idx2), new_id in self.merges.items():
                f.write(f"{idx1} {idx2}\n")

        return model_file

    def load(self, model_file):
        """
        Load a tokenizer model from disk.
        """
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()

            if version != "minbpe v1":
                raise ValueError(
                    "Unsupported tokenizer model version"
                )

            vocab_size = int(f.readline().strip())

            self.merges = {}

            for i in range(vocab_size - 256):
                idx1, idx2 = map(
                    int,
                    f.readline().split()
                )

                self.merges[(idx1, idx2)] = 256 + i

        self.vocab = self._build_vocab()