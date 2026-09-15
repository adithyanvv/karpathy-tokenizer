from collections import Counter


class BasicTokenizer:

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def get_stats(self, ids):
        counts = Counter()

        for pair in zip(ids, ids[1:]):
            counts[pair] += 1

        return counts

    def merge(self, ids, pair, new_id):
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train(self, text, num_merges):
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            stats = self.get_stats(ids)

            if not stats:
                break

            pair = max(stats, key=stats.get)

            new_id = 256 + i

            ids = self.merge(ids, pair, new_id)

            self.merges[pair] = new_id

            self.vocab[new_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

            print(f"Merge {i + 1}: {pair} -> {new_id}")

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self.get_stats(ids)

            pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf"))
            )

            if pair not in self.merges:
                break

            ids = self.merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8")


# -------------------------
# TEST
# -------------------------

tokenizer = BasicTokenizer()

text = "hello hello hello"

print("\nTraining tokenizer...\n")

tokenizer.train(text, 5)

print("\nEncoding...")

tokens = tokenizer.encode(text)

print("Tokens:", tokens)

print("\nDecoding...")

decoded = tokenizer.decode(tokens)

print("Decoded:", decoded)

print("\nOriginal:", text)

print("\nMatch:", decoded == text)