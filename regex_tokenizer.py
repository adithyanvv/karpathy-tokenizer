import regex as re
from basic_tokenizer import BasicTokenizer


class RegexTokenizer(BasicTokenizer):

    PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def split_text(self, text):
        return re.findall(self.PATTERN, text)

    def train(self, text, num_merges):
        chunks = self.split_text(text)

        for chunk in chunks:
            ids = list(chunk.encode("utf-8"))

            for _ in range(num_merges):
                stats = self.get_stats(ids)

                if not stats:
                    break

                pair = max(stats, key=stats.get)

                if pair not in self.merges:
                    new_id = 256 + len(self.merges)

                    self.merges[pair] = new_id

                    self.vocab[new_id] = (
                        self.vocab[pair[0]] + self.vocab[pair[1]]
                    )

                ids = self.merge(ids, pair, self.merges[pair])

    def encode(self, text):
        chunks = self.split_text(text)

        ids = []

        for chunk in chunks:
            chunk_ids = list(chunk.encode("utf-8"))

            while len(chunk_ids) >= 2:
                stats = self.get_stats(chunk_ids)

                if not stats:
                    break

                pair = min(
                    stats,
                    key=lambda p: self.merges.get(p, float("inf"))
                )

                if pair not in self.merges:
                    break

                chunk_ids = self.merge(
                    chunk_ids,
                    pair,
                    self.merges[pair]
                )

            ids.extend(chunk_ids)

        return ids


# -------------------------
# TEST
# -------------------------

tokenizer = RegexTokenizer()

text = "Hello, my name is John!"

print("Original:", text)

print("\nChunks:")
print(tokenizer.split_text(text))

print("\nTraining...")
tokenizer.train(text, 10)

print("\nEncoding...")
tokens = tokenizer.encode(text)

print("Tokens:", tokens)

print("\nDecoding...")
decoded = tokenizer.decode(tokens)

print("Decoded:", decoded)

print("\nMatch:", decoded == text)