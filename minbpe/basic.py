from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def train(self, text, vocab_size):
        """
        Train a byte-level BPE tokenizer.

        vocab_size:
            Total number of tokens we want.

        The first 256 tokens are the raw bytes.
        Everything after 256 is learned through BPE merges.
        """

        if vocab_size < 256:
            raise ValueError(
                "vocab_size must be at least 256"
            )

        # Convert the entire training text into UTF-8 bytes.
        ids = list(text.encode("utf-8"))

        # Start with no learned merges.
        self.merges = {}

        # Number of new tokens we need to learn.
        num_merges = vocab_size - 256

        for i in range(num_merges):

            # Count all consecutive byte/token pairs.
            stats = get_stats(ids)

            if not stats:
                break

            # Find the most common pair.
            pair = max(
                stats,
                key=stats.get
            )

            # New token IDs start at 256.
            new_id = 256 + i

            # Remember this merge.
            self.merges[pair] = new_id

            # Replace the pair everywhere in the training data.
            ids = merge(
                ids,
                pair,
                new_id
            )

        # Build the final vocabulary.
        self.vocab = self._build_vocab()

    def encode(self, text):
        """
        Convert text into token IDs.
        """

        # Convert text into bytes.
        ids = list(text.encode("utf-8"))

        # Keep applying learned merges.
        while len(ids) >= 2:

            stats = get_stats(ids)

            if not stats:
                break

            # Among the pairs that currently exist,
            # choose the one learned earliest.
            pair = min(
                stats,
                key=lambda p: self.merges.get(
                    p,
                    float("inf")
                )
            )

            # If this pair was never learned,
            # we cannot merge it.
            if pair not in self.merges:
                break

            ids = merge(
                ids,
                pair,
                self.merges[pair]
            )

        return ids