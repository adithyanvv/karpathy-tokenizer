import regex as re

from .basic import BasicTokenizer
from .base import get_stats, merge


class RegexTokenizer(BasicTokenizer):

    # This is the regex pattern used to split text
    # into meaningful chunks before BPE is applied.
    PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)"""
        r"""|[^\r\n\p{L}\p{N}]?+\p{L}+"""
        r"""|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*"""
        r"""|\s*[\r\n]"""
        r"""|\s+(?!\S)"""
        r"""|\s+"""
    )

    def __init__(self):
        super().__init__()

        self.compiled_pattern = re.compile(
            self.PATTERN
        )

        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size):
        """
        Train BPE over regex-split text chunks.
        """

        if vocab_size < 256:
            raise ValueError(
                "vocab_size must be at least 256"
            )

        # Split the training text into chunks.
        text_chunks = re.findall(
            self.compiled_pattern,
            text
        )

        # Convert every chunk into UTF-8 bytes.
        ids = [
            list(chunk.encode("utf-8"))
            for chunk in text_chunks
        ]

        self.merges = {}

        num_merges = vocab_size - 256

        for i in range(num_merges):

            # Count pairs across ALL chunks.
            stats = {}

            for chunk_ids in ids:
                get_stats(
                    chunk_ids,
                    stats
                )

            if not stats:
                break

            # Most common pair across the whole corpus.
            pair = max(
                stats,
                key=stats.get
            )

            new_id = 256 + i

            self.merges[pair] = new_id

            # Apply the same merge to every chunk.
            ids = [
                merge(
                    chunk_ids,
                    pair,
                    new_id
                )
                for chunk_ids in ids
            ]

        self.vocab = self._build_vocab()

    def encode_ordinary(self, text):
        """
        Encode normal text without special tokens.
        """

        text_chunks = re.findall(
            self.compiled_pattern,
            text
        )

        ids = []

        for chunk in text_chunks:

            chunk_ids = list(
                chunk.encode("utf-8")
            )

            while len(chunk_ids) >= 2:

                stats = get_stats(
                    chunk_ids
                )

                if not stats:
                    break

                pair = min(
                    stats,
                    key=lambda p: self.merges.get(
                        p,
                        float("inf")
                    )
                )

                if pair not in self.merges:
                    break

                chunk_ids = merge(
                    chunk_ids,
                    pair,
                    self.merges[pair]
                )

            ids.extend(chunk_ids)

        return ids

    def register_special_tokens(self, special_tokens):
        """
        Register tokens such as <|endoftext|>.
        """

        self.special_tokens = dict(
            special_tokens
        )

        self.inverse_special_tokens = {
            token_id: token
            for token, token_id
            in self.special_tokens.items()
        }

    def encode(self, text, allowed_special="none_raise"):
        """
        Encode text, optionally allowing special tokens.
        """

        if allowed_special == "all":
            allowed_special = set(
                self.special_tokens
            )

        elif allowed_special == "none":
            allowed_special = set()

        elif allowed_special == "none_raise":

            for token in self.special_tokens:
                if token in text:
                    raise ValueError(
                        f"Special token {token} "
                        "found in text. "
                        "Use allowed_special='all' "
                        "to allow it."
                    )

            allowed_special = set()

        else:
            allowed_special = set(
                allowed_special
            )

        if not allowed_special:
            return self.encode_ordinary(text)

        # Create a regex for the allowed special tokens.
        special_pattern = (
            "("
            + "|".join(
                re.escape(token)
                for token in allowed_special
            )
            + ")"
        )

        parts = re.split(
            special_pattern,
            text
        )

        ids = []

        for part in parts:

            if part in allowed_special:
                ids.append(
                    self.special_tokens[part]
                )
            elif part:
                ids.extend(
                    self.encode_ordinary(part)
                )

        return ids

    def decode(self, ids):
        """
        Convert token IDs back into text,
        including special tokens.
        """

        parts = []
        byte_buffer = bytearray()

        for token_id in ids:

            if token_id in self.inverse_special_tokens:

                if byte_buffer:
                    parts.append(
                        bytes(byte_buffer).decode(
                            "utf-8",
                            errors="replace"
                        )
                    )
                    byte_buffer.clear()

                parts.append(
                    self.inverse_special_tokens[
                        token_id
                    ]
                )

            else:
                byte_buffer.extend(
                    self.vocab[token_id]
                )

        if byte_buffer:
            parts.append(
                bytes(byte_buffer).decode(
                    "utf-8",
                    errors="replace"
                )
            )

        return "".join(parts)
    