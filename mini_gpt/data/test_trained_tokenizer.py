from pathlib import Path

from minbpe import RegexTokenizer


# This file is already inside mini_gpt/data/
DATA_DIR = Path(__file__).parent
MODEL_FILE = DATA_DIR / "shakespeare_tokenizer.model"


def main():
    print("Loading trained tokenizer...")

    tokenizer = RegexTokenizer()
    tokenizer.load(str(MODEL_FILE))

    print("Vocabulary size:", len(tokenizer.vocab))

    text = "To be, or not to be."

    print("\nOriginal text:")
    print(text)

    tokens = tokenizer.encode(text)

    print("\nEncoded tokens:")
    print(tokens)

    decoded = tokenizer.decode(tokens)

    print("\nDecoded text:")
    print(decoded)

    print("\nRound-trip successful:", decoded == text)


if __name__ == "__main__":
    main()