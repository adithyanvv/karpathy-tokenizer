from pathlib import Path

from minbpe import RegexTokenizer


# This file is inside: mini_gpt/data/
DATA_DIR = Path(__file__).parent

TRAIN_FILE = DATA_DIR / "train.txt"
MODEL_FILE = DATA_DIR / "shakespeare_tokenizer"


def main():
    # Load the training text
    text = TRAIN_FILE.read_text(encoding="utf-8")

    print("Training tokenizer...")
    print("Training characters:", len(text))

    # Create our regex-based BPE tokenizer
    tokenizer = RegexTokenizer()

    # 300 total tokens:
    # 256 original byte tokens + 44 learned tokens
    vocab_size = 300

    # Train the tokenizer on our Shakespeare text
    tokenizer.train(text, vocab_size)

    # Save the trained tokenizer
    tokenizer.save(str(MODEL_FILE))

    print("Tokenizer training complete.")
    print("Vocabulary size:", len(tokenizer.vocab))
    print("Model saved to:", MODEL_FILE)


if __name__ == "__main__":
    main()