from pathlib import Path


# This file is inside: mini_gpt/data/
DATA_DIR = Path(__file__).parent

INPUT_FILE = DATA_DIR / "shakespeare.txt"
TRAIN_FILE = DATA_DIR / "train.txt"
VAL_FILE = DATA_DIR / "val.txt"


def prepare_data():
    # Read the original dataset
    text = INPUT_FILE.read_text(encoding="utf-8")

    # Remove unnecessary empty space at the beginning/end
    text = text.strip()

    # Use 90% for training and 10% for validation
    split_index = int(len(text) * 0.9)

    train_text = text[:split_index]
    val_text = text[split_index:]

    # Save the two datasets
    TRAIN_FILE.write_text(train_text, encoding="utf-8")
    VAL_FILE.write_text(val_text, encoding="utf-8")

    print("Dataset prepared.")
    print("Total characters:", len(text))
    print("Training characters:", len(train_text))
    print("Validation characters:", len(val_text))


if __name__ == "__main__":
    prepare_data()