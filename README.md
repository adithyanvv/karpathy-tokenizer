# Karpathy Tokenizer

A from-scratch implementation of a byte-level Byte Pair Encoding (BPE) tokenizer, inspired by Andrej Karpathy's tokenizer lectures and minbpe project.

## Features

- Byte-level tokenization
- BPE pair-frequency training
- Token encoding
- Token decoding
- Regex-based text splitting
- Unicode text support
- Special token support
- Tokenizer model save/load
- Automated tests with pytest
- Git version control

## Project Structure

```text
karpathy-tokenizer/
├── minbpe/
│   ├── __init__.py
│   ├── base.py
│   ├── basic.py
│   └── regex.py
├── tests/
│   ├── test_basic.py
│   └── test_regex.py
├── basic_tokenizer.py
├── regex_tokenizer.py
├── tokenizer.py
├── requirements.txt
├── .gitignore
└── README.md