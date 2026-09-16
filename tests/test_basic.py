from minbpe import BasicTokenizer


def test_basic_tokenizer_roundtrip():
    text = "aaabdaaabac"

    tokenizer = BasicTokenizer()
    tokenizer.train(text, 259)

    tokens = tokenizer.encode(text)

    # The tokenizer should learn 3 BPE merges.
    assert tokens == [258, 100, 258, 97, 99]

    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_basic_tokenizer_unicode():
    text = "Hello 👋 世界"

    tokenizer = BasicTokenizer()
    tokenizer.train(text, 300)

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_save_and_load(tmp_path):
    text = "aaabdaaabac"

    tokenizer = BasicTokenizer()
    tokenizer.train(text, 259)

    model_file = tmp_path / "test_model"

    tokenizer.save(str(model_file))

    loaded_tokenizer = BasicTokenizer()
    loaded_tokenizer.load(str(model_file) + ".model")

    original_tokens = tokenizer.encode(text)
    loaded_tokens = loaded_tokenizer.encode(text)

    assert loaded_tokens == original_tokens
    assert loaded_tokenizer.decode(loaded_tokens) == text