from minbpe import RegexTokenizer


def test_special_tokens_survive_save_and_load(tmp_path):
    tokenizer = RegexTokenizer()

    tokenizer.train("hello world", 300)

    tokenizer.register_special_tokens({
        "<|endoftext|>": 300
    })

    model_file = tmp_path / "tokenizer"

    tokenizer.save(str(model_file))

    loaded = RegexTokenizer()
    loaded.load(str(model_file) + ".model")

    loaded.register_special_tokens({
        "<|endoftext|>": 300
    })

    encoded = loaded.encode(
        "<|endoftext|>hello",
        allowed_special="all"
    )

    assert encoded[0] == 300

    decoded = loaded.decode(encoded)

    assert decoded == "<|endoftext|>hello"