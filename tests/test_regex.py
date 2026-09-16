from minbpe import RegexTokenizer


def test_regex_tokenizer_roundtrip():
    text = "Hello, my name is John! 123"

    tokenizer = RegexTokenizer()
    tokenizer.train(text, 300)

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_regex_tokenizer_unicode():
    text = "Hello 👋 世界"

    tokenizer = RegexTokenizer()
    tokenizer.train(text, 300)

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_regex_tokenizer_special_token():
    text = "Hello<|endoftext|>world"

    tokenizer = RegexTokenizer()
    tokenizer.train("Hello world", 300)

    tokenizer.register_special_tokens({
        "<|endoftext|>": 300
    })

    tokens = tokenizer.encode(
        text,
        allowed_special="all"
    )

    assert 300 in tokens

    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_regex_tokenizer_rejects_special_token_without_permission():
    tokenizer = RegexTokenizer()
    tokenizer.train("Hello world", 300)

    tokenizer.register_special_tokens({
        "<|endoftext|>": 300
    })

    try:
        tokenizer.encode(
            "Hello<|endoftext|>world",
            allowed_special="none_raise"
        )
        assert False
    except ValueError:
        pass