from minbpe import RegexTokenizer


def test_special_tokens():
    tokenizer = RegexTokenizer()

    text = "hello world"

    # Train a small tokenizer.
    tokenizer.train(text, 300)

    # Add a special token after the normal vocabulary.
    tokenizer.register_special_tokens({
        "<|endoftext|>": 300
    })

    # Encode text containing the special token.
    encoded = tokenizer.encode(
        "<|endoftext|>hello",
        allowed_special="all"
    )

    # The first token must be our special token.
    assert encoded[0] == 300

    # Decode it again.
    decoded = tokenizer.decode(encoded)

    assert decoded == "<|endoftext|>hello"


def test_special_token_requires_permission():
    tokenizer = RegexTokenizer()

    tokenizer.train("hello world", 300)

    tokenizer.register_special_tokens({
        "<|endoftext|>": 300
    })

    # Without allowing the special token, encoding should fail.
    try:
        tokenizer.encode("<|endoftext|>hello")
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(
            "Special token should not be accepted unless explicitly allowed."
        )