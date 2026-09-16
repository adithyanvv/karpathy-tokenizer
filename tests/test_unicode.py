from minbpe import RegexTokenizer


def test_unicode_round_trip():
    tokenizer = RegexTokenizer()

    text = "Hello café 世界 🌍"

    tokenizer.train(text, 300)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text