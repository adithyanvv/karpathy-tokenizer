from minbpe import RegexTokenizer


text = "Hello world"


tokenizer = RegexTokenizer()

print("Training tokenizer...")
tokenizer.train(text, 300)


print("\nRegistering special token...")

tokenizer.register_special_tokens({
    "<|endoftext|>": 300
})


print("\nEncoding normal text...")

tokens = tokenizer.encode(text)
print("Tokens:", tokens)


print("\nEncoding text with special token...")

text_with_special = "Hello<|endoftext|>world"

tokens = tokenizer.encode(
    text_with_special,
    allowed_special="all"
)

print("Tokens:", tokens)


print("\nDecoding...")

decoded = tokenizer.decode(tokens)

print("Decoded:", decoded)
print("Original:", text_with_special)
print("Match:", decoded == text_with_special)