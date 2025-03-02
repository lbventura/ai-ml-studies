import torch
import pandas as pd

from rnn_trans_arch.data_extraction import load_data
from rnn_trans_arch.training_utils import (
    translate_sentence,
)
from pathlib import Path

parent_dir = Path(__file__).parent

data_source_path = parent_dir / "data/pig_latin_data.txt"
model_source_path = parent_dir / "output/pig_latin_data/h20-bs64-rnn-additive-lr-0.005"

# Load the encoder
rnn_encoder = torch.load(
    f"{model_source_path}/encoder.pt",
    map_location=torch.device("cpu"),
    weights_only=False,
)

# Load the decoder
rnn_decoder = torch.load(
    f"{model_source_path}/decoder.pt",
    map_location=torch.device("cpu"),
    weights_only=False,
)

dictionary, _, idx_data = load_data(data_source=str(data_source_path))

gen_dict = {}


for english_word, pig_latin_word in dictionary[:2000]:
    predicted_pig_latin_word = translate_sentence(
        english_word, rnn_encoder, rnn_decoder, idx_data, torch.device("cpu")
    )
    gen_dict[english_word] = {
        "actual": pig_latin_word,
        "predicted": predicted_pig_latin_word,
    }

# Create a DataFrame from the generated dictionary
translation_data = pd.DataFrame(gen_dict).T

# Split the data into small and large words
small_words = [True if len(word) <= 4 else False for word in translation_data.index]
large_words = [not word for word in small_words]
small_df = translation_data[small_words].copy()
large_df = translation_data[large_words].copy()

correct_fraction_small_words = sum(small_df["actual"] == small_df["predicted"]) / len(
    small_df["actual"]
)
print(
    f"Fraction of correct translations for small words: {correct_fraction_small_words%1:.2f}"
)

correct_fraction_large_words = sum(large_df["actual"] == large_df["predicted"]) / len(
    large_df["actual"]
)
print(
    f"Fraction of correct translations for small words: {correct_fraction_large_words%1:.2f}"
)


# Compute the similarity between the actual and predicted words
def count_identical_chars(str1: str, str2: str) -> int:
    return sum(c1 == c2 for c1, c2 in zip(str1, str2))


small_df["string_comparison"] = small_df.apply(
    lambda row: count_identical_chars(row["actual"], row["predicted"])
    / len(row["actual"]),
    axis=1,
)
large_df["string_comparison"] = large_df.apply(
    lambda row: count_identical_chars(row["actual"], row["predicted"])
    / len(row["actual"]),
    axis=1,
)

print(
    f"The string comparison pct (similarity score) for small words was {small_df['string_comparison'].mean()%1:.2f}"
)
print(
    f"The string comparison pct (similarity score) for big words was {large_df['string_comparison'].mean()%1:.2f}"
)


def starts_with_vowel(s: str) -> bool:
    return s[0] in "aeiou"


vowel_words = [starts_with_vowel(word) for word in translation_data.index]
consonant_words = [not starts_with_vowel(word) for word in translation_data.index]

vowel_df = translation_data[vowel_words]
consonant_df = translation_data[consonant_words]

correct_fraction_vowel_words = sum(vowel_df["actual"] == vowel_df["predicted"]) / len(
    vowel_df["actual"]
)
print(
    f"Fraction of correct translations for vowel words: {correct_fraction_vowel_words%1:.2f}"
)

correct_fraction_consonant_words = sum(
    consonant_df["actual"] == consonant_df["predicted"]
) / len(consonant_df["actual"])
print(
    f"Fraction of correct translations for consonant words: {correct_fraction_consonant_words%1:.2f}"
)
