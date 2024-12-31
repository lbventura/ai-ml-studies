import os
import time
from typing import cast
import torch
from rnn_trans_arch.data_types import (
    DecoderType,
    ModelParams,
    TrainingParams,
)
from rnn_trans_arch.data_extraction import (
    get_file,
    load_data,
    create_dict,
)

import torch.nn as nn
import torch.optim as optim
from rnn_trans_arch.gru_encoder import GRUEncoder
from rnn_trans_arch.rnn_decoder import RNNDecoder
from rnn_trans_arch.attention_decoder import RNNAttentionDecoder
from rnn_trans_arch.transformer_decoder import TransformerDecoder
from rnn_trans_arch.training_utils import (
    print_data_stats,
    set_checkpoint_path,
    training_loop,
    translate_sentence,
)
from pathlib import Path

SEED = 1

# Set the random seed manually for reproducibility.
torch.manual_seed(SEED)

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS device for acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("Using CUDA device for acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device.")

TEST_SENTENCE = "the air conditioning is working"


def train(
    training_params: TrainingParams,
    model_params: ModelParams,
) -> tuple[
    nn.Module,
    nn.Module,
    dict[tuple[int, int], list[tuple[str, str]]],
    dict[tuple[int, int], list[tuple[str, str]]],
    dict[str, dict[str, int] | dict[int, str] | int],
]:
    # Load the data and print some stats/examples
    line_pairs, vocab_size, idx_dict = load_data(
        data_source=training_params.data_source
    )
    char_to_index = cast(dict[str, int], idx_dict["char_to_index"])
    print_data_stats(
        line_pairs=line_pairs, vocab_size=vocab_size, char_to_index=char_to_index
    )

    # Split the line pairs into an 80% train and 20% val split
    training_data_percentage = 0.8
    num_lines = len(line_pairs)
    num_train = int(training_data_percentage * num_lines)
    train_pairs, val_pairs = line_pairs[:num_train], line_pairs[num_train:]

    # Group the data by the lengths of the source and target words, to form batches
    train_dict = create_dict(train_pairs)
    val_dict = create_dict(val_pairs)

    # Model setup
    encoder = GRUEncoder(
        vocab_size=vocab_size,
        hidden_size=model_params.hidden_size,
        device=training_params.device,
    )

    if model_params.decoder_type == DecoderType.rnn:
        decoder = RNNDecoder(
            vocab_size=vocab_size, hidden_size=model_params.hidden_size
        )
    elif model_params.decoder_type == DecoderType.rnn_attention:
        decoder = RNNAttentionDecoder(
            vocab_size=vocab_size,
            hidden_size=model_params.hidden_size,
            attention_type=model_params.attention_type,
        )
    elif model_params.decoder_type == DecoderType.transformer:
        assert model_params.num_transformer_layers is not None  # make mypy happy
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=model_params.hidden_size,
            num_layers=model_params.num_transformer_layers,
        )
    else:
        raise NotImplementedError

    # Define checkpoint path
    current_path = Path(__file__).parent
    set_checkpoint_path(
        current_path=current_path,
        training_params=training_params,
        model_params=model_params,
    )

    if not os.path.exists(training_params.checkpoint_dir):
        os.makedirs(training_params.checkpoint_dir)

    if training_params.device.type == "cuda" or training_params.device.type == "mps":
        encoder.to(training_params.device)
        decoder.to(training_params.device)
        print("Moved models to GPU!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=training_params.learning_rate,
    )

    # Train the model
    try:
        training_loop(
            train_dict=train_dict,
            val_dict=val_dict,
            idx_dict=idx_dict,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            optimizer=optimizer,
            training_params=training_params,
            model_params=model_params,
            test_sentence=TEST_SENTENCE,
        )
    except KeyboardInterrupt:
        print("Exiting early from training.")
        return encoder, decoder  # type: ignore

    return encoder, decoder, train_dict, val_dict, idx_dict


if __name__ == "__main__":
    data_source = "pig_latin_data.txt"
    training_params = TrainingParams(
        data_source=data_source,
    )
    model_params = ModelParams()

    data_fpath = get_file(
        fname=data_source,
        origin="http://www.cs.toronto.edu/~jba/pig_latin_data.txt",
        untar=False,
    )
    start_time = time.time()
    (
        rnn_attn_encoder_scaled_dot,
        rnn_attn_decoder_scaled_dot,
        train_dict,
        test_dict,
        idx_dict,
    ) = train(training_params=training_params, model_params=model_params)
    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

    translated = translate_sentence(
        TEST_SENTENCE,
        rnn_attn_encoder_scaled_dot,
        rnn_attn_decoder_scaled_dot,
        idx_dict,
        device=training_params.device,
    )
    print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))
