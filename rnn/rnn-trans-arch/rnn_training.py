import os
from typing import cast
import torch
from rnn_trans_arch.data_types import DecoderType, ModelParams, TrainingParams
from rnn_trans_arch.data_extraction import (
    get_file,
    load_data,
    create_dict,
)

import torch.nn as nn
import torch.optim as optim
from rnn_trans_arch.gru import GRUEncoder
from rnn_trans_arch.rnn_decoder import RNNDecoder
from rnn_trans_arch.attention_decoder import RNNAttentionDecoder
from rnn_trans_arch.transformer_decoder import TransformerDecoder
from rnn_trans_arch.training_utils import (
    print_data_stats,
    training_loop,
    translate_sentence,
)

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
    line_pairs, vocab_size, idx_dict = load_data()
    char_to_index = cast(dict[str, int], idx_dict["char_to_index"])
    print_data_stats(
        line_pairs=line_pairs, vocab_size=vocab_size, char_to_index=char_to_index
    )

    # Split the line pairs into an 80% train and 20% val split
    num_lines = len(line_pairs)
    num_train = int(0.8 * num_lines)
    train_pairs, val_pairs = line_pairs[:num_train], line_pairs[num_train:]

    # Group the data by the lengths of the source and target words, to form batches
    train_dict = create_dict(train_pairs)
    val_dict = create_dict(val_pairs)

    # Model setup
    encoder = GRUEncoder(
        vocab_size=vocab_size,
        hidden_size=model_params.hidden_size,
        cuda=training_params.cuda,
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
    model_name = "h{}-bs{}-{}".format(
        model_params.hidden_size, training_params.batch_size, model_params.decoder_type
    )
    training_params.checkpoint_dir = model_name
    if not os.path.exists(training_params.checkpoint_dir):
        os.makedirs(training_params.checkpoint_dir)

    if training_params.cuda:
        encoder.cuda()
        decoder.cuda()
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
    torch.manual_seed(1)
    training_params = TrainingParams()
    model_params = ModelParams()

    data_fpath = get_file(
        fname="pig_latin_data.txt",
        origin="http://www.cs.toronto.edu/~jba/pig_latin_data.txt",
        untar=False,
    )

    (
        rnn_attn_encoder_scaled_dot,
        rnn_attn_decoder_scaled_dot,
        train_dict,
        test_dict,
        idx_dict,
    ) = train(training_params=training_params, model_params=model_params)

    translated = translate_sentence(
        TEST_SENTENCE,
        rnn_attn_encoder_scaled_dot,
        rnn_attn_decoder_scaled_dot,
        idx_dict,
        cuda=training_params.cuda,
    )
    print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))
