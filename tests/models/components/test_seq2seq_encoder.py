import pytest
import torch

from pie_models.models.components.seq2seq_encoder import (
    ACTIVATION_TYPE2CLASS,
    RNN_TYPE2CLASS,
    build_seq2seq_encoder,
)


def test_no_encoder():
    seq2seq_dict = {}
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is None

    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": "none",
        },
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert len(encoder) == 0


@pytest.mark.parametrize("seq2seq_enc_type", ["lstm", "gru", "rnn"])
def test_rnn_encoder(seq2seq_enc_type):
    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": seq2seq_enc_type,
            "hidden_size": 10,
        },
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert (
        encoder[0].rnn.mode
        == RNN_TYPE2CLASS[seq2seq_enc_type](
            input_size=10,
            hidden_size=10,
            batch_first=True,
        ).mode
    )


@pytest.mark.parametrize("bidirectional", [True, False])
def test_output_size(bidirectional):
    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": "lstm",
            "bidirectional": bidirectional,
            "hidden_size": 10,
        },
    }

    input_size = 10
    assert_output_size = input_size * 2 if bidirectional else input_size
    _, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert output_size == assert_output_size


@pytest.mark.parametrize("activation_type", ["relu", "tanh", "sigmoid", "gelu"])
def test_activations(activation_type):
    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": activation_type,
        },
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert encoder[0]._get_name() == ACTIVATION_TYPE2CLASS[activation_type]()._get_name()


def test_dropout():
    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": "dropout",
            "p": 0.5,
        },
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert encoder[0]._get_name() == torch.nn.Dropout(p=0.5)._get_name()


def test_linear():
    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": "linear",
            "out_features": 10,
        },
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert encoder[0]._get_name() == torch.nn.Linear(10, 10)._get_name()


def test_unknown_rnn_type():
    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": "unknown",
        },
    }
    with pytest.raises(ValueError):
        build_seq2seq_encoder(seq2seq_dict, 10)
