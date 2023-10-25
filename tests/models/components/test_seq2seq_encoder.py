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


@pytest.mark.parametrize("seq2seq_enc_type", list(RNN_TYPE2CLASS))
@pytest.mark.parametrize("bidirectional", [True, False])
def test_rnn_encoder(seq2seq_enc_type, bidirectional):
    seq2seq_dict = {
        "type": seq2seq_enc_type,
        "hidden_size": 10,
        "bidirectional": bidirectional,
    }
    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert encoder is not None
    assert isinstance(encoder.rnn, type(RNN_TYPE2CLASS[seq2seq_enc_type](10, 10)))

    expected_output_size = input_size * 2 if bidirectional else input_size
    assert output_size is not None
    assert output_size == expected_output_size


@pytest.mark.parametrize("activation_type", list(ACTIVATION_TYPE2CLASS))
def test_activations(activation_type):
    seq2seq_dict = {
        "type": activation_type,
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert isinstance(encoder, type(ACTIVATION_TYPE2CLASS[activation_type]()))


def test_dropout():
    seq2seq_dict = {
        "type": "dropout",
        "p": 0.5,
    }
    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert isinstance(encoder, type(torch.nn.Dropout()))


def test_linear():
    seq2seq_dict = {
        "type": "linear",
        "out_features": 10,
    }

    encoder, _ = build_seq2seq_encoder(seq2seq_dict, 10)
    assert encoder is not None
    assert isinstance(encoder, type(torch.nn.Linear(10, 10)))


def test_unknown_rnn_type():
    seq2seq_dict = {
        "type": "unknown",
    }
    with pytest.raises(ValueError) as e:
        build_seq2seq_encoder(seq2seq_dict, 10)
        assert e == "Unknown seq2seq_encoder_type: unknown"
