import json

import pytest
import torch
import transformers
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from pie_models.models.token_classification_with_seq2seq_encoder_and_crf import (
    HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE,
    TokenClassificationModelWithSeq2SeqEncoderAndCrf,
)
from tests import FIXTURES_ROOT

DUMP_FIXTURES = False
FIXTURES_TASKMODULE_DATA_PATH = (
    FIXTURES_ROOT / "taskmodules" / "token_classification_with_seq2seq_encoder_and_crf"
)


@pytest.fixture(scope="module")
def documents(dataset):
    return dataset["train"]


@pytest.mark.skipif(
    condition=not DUMP_FIXTURES, reason="Only need to dump the data if taskmodule has changed"
)
def test_dump_fixtures(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerTokenClassificationTaskModule(
        entity_annotation="entities",
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    taskmodule.prepare(documents)
    encodings = taskmodule.encode(documents, encode_target=True, as_dataset=True)
    batch_encoding = taskmodule.collate(encodings[:4])

    FIXTURES_TASKMODULE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = FIXTURES_TASKMODULE_DATA_PATH / "batch_encoding_inputs.json"

    inputs = {key: tensor.tolist() for key, tensor in batch_encoding[0].items()}
    targets = batch_encoding[1].tolist()
    converted_batch_encoding = {
        "inputs": inputs,
        "targets": targets,
    }
    with open(filepath, "w") as f:
        json.dump(converted_batch_encoding, f)
    return converted_batch_encoding


@pytest.fixture
def batch(documents):
    filepath = FIXTURES_TASKMODULE_DATA_PATH / "batch_encoding_inputs.json"
    with open(filepath) as f:
        batch_encoding = json.load(f)
    inputs = {
        "input_ids": torch.LongTensor(batch_encoding["inputs"]["input_ids"]),
        "token_type_ids": torch.LongTensor(batch_encoding["inputs"]["token_type_ids"]),
        "attention_mask": torch.LongTensor(batch_encoding["inputs"]["attention_mask"]),
    }
    targets = torch.LongTensor(batch_encoding["targets"])
    return (inputs, targets)


def get_model(
    monkeypatch,
    model_type,
    num_classes,
    batch_size,
    seq_len,
    add_dummy_linear=False,
    **model_kwargs,
):
    class MockConfig:
        def __init__(
            self,
            hidden_size: int = 10,
            classifier_dropout: float = 0.1,
            model_type=model_type,
        ) -> None:
            self.hidden_size = hidden_size
            self.model_type = model_type
            classifier_dropout_attr = HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE.get(
                model_type, "classifier_dropout"
            )
            setattr(self, classifier_dropout_attr, classifier_dropout)

    class MockModel(torch.nn.Module):
        def __init__(self, batch_size, seq_len, hidden_size, add_dummy_linear) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)

        def __call__(self, *args, **kwargs):
            last_hidden_state = torch.FloatTensor(
                torch.rand(self.batch_size, self.seq_len, self.hidden_size)
            )
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=last_hidden_state,
            )

    hidden_size = 10

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_name_or_path: MockConfig(
            hidden_size=hidden_size, classifier_dropout=0.1, model_type=model_type
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name_or_path, config: MockModel(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            add_dummy_linear=add_dummy_linear,
        ),
    )

    # set seed to make the classifier deterministic
    torch.manual_seed(42)
    result = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="bert",
        num_classes=num_classes,
        warmup_proportion=0.0,
        **model_kwargs,
    )
    assert not result.is_from_pretrained

    return result


@pytest.fixture
def model(monkeypatch, batch):
    inputs, targets = batch
    model = get_model(
        monkeypatch=monkeypatch,
        model_type="bert",
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
        num_classes=int(torch.max(targets) + 1),
    )
    return model


@pytest.mark.parametrize("model_type", ["bert", "albert", "distilbert", "longformer"])
def test_config_model_classifier_dropout(monkeypatch, model_type):
    model = get_model(
        monkeypatch=monkeypatch,
        model_type=model_type,
        batch_size=4,
        seq_len=10,
        num_classes=5,
    )

    assert model is not None
    assert isinstance(model, TokenClassificationModelWithSeq2SeqEncoderAndCrf)


def test_freeze_base_model(monkeypatch, batch):
    inputs, target = batch
    # set seed to make the classifier deterministic
    model = get_model(
        monkeypatch,
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
        add_dummy_linear=True,
        freeze_base_model=True,
    )
    base_model_params = list(model.model.parameters())
    # the dummy linear from the mock base model has 2 parameters
    assert len(base_model_params) == 2
    for param in base_model_params:
        assert not param.requires_grad


def test_tune_base_model(monkeypatch, batch):
    inputs, target = batch
    # set seed to make the classifier deterministic
    model = get_model(
        monkeypatch,
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
        add_dummy_linear=True,
        freeze_base_model=False,
    )
    base_model_params = list(model.model.parameters())
    # the dummy linear from the mock base model has 2 parameters
    assert len(base_model_params) == 2
    for param in base_model_params:
        assert param.requires_grad


def test_forward(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    num_classes = int(torch.max(targets) + 1)

    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"logits"}
    logits = output["logits"]
    assert logits.shape == (batch_size, seq_len, num_classes)
    assert_logits = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
        ]
    )

    torch.testing.assert_close(logits.data, assert_logits)


def test_step(batch, model):
    torch.manual_seed(42)
    loss = model.step("train", batch)
    assert_loss = torch.tensor(57.9808)
    assert loss is not None
    torch.testing.assert_close(loss, assert_loss)


def test_training_step(batch, model):
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None


def test_validation_step(batch, model):
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None


def test_test_step(batch, model):
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08


def test_configure_optimizers_with_task_learning_rate(monkeypatch):
    model = get_model(
        monkeypatch=monkeypatch,
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
        add_dummy_linear=True,
        task_learning_rate=0.1,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    param_group = optimizer.param_groups[0]
    assert param_group["lr"] == 1e-05
    # the dummy linear from the mock base model has 2 parameters
    assert len(param_group["params"]) == 2
    assert param_group["params"][0].shape == torch.Size([99, 10])
    assert param_group["params"][1].shape == torch.Size([99])
    param_group = optimizer.param_groups[1]
    assert param_group["lr"] == 0.1
    # the classifier head has 5 parameters
    assert len(param_group["params"]) == 5
    assert param_group["params"][0].shape == torch.Size([5, 10])
    assert param_group["params"][1].shape == torch.Size([5])
