import json

import pytest
import torch
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from transformers import BatchEncoding

from pie_models.models import TokenClassificationModelWithSeq2SeqEncoderAndCrf
from tests import FIXTURES_ROOT

DUMP_FIXTURES = False
FIXTURES_TASKMODULE_DATA_PATH = (
    FIXTURES_ROOT / "taskmodules" / "TransformerTokenClassificationTaskModule"
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
        "num_classes": len(taskmodule.label_to_id),
    }
    with open(filepath, "w") as f:
        json.dump(converted_batch_encoding, f)
    return converted_batch_encoding


@pytest.fixture
def batch(documents):
    filepath = FIXTURES_TASKMODULE_DATA_PATH / "batch_encoding_inputs.json"
    with open(filepath) as f:
        batch_encoding = json.load(f)
    inputs = BatchEncoding(batch_encoding["inputs"], tensor_type="pt")
    targets = torch.Tensor(batch_encoding["targets"])
    num_classes = batch_encoding["num_classes"]
    return (inputs, targets, num_classes)


def get_model(batch):
    _, _, num_classes = batch
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="bert-base-cased",
        num_classes=num_classes,
        # warmup_proportion would need a trainer
        warmup_proportion=0.0,
    )
    return model


def test_get_model(batch):
    model = get_model(batch)
    assert model is not None


def test_forward(batch):
    inputs, targets, _ = batch
    batch_size, seq_len = inputs["input_ids"].shape
    model = get_model(batch)
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"logits"}
    logits = output["logits"]
    assert logits.shape == (batch_size, seq_len, model.num_classes)


# def test_step():
#     pass
#
#
# def test_training_step():
#     pass
#
#
# def test_validation_step():
#     pass
#
#
# def test_test_step():
#     pass
#
#
# def test_configure_optimizers():
#     pass
