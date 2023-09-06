import pytest
import torch
import transformers
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutputWithPast,
)

from pie_models.models import SequenceClassificationModel
from pie_models.models.components.pooler import CLS_TOKEN, MENTION_POOLING, START_TOKENS
from tests import _config_to_str

# only certain combinations of pooler types and use_auto_model_for_sequence_classification are supported
CONFIGS = [
    {"pooler": CLS_TOKEN},
    {"pooler": START_TOKENS},
    {"pooler": MENTION_POOLING},
    {"pooler": CLS_TOKEN, "use_auto_model_for_sequence_classification": True},
]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    return CONFIGS_DICT[request.param]


@pytest.fixture
def inputs():
    result_dict = {
        "input_ids": torch.tensor(
            [
                [
                    101,
                    28998,
                    13832,
                    3121,
                    2340,
                    138,
                    28996,
                    1759,
                    1120,
                    28999,
                    139,
                    28997,
                    119,
                    102,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    146,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    145,
                    119,
                    1262,
                    1771,
                    28999,
                    146,
                    28997,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    144,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    28998,
                    146,
                    28996,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    150,
                    28996,
                    1759,
                    1120,
                    28999,
                    151,
                    28997,
                    119,
                    1262,
                    1122,
                    1771,
                    152,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28998,
                    1122,
                    28996,
                    1771,
                    28999,
                    152,
                    28997,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28999,
                    1122,
                    28997,
                    1771,
                    28998,
                    152,
                    28996,
                    119,
                    102,
                ],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "pooler_start_indices": torch.tensor(
            [[2, 10], [5, 13], [5, 17], [17, 11], [5, 13], [14, 18], [18, 14]]
        ),
        "pooler_end_indices": torch.tensor(
            [[6, 11], [9, 14], [9, 18], [18, 12], [9, 14], [15, 19], [19, 15]]
        ),
    }

    return result_dict


@pytest.fixture
def targets():
    return torch.tensor([0, 1, 2, 3, 1, 2, 3])


# @pytest.fixture()
# def pooler_type(config):
#    return config.get("pooler", None)


def get_model(
    monkeypatch,
    pooler,
    batch_size,
    seq_len,
    num_classes,
    add_dummy_linear=False,
    model_type="bert",
    base_model_prefix="dummy_linear",
    **model_kwargs,
):
    class MockConfig:
        def __init__(
            self,
            hidden_size: int = 10,
            classifier_dropout: float = 0.1,
            model_type="bert",
            num_labels=None,
        ) -> None:
            self.hidden_size = hidden_size
            self.model_type = model_type
            if self.model_type == "distilbert":
                self.seq_classif_dropout = classifier_dropout
            elif self.model_type == "albert":
                self.classifier_dropout_prob = classifier_dropout
            else:
                self.classifier_dropout = classifier_dropout
            self.num_labels = num_labels

    class MockModel(torch.nn.Module):
        def __init__(self, batch_size, seq_len, hidden_size, add_dummy_linear) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)

        def __call__(self, *args, **kwargs):
            last_hidden_state = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
            return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)

        def resize_token_embeddings(self, new_num_tokens):
            pass

    class MockSequenceModel(torch.nn.Module):
        def __init__(
            self, batch_size, seq_len, hidden_size, num_classes, add_dummy_linear
        ) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)
            self.classifier = torch.nn.Linear(self.hidden_size, self.num_classes)

        def __call__(self, *args, **kwargs):
            logits = torch.rand(self.batch_size, self.num_classes)
            loss = torch.rand(1)[0]
            return SequenceClassifierOutputWithPast(logits=logits, loss=loss)

        def resize_token_embeddings(self, new_num_tokens):
            pass

    hidden_size = 10
    tokenizer_vocab_size = 30000

    def get_config(*args, **kwargs):
        return MockConfig(
            hidden_size=hidden_size,
            classifier_dropout=0.1,
            model_type=model_type,
            num_labels=num_classes,
        )

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        get_config,
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name_or_path, device_map, config: MockModel(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            add_dummy_linear=add_dummy_linear,
        ),
    )

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda model_name_or_path, config, device_map: MockSequenceModel(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_classes=config.num_labels,
            add_dummy_linear=add_dummy_linear,
        ),
    )

    # set seed to make the classifier deterministic
    torch.manual_seed(42)
    result = SequenceClassificationModel(
        model_name_or_path="some-model-name",
        num_classes=num_classes,
        tokenizer_vocab_size=tokenizer_vocab_size,
        ignore_index=0,
        pooler=pooler,
        # disable warmup because it would require a trainer and a datamodule to get the total number of training steps
        warmup_proportion=0.0,
        base_model_prefix=base_model_prefix,
        **model_kwargs,
    )
    assert not result.is_from_pretrained

    return result


@pytest.fixture
def num_classes(targets):
    return int(max(targets) + 1)


@pytest.fixture
def model(monkeypatch, inputs, num_classes, config):
    return get_model(
        monkeypatch=monkeypatch,
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
        num_classes=num_classes,
        **config,
    )


def test_forward(inputs, model, config, num_classes):
    batch_size, seq_len = inputs["input_ids"].shape
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    logits = output["logits"]
    if config == {"use_auto_model_for_sequence_classification": True, "pooler": CLS_TOKEN}:
        assert set(output) == {"logits", "loss"}
        logits = output["logits"]
        assert logits.shape == (batch_size, num_classes)
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        0.8822692632675171,
                        0.9150039553642273,
                        0.38286375999450684,
                        0.9593056440353394,
                    ],
                    [
                        0.3904482126235962,
                        0.600895345211029,
                        0.2565724849700928,
                        0.7936413288116455,
                    ],
                    [
                        0.9407714605331421,
                        0.13318592309951782,
                        0.9345980882644653,
                        0.5935796499252319,
                    ],
                    [
                        0.8694044351577759,
                        0.5677152872085571,
                        0.7410940527915955,
                        0.42940449714660645,
                    ],
                    [
                        0.8854429125785828,
                        0.5739044547080994,
                        0.2665800452232361,
                        0.6274491548538208,
                    ],
                    [
                        0.26963168382644653,
                        0.4413635730743408,
                        0.2969208359718323,
                        0.831685483455658,
                    ],
                    [
                        0.10531491041183472,
                        0.26949483156204224,
                        0.3588126301765442,
                        0.19936376810073853,
                    ],
                ]
            ),
        )
        loss = output["loss"]
        torch.testing.assert_close(loss, torch.tensor(0.5471915602684021))
    elif config == {"pooler": CLS_TOKEN}:
        assert set(output) == {"logits"}
        assert logits.shape == (batch_size, 4)
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        1.25642395019531250000,
                        0.39357030391693115234,
                        -0.29225713014602661133,
                        0.79580175876617431641,
                    ],
                    [
                        0.53891825675964355469,
                        0.34787857532501220703,
                        -0.24634249508380889893,
                        0.59947609901428222656,
                    ],
                    [
                        0.74169844388961791992,
                        0.30519056320190429688,
                        -0.55728095769882202148,
                        0.49557113647460937500,
                    ],
                    [
                        0.35605597496032714844,
                        0.19517414271831512451,
                        -0.00861304998397827148,
                        0.65302681922912597656,
                    ],
                    [
                        0.29554772377014160156,
                        0.71216350793838500977,
                        -0.62688910961151123047,
                        0.92307460308074951172,
                    ],
                    [
                        0.67893451452255249023,
                        0.30236703157424926758,
                        -0.35009318590164184570,
                        0.49039006233215332031,
                    ],
                    [
                        0.33092185854911804199,
                        0.47906285524368286133,
                        -0.39155155420303344727,
                        0.57707947492599487305,
                    ],
                ]
            ),
        )
    elif config == {"pooler": START_TOKENS}:
        assert set(output) == {"logits"}
        assert logits.shape == (batch_size, 4)
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        0.28744211792945861816,
                        -0.07656848430633544922,
                        0.06205615401268005371,
                        -0.94508385658264160156,
                    ],
                    [
                        0.29196885228157043457,
                        0.02899619936943054199,
                        -0.21342960000038146973,
                        -0.36053514480590820312,
                    ],
                    [
                        0.36177605390548706055,
                        -0.64715611934661865234,
                        -0.26786345243453979492,
                        -0.80762034654617309570,
                    ],
                    [
                        0.42720466852188110352,
                        -0.25489825010299682617,
                        -0.19527786970138549805,
                        -0.49900960922241210938,
                    ],
                    [
                        0.20688854157924652100,
                        -0.29307979345321655273,
                        -0.12208836525678634644,
                        -0.89110243320465087891,
                    ],
                    [
                        0.35013008117675781250,
                        -0.49105945229530334473,
                        -0.18206793069839477539,
                        -1.19002366065979003906,
                    ],
                    [
                        0.31203818321228027344,
                        -0.37706983089447021484,
                        0.07198116183280944824,
                        -0.81837034225463867188,
                    ],
                ]
            ),
        )
    elif config == {"pooler": MENTION_POOLING}:
        assert set(output) == {"logits"}
        assert logits.shape == (batch_size, 4)
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        0.48370990157127380371,
                        -0.27870815992355346680,
                        -0.13999497890472412109,
                        -0.73714041709899902344,
                    ],
                    [
                        0.54668211936950683594,
                        -0.29652747511863708496,
                        -0.26315566897392272949,
                        -0.95955950021743774414,
                    ],
                    [
                        0.22266633808612823486,
                        -0.24484989047050476074,
                        -0.03910681605339050293,
                        -0.94041651487350463867,
                    ],
                    [
                        0.42026358842849731445,
                        -0.47725573182106018066,
                        -0.30766916275024414062,
                        -0.59111309051513671875,
                    ],
                    [
                        0.23630522191524505615,
                        -0.29734912514686584473,
                        -0.30620723962783813477,
                        -0.75251650810241699219,
                    ],
                    [
                        0.14205220341682434082,
                        -0.39235562086105346680,
                        -0.41546288132667541504,
                        -0.77219748497009277344,
                    ],
                    [
                        0.44709876179695129395,
                        -0.20209559798240661621,
                        -0.18925097584724426270,
                        -0.64976799488067626953,
                    ],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")


@pytest.fixture
def batch(inputs, targets):
    return (inputs, targets)


def test_step(batch, model, config):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.step("train", batch)
    if config == {"pooler": CLS_TOKEN}:
        torch.testing.assert_close(loss, torch.tensor(1.417838096618652))
    elif config == {"pooler": START_TOKENS}:
        torch.testing.assert_close(loss, torch.tensor(1.498929619789123))
    elif config == {"pooler": MENTION_POOLING}:
        torch.testing.assert_close(loss, torch.tensor(1.489617109298706))
    elif config == {"pooler": CLS_TOKEN, "use_auto_model_for_sequence_classification": True}:
        torch.testing.assert_close(loss, torch.tensor(0.5471915602684021))
    else:
        raise ValueError(f"Unknown config: {config}")


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


@pytest.mark.parametrize(
    "pooler",
    [START_TOKENS, MENTION_POOLING],
)
def test_use_auto_model_for_sequence_classification_wrong_pooler(
    monkeypatch, pooler, inputs, targets
):
    with pytest.raises(ValueError) as excinfo:
        get_model(
            monkeypatch=monkeypatch,
            batch_size=inputs["input_ids"].shape[0],
            seq_len=inputs["input_ids"].shape[1],
            num_classes=4,
            pooler=pooler,
            use_auto_model_for_sequence_classification=True,
        )
    assert (
        str(excinfo.value)
        == "pooler type must be 'cls_token' when using AutoModelForSequenceClassification"
    )


def test_base_model_named_parameters_wrong_base_model_prefix(monkeypatch, inputs, targets):
    model = get_model(
        monkeypatch=monkeypatch,
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
        num_classes=4,
        pooler=CLS_TOKEN,
        base_model_prefix="wrong_prefix",
        use_auto_model_for_sequence_classification=True,
    )
    with pytest.raises(ValueError) as excinfo:
        model.base_model_named_parameters()
    assert (
        str(excinfo.value)
        == "No base model parameters found. Is base_model_prefix=wrong_prefix for MockSequenceModel correct?"
    )


@pytest.mark.parametrize(
    "use_auto_model_for_sequence_classification",
    [True, False],
)
def test_configure_optimizers_with_task_learning_rate(
    monkeypatch, use_auto_model_for_sequence_classification
):
    model = get_model(
        monkeypatch=monkeypatch,
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
        task_learning_rate=0.1,
        pooler=CLS_TOKEN,
        use_auto_model_for_sequence_classification=use_auto_model_for_sequence_classification,
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
    # the classifier head has 2 parameters
    assert len(param_group["params"]) == 2
    assert param_group["params"][0].shape == torch.Size([4, 10])
    assert param_group["params"][1].shape == torch.Size([4])


@pytest.mark.parametrize(
    "use_auto_model_for_sequence_classification",
    [True, False],
)
def test_freeze_base_model(
    monkeypatch, inputs, targets, use_auto_model_for_sequence_classification
):
    # set seed to make the classifier deterministic
    model = get_model(
        monkeypatch,
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
        freeze_base_model=True,
        pooler=CLS_TOKEN,
        use_auto_model_for_sequence_classification=use_auto_model_for_sequence_classification,
    )
    named_base_model_params = model.base_model_named_parameters()
    # the dummy linear from the mock base model has 2 parameters
    assert len(named_base_model_params) == 2
    for name, param in named_base_model_params.items():
        assert not param.requires_grad


@pytest.mark.parametrize(
    "model_type", ["bert", "albert", "distilbert", "roberta", "deberta", "electra", "xlm-roberta"]
)
def test_config_model_classifier_dropout(monkeypatch, model_type):
    model = get_model(
        monkeypatch,
        batch_size=7,
        seq_len=22,
        num_classes=4,
        model_type=model_type,
        pooler=CLS_TOKEN,
    )
    assert model is not None
