from typing import Any, Dict, Optional, Tuple

import torch
import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from .components.seq2seq_encoder import build_seq2seq_encoder

ModelBatchEncodingType: TypeAlias = BatchEncoding
ModelBatchOutputType: TypeAlias = Dict[str, Any]

StepBatchEncodingType: TypeAlias = Tuple[
    Dict[str, Tensor],
    Optional[Tensor],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"


@PyTorchIEModel.register()
class TokenClassificationModelWithSeq2SeqEncoderAndCrf(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_token_id: int = -100,
        ignore_index: int = 0,
        use_crf: bool = True,
        freeze_base_model: bool = False,
        seq2seq_encoder: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.ignore_index = ignore_index

        self.learning_rate = learning_rate
        self.label_pad_token_id = label_pad_token_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        if freeze_base_model:
            self.model.requires_grad_(False)

        hidden_size = config.hidden_size
        self.seq2seq_encoder = None
        if seq2seq_encoder is not None:
            self.seq2seq_encoder, hidden_size = build_seq2seq_encoder(
                config=seq2seq_encoder, input_size=hidden_size
            )

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.crf = CRF(num_tags=num_classes, batch_first=True) if use_crf else None

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(self, inputs: ModelBatchEncodingType) -> ModelBatchOutputType:
        labels = inputs.pop("labels", None)

        outputs = self.model(**inputs)
        sequence_output = outputs[0]

        if self.seq2seq_encoder is not None:
            sequence_output = self.seq2seq_encoder(sequence_output)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if "attention_mask" in inputs:
            mask_bool = inputs["attention_mask"].to(torch.bool)
        else:
            # the crf expects a bool mask and fails if it is not provided
            mask_bool = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
        if labels is not None:
            if self.crf is not None:
                # replace the padding labels with the ignore_index (not inplace to mitigate side effects)
                labels_valid = torch.where(
                    labels == self.label_pad_token_id,
                    torch.tensor(self.ignore_index).to(device=logits.device),
                    labels,
                )
                log_likelihood = self.crf(emissions=logits, tags=labels_valid, mask=mask_bool)
                loss = -log_likelihood
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        if self.crf is not None:
            decoded_tags = self.crf.decode(emissions=logits, mask=mask_bool)
            # re-construct the logits from the decoded tags to be compatible with the default taskmodule
            seq_len = logits.shape[1]
            padded_tags = [
                tags + [self.ignore_index] * (seq_len - len(tags)) for tags in decoded_tags
            ]
            padded_tags_tensor = torch.tensor(padded_tags, dtype=torch.long, device=logits.device)
            logits = torch.nn.functional.one_hot(
                padded_tags_tensor, num_classes=self.num_classes
            ).to(torch.float)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def step(
        self,
        stage: str,
        batch: StepBatchEncodingType,
    ):
        input_, target = batch
        assert target is not None, "target has to be available for training"

        input_["labels"] = target
        output = self(input_)

        loss = output.loss
        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        target_flat = target.view(-1)

        valid_indices = target_flat != self.label_pad_token_id
        valid_logits = output.logits.view(-1, self.num_classes)[valid_indices]
        valid_target = target_flat[valid_indices]

        f1 = self.f1[f"stage_{stage}"]
        f1(valid_logits, valid_target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: StepBatchEncodingType, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepBatchEncodingType, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: StepBatchEncodingType, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
