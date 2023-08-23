import logging
from typing import Any, Callable, Dict, Tuple

import torch
from torch import Tensor, cat, nn

# possible pooler types
CLS_TOKEN = "cls_token"  # CLS token
START_TOKENS = "start_tokens"  # MTB start tokens concat
MENTION_POOLING = "mention_pooling"  # mention token pooling and concat


logger = logging.getLogger(__name__)


def pool_cls(hidden_state: Tensor, **kwargs) -> Tensor:
    return hidden_state[:, 0, :]


class AtIndexPooler(nn.Module):
    """Pooler that takes the hidden state at a given index.

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.
        offset: The offset to add to the indices.
    """

    def __init__(self, input_dim: int, num_indices: int = 2, offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.offset = offset
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(self, hidden_state: Tensor, indices: Tensor, **kwargs) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of indices [{indices.shape[1]}] has to be the same as num_types [{self.num_indices}]."
            )

        # times num_types due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx, current_indices in enumerate(indices):
            current_embeddings = [
                hidden_state[batch_idx, current_indices[i] + self.offset, :]
                if current_indices[i] >= 0
                else self.missing_embeddings[i]
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)
        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


class ArgumentWrappedPooler(nn.Module):
    """Wraps a pooler and maps the arguments to the pooler.

    Args:
        pooler: The pooler to wrap.
        argument_mapping: A mapping from the arguments of the forward method to the arguments of the pooler.
    """

    def __init__(self, pooler: nn.Module, argument_mapping: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.argument_mapping = argument_mapping

    def forward(self, hidden_state: Tensor, **kwargs) -> Tensor:
        pooler_kwargs = {}
        for k, v in kwargs.items():
            if k in self.argument_mapping:
                pooler_kwargs[self.argument_mapping[k]] = v
        return self.pooler(hidden_state, **pooler_kwargs)

    @property
    def output_dim(self) -> int:
        return self.pooler.output_dim


class SpanMaxPooler(nn.Module):
    """Pooler that takes the max hidden state over a span.

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.
    """

    def __init__(self, input_dim: int, num_indices: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(
        self, hidden_state: Tensor, start_indices: Tensor, end_indices: Tensor, **kwargs
    ) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if start_indices.shape != end_indices.shape:
            raise ValueError(
                f"start_indices shape [{start_indices.shape}] has to be the same as end_indices shape "
                f"[{end_indices.shape}]."
            )
        if start_indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of indices [{start_indices.shape[1]}] has to be the same as num_types [{self.num_indices}]."
            )

        # times num_indices due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx in range(batch_size):
            current_start_indices = start_indices[batch_idx]
            current_end_indices = end_indices[batch_idx]
            current_embeddings = [
                torch.amax(
                    hidden_state[batch_idx, current_start_indices[i] : current_end_indices[i], :],
                    0,
                )
                if current_start_indices[i] >= 0 and current_end_indices[i] >= 0
                else self.missing_embeddings[i]
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)

        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


def get_pooler_and_output_size(config: Dict[str, Any], input_dim: int) -> Tuple[Callable, int]:
    pooler_config = dict(config)
    pooler_type = pooler_config.pop("type", CLS_TOKEN)
    if pooler_type == CLS_TOKEN:
        return pool_cls, input_dim
    elif pooler_type == START_TOKENS:
        pooler = AtIndexPooler(input_dim=input_dim, offset=-1, **pooler_config)
        pooler_wrapped = ArgumentWrappedPooler(
            pooler=pooler, argument_mapping={"start_indices": "indices"}
        )
        return pooler_wrapped, pooler.output_dim
    elif pooler_type == MENTION_POOLING:
        pooler = SpanMaxPooler(input_dim=input_dim, **pooler_config)
        return pooler, pooler.output_dim
    else:
        raise ValueError(f"Unknown pooler type {pooler_type}")
