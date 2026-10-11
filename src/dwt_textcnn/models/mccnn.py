"""Plain multichannel CNN baseline from the IJCAI 2018 reference paper."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MCCNN(nn.Module):
    """Two trainable embedding channels with shared convolution filters."""

    def __init__(
        self,
        pretrained_weights: torch.Tensor,
        *,
        pad_id: int = 0,
        kernel_sizes: Sequence[int] = (5, 6, 7),
        filters_per_size: int = 100,
        dropout: float = 0.5,
        num_classes: int = 2,
        initialization_std: float = 0.1,
    ) -> None:
        super().__init__()
        if pretrained_weights.ndim != 2:
            raise ValueError("pretrained_weights must have shape [vocabulary, dimension]")
        vocabulary_size, embedding_dimension = pretrained_weights.shape
        if not 0 <= pad_id < vocabulary_size:
            raise ValueError("pad_id is outside the vocabulary")
        if not kernel_sizes or any(size < 1 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")
        if filters_per_size < 1 or num_classes < 2:
            raise ValueError("filters_per_size and num_classes are invalid")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if initialization_std <= 0:
            raise ValueError("initialization_std must be positive")

        self.pad_id = pad_id
        self.embedding_dimension = embedding_dimension
        self.kernel_sizes = tuple(kernel_sizes)
        self.filters_per_size = filters_per_size
        self.initialization_std = initialization_std

        weights = pretrained_weights.detach().to(dtype=torch.float32).clone()
        weights[pad_id].zero_()
        self.pretrained_embedding = nn.Embedding.from_pretrained(
            weights, freeze=False, padding_idx=pad_id
        )
        self.random_embedding = nn.Embedding(
            vocabulary_size, embedding_dimension, padding_idx=pad_id
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dimension,
                out_channels=filters_per_size,
                kernel_size=kernel_size,
                bias=True,
            )
            for kernel_size in self.kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            filters_per_size * len(self.kernel_sizes), num_classes
        )
        self._initialize_trainable_parameters()
        self.enforce_padding_zero()

    def _truncated_normal(self, tensor: torch.Tensor) -> None:
        bound = 2.0 * self.initialization_std
        nn.init.trunc_normal_(
            tensor,
            mean=0.0,
            std=self.initialization_std,
            a=-bound,
            b=bound,
        )

    def _initialize_trainable_parameters(self) -> None:
        self._truncated_normal(self.random_embedding.weight)
        for convolution in self.convolutions:
            self._truncated_normal(convolution.weight)
            nn.init.zeros_(convolution.bias)
        self._truncated_normal(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @torch.no_grad()
    def enforce_padding_zero(self) -> None:
        self.pretrained_embedding.weight[self.pad_id].zero_()
        self.random_embedding.weight[self.pad_id].zero_()

    def extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] < max(self.kernel_sizes):
            raise ValueError(
                f"sequence length must be at least {max(self.kernel_sizes)}"
            )

        pretrained = self.pretrained_embedding(input_ids).transpose(1, 2)
        random = self.random_embedding(input_ids).transpose(1, 2)
        lengths = torch.count_nonzero(input_ids != self.pad_id, dim=1)
        pooled_features = []
        for kernel_size, convolution in zip(
            self.kernel_sizes, self.convolutions, strict=True
        ):
            pretrained_features = F.relu(convolution(pretrained))
            random_features = F.relu(convolution(random))
            merged = pretrained_features + random_features
            valid_widths = torch.clamp(lengths - kernel_size + 1, min=1)
            positions = torch.arange(merged.shape[2], device=input_ids.device)
            valid_positions = positions.unsqueeze(0) < valid_widths.unsqueeze(1)
            merged = merged.masked_fill(~valid_positions.unsqueeze(1), -torch.inf)
            pooled_features.append(torch.amax(merged, dim=2))
        return torch.cat(pooled_features, dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(input_ids)
        return self.classifier(self.dropout(features))
