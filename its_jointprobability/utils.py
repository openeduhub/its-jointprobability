import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import reduce
from typing import Optional, TypeVar

import nlprep.spacy.props as nlp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nlprep import Collection, partial, tokenize_documents
from pydantic import BaseModel

T = TypeVar("T")


def balanced_subset_mask(
    target: torch.Tensor, target_size_per_category: Optional[int]
) -> list[bool]:
    def find_split(
        tensor: torch.Tensor,
        target_count: int,
        index_bounds: Optional[tuple[int, int]] = None,
    ) -> Optional[int]:
        """Find the split point using binary search"""
        if index_bounds is None:
            index_bounds = (0, tensor.shape[-1] - 1)

        index = int(np.mean(index_bounds))
        lower_lim, upper_lim = index_bounds

        if index >= tensor.shape[-1] or lower_lim > upper_lim:
            return None

        count = tensor[..., :index].sum()
        if count == target_count:
            return index
        elif count < target_count:
            return find_split(tensor, target_count, index_bounds=(index + 1, upper_lim))
        else:
            return find_split(tensor, target_count, index_bounds=(lower_lim, index - 1))

    if target_size_per_category is None:
        return [True for _ in range(len(target))]

    # find the split point for each target category
    splits = [
        find_split(target[:, index], target_size_per_category)
        for index in range(target.shape[-1])
    ]

    # get the indices for all relevant and necessary data
    kept: list[torch.Tensor] = []
    for index, split in enumerate(splits):
        kept_category = torch.zeros_like(target[:, index])
        kept_category[:split] = 1
        kept.append(torch.logical_and(target[:, index], kept_category))
    kept_tensor: torch.Tensor = reduce(torch.logical_or, kept)

    return kept_tensor.tolist()


def texts_to_bow_tensor(
    *texts, tokens: Sequence[str], device: Optional[torch.device] = None
) -> torch.Tensor:
    """Helper function to turn texts into the format used in the model."""
    tokens_set = set(tokens)
    # tokenize the text
    docs = list(tokenize_documents(texts, nlp.tokenize_as_lemmas))
    # select only the tokens from the dictionary
    docs_as_tensor = torch.stack(
        [
            torch.tensor(
                [
                    tokens.index(token.lower())
                    for token in doc
                    if token.lower() in tokens_set
                ],
                device=device,
            )
            for doc in docs
        ]
    )
    return F.one_hot(docs_as_tensor, num_classes=len(tokens)).sum(-2).float()


Batch_Strategy = Iterator[tuple[bool, list[int]]]


def get_sequential_batch_strategy(
    length: int, batch_size: Optional[int]
) -> Batch_Strategy:
    """A batching strategy that sequentially selects data."""
    if batch_size is None:
        while True:
            yield True, list(range(length))
    i = 0
    while True:
        results = [
            index
            for index in range(i * batch_size, (i + 1) * batch_size)
            if index < length
        ]
        i += 1
        # this was the last batch if the new batch starts with indices
        # higher than the total length
        last_batch_in_epoch = i * batch_size >= length
        yield last_batch_in_epoch, results
        if last_batch_in_epoch:
            i = 0


def get_random_batch_strategy(length: int, batch_size: Optional[int]) -> Batch_Strategy:
    """
    A batching strategy that randomly selects data.

    Does not select data that has already been selected within the same epoch.
    """
    if batch_size is None:
        while True:
            yield True, list(range(length))

    last_batch_index = math.ceil(length / batch_size) - 1
    while True:
        available_hits = random.sample(range(length), k=length)
        for i in range(math.ceil(length / batch_size)):
            results = available_hits[i * batch_size : (i + 1) * batch_size]
            last_batch_in_epoch = i == last_batch_index
            yield last_batch_in_epoch, results


#: A batch is a sequence of individual data (Tensor) or missing data (None)
Batch = Sequence[torch.Tensor | None]

#: A data loader is an (infinite) iterator that yields a the following:
#:
#: 1. An indicator whether this is the the last batch of the epoch.
#: 2. The batched data itself.
Data_Loader = Iterator[tuple[bool, Batch]]


def get_batch_size(batch: Batch) -> int:
    return max(len(batch_item) if batch_item is not None else 0 for batch_item in batch)


def _abstract_data_loader(
    *tensors: torch.Tensor | None,
    batch_strategy: Batch_Strategy,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype | Collection[torch.dtype]] = None,
) -> Data_Loader:
    for last_batch_in_epoch, batch in batch_strategy:
        batched_data = [
            tensor[batch].to(device) if tensor is not None else None
            for tensor in tensors
        ]
        if dtype is not None:
            if isinstance(dtype, Iterable):
                batched_data = [
                    batch.type(dtype_entry) if batch is not None else None
                    for batch, dtype_entry in zip(batched_data, dtype)
                ]
            else:
                batched_data = [
                    batch.type(dtype) if batch is not None else None
                    for batch in batched_data
                ]

        yield last_batch_in_epoch, batched_data


def _empty_data_loader(*tensors) -> Data_Loader:
    while True:
        yield True, [None for _ in tensors]


def _data_loader_from_strategy(
    get_batch_strategy: Callable[[int, int], Batch_Strategy],
    *tensors: torch.Tensor | None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype | Collection[torch.dtype]] = None,
) -> Data_Loader:
    n = None
    for tensor in tensors:
        if tensor is None:
            continue
        if n is None:
            n = len(tensor)
        assert n == len(tensor)

    if n is None:
        return _empty_data_loader(*tensors)

    if batch_size is None:
        batch_size = _default_batch_size(n)

    batch_strategy = get_batch_strategy(n, batch_size)

    return _abstract_data_loader(
        *tensors, batch_strategy=batch_strategy, device=device, dtype=dtype
    )


#: Converts the given data into an infinite sequence of random batches.
#:
#: :param batch_size: If given, create batches of this size.
#:     Note that the final batch of each epoch may be smaller, in order to
#:     ensure that no data is visited twice per epoch.
#: :param device: If given, move each batch to this device.
#:     This is useful when dealing with particularly large data sets that do
#:     not fully fit e.g. onto a GPU.
#: :param dtype: If given, convert each batch to this data type.
#:     If this is a singular value, convert all data to this type.
#:     If this is a collection, convert the batch of the nth data tensor
#:     to the nth dtype in the sequence.
default_data_loader = partial(_data_loader_from_strategy, get_random_batch_strategy)

#: Like the default data loader, but sequentially batches the data.
sequential_data_loader = partial(
    _data_loader_from_strategy, get_sequential_batch_strategy
)


def _default_batch_size(n: int, num_options: int = 5) -> int:
    center = np.ceil(n ** (2 / 3))
    options = np.arange(center - num_options, center + num_options + 1)
    options = options[options > 0]
    remainders = n % options
    remainders[remainders == 0] = options[remainders == 0]

    return int(options[remainders.argmax()])


def batch_to_list(
    batch_strategy: Iterator[tuple[bool, Sequence[T]]]
) -> list[Sequence[T]]:
    """
    Turn a batch strategy or data loader into a list of batches (of one epoch).
    """
    results: list[Sequence[T]] = []
    for last_batch, batch in batch_strategy:
        results.append(batch)
        if last_batch:
            return results

    return results


class Quality_Result(BaseModel):
    """Various quality metrics for predictions."""

    accuracy: float | list[float]
    precision: float | list[float]
    recall: float | list[float]
    f1_score: float | list[float]
    cutoff: float


def quality_measures(
    samples: torch.Tensor,
    targets: torch.Tensor,
    cutoff: Optional[float] = None,
    mean_dim: Optional[int] = None,
    parallel_dim: Optional[int] = None,
    use_median: bool = False,
) -> Quality_Result:
    """Compute quality metrics for a given set of predictions and their true values."""
    samples = (
        samples
        if mean_dim is None
        else (samples.mean(mean_dim) if not use_median else samples.median(mean_dim)[0])
    )

    # automatically compute the cutoff at which to consider a prediction
    # to be positive
    if cutoff is None:
        # the cutoffs to try
        # this follows a logistic interpolation between the min and max
        cutoffs = samples.min() + (samples.max() - samples.min()) / (
            1 + torch.exp(-torch.arange(-100, 100, device=samples.device) / 10)
        )

        # the quality measures for each cutoff
        scores = [
            quality_measures(samples, targets, float(cutoff), None, None, use_median)
            for cutoff in cutoffs
        ]
        f1_scores = torch.tensor([score.f1_score for score in scores]).nan_to_num()

        # select the cutoff where the F1 score is maximized
        optim_cutoff = float(cutoffs[f1_scores.argmax()])
        return quality_measures(
            samples, targets, optim_cutoff, None, parallel_dim, use_median
        )

    targets = targets.bool()
    predictions = samples >= cutoff

    # negative indices start at the far right
    n = len(samples.shape)
    parallel_dims = (
        {}
        if parallel_dim is None
        else {parallel_dim if parallel_dim >= 0 else n + parallel_dim}
    )
    dims = [index for index in range(len(samples.shape)) if index not in parallel_dims]

    # compute the quality metrics
    is_true_positive = torch.logical_and(targets, predictions)
    is_true_negative = torch.logical_and(~targets, ~predictions)
    is_false_positive = torch.logical_and(~targets, predictions)
    is_false_negative = torch.logical_and(targets, ~predictions)

    tp = is_true_positive.sum(dims)  # true positives
    tn = is_true_negative.sum(dims)  # true negatives
    fp = is_false_positive.sum(dims)  # false positives
    fn = is_false_negative.sum(dims)  # false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return Quality_Result(
        # accuracy is the relative amount of correct predictions
        accuracy=((tp + tn) / torch.ones_like(samples).sum(dims)).tolist(),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1_score=(2 * (precision * recall) / (precision + recall)).tolist(),
        cutoff=cutoff,
    )
