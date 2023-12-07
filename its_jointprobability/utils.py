# use a custom batching strategy
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from typing import Optional, TypeVar

import nlprep.spacy.props as nlp
import torch
import torch.nn.functional as F
from nlprep import tokenize_documents
from pydantic import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def texts_to_bow_tensor(*texts, token_dict) -> torch.Tensor:
    keys = list(token_dict.keys())
    tokens = list(token_dict.values())
    tokens_set = set(token_dict.values())
    # tokenize the text
    docs = list(tokenize_documents(texts, nlp.tokenize_as_lemmas))
    # select only the tokens from the dictionary
    docs_as_tensor = torch.stack(
        [
            torch.tensor(
                [
                    keys[tokens.index(token.lower())]
                    for token in doc
                    if token.lower() in tokens_set
                ],
                device=device,
            )
            for doc in docs
        ]
    )
    return F.one_hot(docs_as_tensor, num_classes=len(tokens)).sum(-2).float()


T = TypeVar("T")


def labels_to_tensor(
    *labels_col: Iterable[T], label_values: Sequence[T]
) -> torch.Tensor:
    """Transform the given labels to a boolean tensor"""
    labels_indexes = [
        torch.tensor([label_values.index(label) for label in labels], device=device)
        for labels in labels_col
    ]

    return torch.stack(
        [
            F.one_hot(labels, num_classes=len(label_values)).sum(-2).float()
            for labels in labels_indexes
        ]
    )


def get_sequential_batch_strategy(
    length: int, batch_size: Optional[int]
) -> Iterator[tuple[bool, list[int]]]:
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


def get_random_batch_strategy(
    length: int, batch_size: Optional[int]
) -> Iterator[tuple[bool, list[int]]]:
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


def batch_to_list(batch_strategy: Iterator[tuple[bool, list[int]]]) -> list[list[int]]:
    results: list[list[int]] = []
    for last_batch, batch in batch_strategy:
        results.append(batch)
        if last_batch:
            return results

    return results


class Quality_Result(BaseModel):
    accuracy: float | list[float]
    precision: float | list[float]
    recall: float | list[float]
    f1_score: float | list[float]
    cutoff: float


def quality_measures(
    samples: torch.Tensor,
    labels: torch.Tensor,
    cutoff: Optional[float] = None,
    mean_dim: Optional[int] = None,
    parallel_dim: Optional[int] = None,
    use_median: bool = False,
) -> Quality_Result:
    samples = (
        samples
        if mean_dim is None
        else (samples.mean(mean_dim) if not use_median else samples.median(mean_dim)[0])
    )

    if cutoff is None:
        cutoffs = samples.min() + (samples.max() - samples.min()) / (
            1 + torch.exp(-torch.arange(-100, 100, device=device) / 10)
        )
        scores = [
            quality_measures(samples, labels, float(cutoff), None, None, use_median)
            for cutoff in cutoffs
        ]
        f1_scores = torch.tensor([score.f1_score for score in scores]).nan_to_num()
        optim_cutoff = float(cutoffs[f1_scores.argmax()])
        return quality_measures(
            samples, labels, optim_cutoff, None, parallel_dim, use_median
        )

    labels = labels.bool()
    predictions = samples >= cutoff
    is_true_positive = torch.logical_and(labels, predictions)
    is_true_negative = torch.logical_and(~labels, ~predictions)
    is_false_positive = torch.logical_and(~labels, predictions)
    is_false_negative = torch.logical_and(labels, ~predictions)

    # negative indices start at the far right
    n = len(samples.shape)
    parallel_dims = (
        {}
        if parallel_dim is None
        else {parallel_dim if parallel_dim >= 0 else n + parallel_dim}
    )

    dims = [index for index in range(len(samples.shape)) if index not in parallel_dims]

    tp = is_true_positive.sum(dims)
    tn = is_true_negative.sum(dims)
    fp = is_false_positive.sum(dims)
    fn = is_false_negative.sum(dims)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return Quality_Result(
        accuracy=((tp + tn) / torch.ones_like(samples).sum(dims)).tolist(),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1_score=(2 * (precision * recall) / (precision + recall)).tolist(),
        cutoff=cutoff,
    )
