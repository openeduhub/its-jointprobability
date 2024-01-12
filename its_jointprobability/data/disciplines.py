from pathlib import Path
from typing import NamedTuple, Optional

import torch
from icecream import ic
from its_jointprobability.utils import balanced_subset_mask


class Data(NamedTuple):
    docs: torch.Tensor
    targets: torch.Tensor


def get_train_data(
    path: Path, n: Optional[int] = None, always_include_confirmed: bool = True
) -> Data:
    train_targets: torch.Tensor = torch.load(
        path / "train_targets",
        map_location=torch.device("cpu"),
    )
    kept = torch.Tensor(balanced_subset_mask(train_targets, n)).bool()

    if always_include_confirmed:
        redaktionsbuffet_mask = torch.load(
            path / "redaktionsbuffet_train", map_location=torch.device("cpu")
        )
        kept = torch.logical_or(kept, redaktionsbuffet_mask)

    train_data: torch.Tensor = torch.load(
        path / "train_data_labeled",
        map_location=torch.device("cpu"),
    )[kept]

    ic(train_targets[kept].sum(-2))
    return Data(docs=train_data, targets=train_targets[kept])


def get_test_data(path: Path) -> Data:
    test_targets: torch.Tensor = torch.load(
        path / "test_targets",
        map_location=torch.device("cpu"),
    )
    test_data: torch.Tensor = torch.load(
        path / "test_data_labeled",
        map_location=torch.device("cpu"),
    )

    ic(test_targets.sum(-2))
    return Data(docs=test_data, targets=test_targets)


class Meta_Data(NamedTuple):
    uris: tuple[str, ...]
    titles: tuple[str, ...]
    uri_title_dict: dict[str, str]
    word_id_meanings: dict[int, str]


def get_metadata(path: Path) -> Meta_Data:
    uris: tuple[str, ...] = tuple(torch.load(path / "uris"))
    uri_title_dict: dict[str, str] = torch.load(path / "uri_title_dict")
    titles: tuple[str, ...] = tuple(uri_title_dict[uri] for uri in uris)
    dictionary: dict[int, str] = torch.load(path / "dictionary")

    return Meta_Data(
        uris=uris,
        uri_title_dict=uri_title_dict,
        titles=titles,
        word_id_meanings=dictionary,
    )
