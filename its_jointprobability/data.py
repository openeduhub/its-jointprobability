from functools import reduce
from pathlib import Path
from typing import NamedTuple, Optional, TypeVar

import numpy as np
import pyro
import torch
from data_utils.default_pipelines.data import BoW_Data, subset_data_points
from data_utils.default_pipelines.its_jointprobability import generate_data
from data_utils.defaults import Fields

from its_jointprobability.models.model import Model
from its_jointprobability.utils import balanced_subset_mask


class Split_Data(NamedTuple):
    train: BoW_Data
    test: BoW_Data


def make_data(
    path: Path, n: Optional[int] = None, always_include_confirmed=True, seed=0, **kwargs
) -> Split_Data:
    # create the cache directory
    nlp_cache = path / "nlp_cache"
    nlp_cache.mkdir(parents=True, exist_ok=True)

    data = generate_data(
        path / "data.json",
        target_fields=[
            Fields.EDUCATIONAL_CONTEXT.value,
            Fields.INTENDED_ENDUSER.value,
            Fields.TAXONID.value,
        ],
        cache_dir=nlp_cache,
        **kwargs,
    )

    split_data = split_train_test(data)

    def get_train_data(x: Split_Data):
        return torch.tensor(x.train.target_data[Fields.TAXONID.value].arr)

    # if the test data set is empty, create one from the training data
    # by randomly taking 5% of the editorially confirmed materials
    if len(split_data.test.editor_arr) == 0:
        size = split_data.train.editor_arr.sum() // 20
        editor_indices = np.where(split_data.train.editor_arr)[0]
        rng = np.random.default_rng(seed=seed)
        test_indices = rng.choice(editor_indices, size)
        test_mask = np.zeros_like(split_data.train.editor_arr, dtype=bool)
        test_mask[test_indices] = True

        split_data = Split_Data(
            train=subset_data_points(split_data.train, np.where(~test_mask)[0]),
            test=subset_data_points(split_data.train, np.where(test_mask)[0]),
        )

    kept = np.array(
        balanced_subset_mask(
            target=get_train_data(split_data), target_size_per_category=n
        ),
        dtype=bool,
    )

    if always_include_confirmed:
        kept = np.logical_or(kept, split_data.train.editor_arr)

    print(
        f"{np.logical_and(kept, ~split_data.train.editor_arr).sum()} / {kept.sum()} training materials have not been confirmed by editors"
    )

    return Split_Data(
        train=subset_data_points(split_data.train, kept), test=split_data.test
    )


def split_train_test(data: BoW_Data) -> Split_Data:
    test_indices = reduce(
        np.logical_or,
        [target_data.in_test_set for target_data in data.target_data.values()],
        np.zeros_like(data.editor_arr, dtype=bool),
    )

    return Split_Data(
        test=subset_data_points(data, np.where(test_indices)[0]),
        train=subset_data_points(data, np.where(~test_indices)[0]),
    )


def save_data(path: Path, data: Split_Data):
    torch.save(data, path / "data.pt")


def load_data(path: Path) -> Split_Data:
    return torch.load(path / "data.pt")


Model_Subtype = TypeVar("Model_Subtype", bound=Model)


def save_model(model: Model, path: Path, suffix: str = ""):
    cls_name = model.__class__.__name__
    if suffix:
        cls_name += f"_{suffix}"
    torch.save(model.args, path / f"{cls_name}_kwargs.pt")
    torch.save(model.state_dict(), path / f"{cls_name}_state.pt")
    pyro.get_param_store().save(path / f"{cls_name}_pyro.pt")


def load_model(
    model_class: type[Model_Subtype], path: Path, device: torch.device, suffix: str = ""
) -> Model_Subtype:
    cls_name = model_class.__name__
    if suffix:
        cls_name += f"_{suffix}"
    pyro.get_param_store().load(path / f"{cls_name}_pyro.pt", map_location=device)

    kwargs = torch.load(path / f"{cls_name}_kwargs.pt", map_location=device)
    state_dict = torch.load(path / f"{cls_name}_state.pt", map_location=device)
    model = model_class(device=device, **kwargs)

    model.load_state_dict(state_dict)

    # ensure that the imported model is in evaluation mode
    model.eval()

    return model
