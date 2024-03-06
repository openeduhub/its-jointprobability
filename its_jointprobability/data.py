from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np
import pyro
import torch
from data_utils.default_pipelines.data import (
    BoW_Data,
    Processed_Data,
    balanced_split,
    import_published,
)
from data_utils.default_pipelines.its_jointprobability import generate_data
from data_utils.defaults import Fields

from its_jointprobability.models.model import Model


class Split_Data(NamedTuple):
    train: BoW_Data
    test: BoW_Data


def make_data(data_dir: Path) -> Split_Data:
    nlp_cache = data_dir / "nlp_cache"
    nlp_cache.mkdir(parents=True, exist_ok=True)

    data = generate_data(
        data_dir / "data.json",
        target_fields=[
            Fields.TAXONID.value,
            Fields.EDUCATIONAL_CONTEXT.value,
            Fields.INTENDED_ENDUSER.value,
            Fields.TOPIC.value,
            Fields.LRT.value,
        ],
        cache_dir=nlp_cache,
    )

    return Split_Data(
        *balanced_split(
            data, field=Fields.TAXONID.value, ratio=0.3, randomize=True, seed=0
        )
    )


def import_data(data_dir: Path) -> Split_Data:
    data: list[Processed_Data] = list()
    names = ["train", "test"]
    for name in names:
        data.append(
            import_published(
                data_file=data_dir / f"{name}_data.csv",
                metadata_file=data_dir / f"{name}_metadata.csv",
                processed_text_file=data_dir / f"{name}_processed_text.csv",
            )
        )

    # because the data does not come as bag of words, create this
    # representation here
    # collect all unique words from training and testing data
    words = set().union(
        *[set(doc) for sub_data in data for doc in sub_data.processed_texts]
    )
    bow_data = [BoW_Data.from_processed_data(x, words) for x in data]

    return Split_Data(*bow_data)


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
