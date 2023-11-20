import argparse
from collections.abc import Sequence
from pathlib import Path

import pyro.ops.stats
import torch
from torch.nn.modules import loss
import uvicorn
from fastapi import FastAPI
from icecream import ic
from pydantic import BaseModel

from its_jointprobability._version import __version__
from its_jointprobability.models.prodslda_sep import Classification, import_data
from its_jointprobability.utils import labels_to_tensor, texts_to_bow_tensor, device


class Prediction_Data(BaseModel):
    text: str
    num_samples: int = 100
    interval_size: float = 0.8


class Update_Input(BaseModel):
    text: str
    classification: list[str]
    learning_rate: float = 1
    gamma: float = 0.001
    num_repeats: int = 10
    num_train_iterations: int = 250
    num_losses_head: int = 2
    num_losses_tail: int = 2


class Update_Output(BaseModel):
    losses_head: list[float]
    losses_tail: list[float]
    num_train_iterations: int


class Discipline(BaseModel):
    id: str
    mean_prob: float
    median_prob: float
    prob_interval: tuple[float, float]


class Prediction_Result(BaseModel):
    disciplines: list[Discipline]
    version: str = __version__


class Webservice:
    def __init__(
        self, model: Classification, token_dict: dict[int, str], labels: Sequence[str]
    ) -> None:
        self.model = model
        self.token_dict = token_dict
        self.labels = labels

    def disciplines(self) -> Sequence[str]:
        return self.labels

    def predict_disciplines(self, inp: Prediction_Data) -> Prediction_Result:
        posterior_samples = self.model.draw_posterior_samples_from_texts(
            inp.text,
            token_dict=self.token_dict,
            num_samples=inp.num_samples,
            return_sites=["a"],
        )["a"].squeeze(-2)
        probs = 1 / (1 + torch.exp(-posterior_samples))
        mean_probs = probs.mean(0)
        median_probs = probs.median(0)[0]
        intervals: list[list[float]] = (
            pyro.ops.stats.hpdi(probs, inp.interval_size).squeeze(-1).T
        ).tolist()
        intervals_tuples: list[tuple[float, float]] = [
            tuple(interval) for interval in intervals
        ]  # type: ignore

        disciplines = sorted(
            [
                Discipline(
                    id=label,
                    mean_prob=float(mean_prob),
                    median_prob=float(median_prob),
                    prob_interval=interval,
                )
                for label, mean_prob, median_prob, interval in zip(
                    self.labels, mean_probs, median_probs, intervals_tuples
                )
            ],
            key=lambda x: x.median_prob,
            reverse=True,
        )

        return Prediction_Result(disciplines=disciplines)

    def update_model(self, inp: Update_Input) -> Update_Output:
        texts = [inp.text for _ in range(inp.num_repeats)]
        labels = [inp.classification for _ in range(inp.num_repeats)]

        bows_tensor = texts_to_bow_tensor(*texts, token_dict=self.token_dict)
        labels_tensor = labels_to_tensor(*labels, label_values=self.labels)

        losses = self.model.bayesian_update(
            docs=bows_tensor,
            labels=labels_tensor,
            num_particles=10,
            min_epochs=inp.num_train_iterations // 10,
            max_epochs=inp.num_train_iterations,
            initial_lr=inp.learning_rate,
        )

        losses_head = losses[-inp.num_losses_head :]
        losses_tail = losses[: inp.num_losses_tail]

        return Update_Output(
            losses_head=losses_head,
            losses_tail=losses_tail,
            num_train_iterations=len(losses),
        )

    @property
    def app(self) -> FastAPI:
        app = FastAPI()

        app.get("/disciplines")(self.disciplines)
        app.post("/predict_disciplines")(self.predict_disciplines)
        app.post("/update_model")(self.update_model)

        return app


def main():
    # define CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default=8080, help="Port to listen on", type=int
    )
    parser.add_argument(
        "--host", action="store", default="0.0.0.0", help="Hosts to listen on", type=str
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    # read passed CLI arguments
    args = parser.parse_args()

    # import the model and auxiliary data
    data_dir = Path.cwd() / "data"
    model, dictionary, labels = import_data(data_dir)

    webservice = Webservice(model, dictionary, labels)
    app = webservice.app

    print(f"running on device {device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
