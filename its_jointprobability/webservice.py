import argparse
from collections.abc import Sequence
from pathlib import Path

import pyro.ops.stats
import torch
from torch.nn.modules import loss
import uvicorn
from fastapi import FastAPI
from icecream import ic
from pydantic import BaseModel, Field
from enum import Enum

from its_jointprobability._version import __version__
from its_jointprobability.models.prodslda_sep import Classification, import_data
from its_jointprobability.utils import labels_to_tensor, texts_to_bow_tensor, device


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
    model, dictionary, disciplines = import_data(data_dir)

    Disciplines_Enum = Enum(
        "Disciplines_Enum",
        dict((discipline, discipline) for discipline in disciplines),
        type=str,
    )

    class Discipline(BaseModel):
        name: Disciplines_Enum
        mean_prob: float
        median_prob: float
        prob_interval: list[float] = Field(min_items=2, max_items=2)

    class Prediction_Data(BaseModel):
        text: str
        num_samples: int = Field(default=100, gt=1, le=1000)
        interval_size: float = Field(default=0.8, gt=0.0, lt=1.0)

    class Prediction_Result(BaseModel):
        disciplines: list[Discipline]
        version: str = __version__

    class Update_Input(BaseModel):
        text: str
        classification: list[Disciplines_Enum]
        learning_rate: float = Field(default=1.0, le=1.0, gt=0.0)
        gamma: float = Field(default=0.001, le=1.0, gt=0.0)
        num_repeats: int = Field(default=10, gt=0, le=1000)
        num_train_iterations: int = Field(default=100, gt=0, le=1000)
        num_losses_head: int = Field(default=2, gt=0)
        num_losses_tail: int = Field(default=2, gt=0)

    class Update_Output(BaseModel):
        losses_head: list[float]
        losses_tail: list[float]
        num_train_iterations: int

    class Webservice:
        def __init__(
            self,
            model: Classification,
            token_dict: dict[int, str],
            labels: Sequence[str],
        ) -> None:
            self.model = model
            self.token_dict = token_dict
            self.labels = labels

        def disciplines(self) -> Sequence[str]:
            return self.labels

        def predict_disciplines(self, inp: Prediction_Data) -> Prediction_Result:
            try:
                posterior_samples = self.model.draw_posterior_samples_from_texts(
                    inp.text,
                    token_dict=self.token_dict,
                    num_samples=inp.num_samples,
                    return_sites=["a"],
                )["a"].squeeze(-2)
            except RuntimeError:
                return Prediction_Result(disciplines=[])

            probs = 1 / (1 + torch.exp(-posterior_samples))
            mean_probs = probs.mean(0)
            median_probs = probs.median(0)[0]
            intervals: list[list[float]] = (
                pyro.ops.stats.hpdi(probs, inp.interval_size).squeeze(-1).T
            ).tolist()

            disciplines = sorted(
                [
                    Discipline(
                        name=label,
                        mean_prob=float(mean_prob),
                        median_prob=float(median_prob),
                        prob_interval=interval,
                    )
                    for label, mean_prob, median_prob, interval in zip(
                        Disciplines_Enum, mean_probs, median_probs, intervals
                    )
                ],
                key=lambda x: x.median_prob,
                reverse=True,
            )

            return Prediction_Result(disciplines=disciplines)

        def update_model(self, inp: Update_Input) -> Update_Output:
            texts = [inp.text for _ in range(inp.num_repeats)]
            # ignore multiple identical labels
            labels = [list(set(inp.classification)) for _ in range(inp.num_repeats)]

            try:
                bows_tensor = texts_to_bow_tensor(*texts, token_dict=self.token_dict)
                labels_tensor = labels_to_tensor(
                    *labels, label_values=[e for e in Disciplines_Enum]
                )
            except RuntimeError:
                return Update_Output(
                    losses_head=[], losses_tail=[], num_train_iterations=0
                )

            losses = self.model.bayesian_update(
                docs=bows_tensor,
                labels=labels_tensor,
                num_particles=10,
                min_epochs=inp.num_train_iterations // 10,
                max_epochs=inp.num_train_iterations,
                initial_lr=inp.learning_rate,
            )

            losses_tail = losses[-inp.num_losses_tail :]
            losses_head = losses[: inp.num_losses_head]

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

    webservice = Webservice(model, dictionary, disciplines)
    app = webservice.app

    print(f"running on device {device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
