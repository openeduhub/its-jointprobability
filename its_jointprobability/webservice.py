"""The webservice that allows for interaction with the Bayesian model."""
import argparse
from collections.abc import Sequence
from pathlib import Path

import pyro.ops.stats
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from icecream import ic
from pydantic import BaseModel, Field

from its_jointprobability._version import __version__
from its_jointprobability.data import load_model
from its_jointprobability.models.model import Simple_Model
from its_jointprobability.models.prodslda import ProdSLDA


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
        "--debug",
        action="store_true",
        help="Whether to enable debug mode (more constrained allowed values). Primarily useful for automated testing.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    # read passed CLI arguments
    args = parser.parse_args()
    debug: bool = args.debug

    # import the model and auxiliary data
    data_dir = Path.cwd() / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ProdSLDA, data_dir, device)

    class Prediction_Data(BaseModel):
        """Input to be used for prediction."""

        text: str
        num_samples: int = Field(
            default=100 if not debug else 2, gt=1, le=100000 if not debug else 10
        )
        num_predictions: int = Field(default=10, gt=0)
        interval_size: float = Field(default=0.8, gt=0.0, lt=1.0)

    # classes that define interfaces for the API
    class Prediction_Score(BaseModel):
        """An individual prediction for a particular school discipline."""

        id: str
        name: str
        mean_prob: float
        median_prob: float
        prob_interval: list[float] = Field(min_items=2, max_items=2)

    class Prediction_Result(BaseModel):
        """The output of the prediction."""

        predictions: dict[str, list[Prediction_Score]]
        version: str = __version__

    class Webservice:
        """The actual web service."""

        def __init__(self, model: ProdSLDA, token_dict: Sequence[str]) -> None:
            self.model = model
            self.token_dict = token_dict

        def predict_disciplines(self, inp: Prediction_Data) -> Prediction_Result:
            ic.disable()
            try:
                posterior_samples_by_field = (
                    self.model.draw_posterior_samples_from_texts(
                        inp.text,
                        tokens=self.token_dict,
                        num_samples=inp.num_samples,
                        return_sites=["a"],
                    )["a"].split(self.model.target_sizes, -1)
                )
            except RuntimeError:
                return Prediction_Result(predictions=dict())

            ic.enable()

            predictions = dict()
            for field, posterior_samples, id_label_dict in zip(
                model.target_names,
                posterior_samples_by_field,
                model.id_label_dicts,
            ):
                probs = F.sigmoid(posterior_samples.squeeze(-2).squeeze(-2))
                mean_probs = probs.mean(0)
                median_probs = probs.median(0)[0]
                intervals: list[list[float]] = (
                    pyro.ops.stats.hpdi(probs, inp.interval_size).squeeze(-1).T
                ).tolist()

                prediction = sorted(
                    [
                        Prediction_Score(
                            id=uri,
                            name=label,
                            mean_prob=float(mean_prob),
                            median_prob=float(median_prob),
                            prob_interval=interval,
                        )
                        for label, uri, mean_prob, median_prob, interval in zip(
                            id_label_dict.values(),
                            id_label_dict.keys(),
                            mean_probs,
                            median_probs,
                            intervals,
                        )
                    ],
                    key=lambda x: x.median_prob,
                    reverse=True,
                )
                predictions[field] = prediction[
                    : min(len(prediction), inp.num_predictions)
                ]

            return Prediction_Result(predictions=predictions)

        @property
        def app(self) -> FastAPI:
            app = FastAPI()

            @app.get("/_ping")
            def _ping():
                pass

            app.post(
                "/predict_disciplines",
                summary="Predict the disciplines belonging to the given text.",
                description="""
                Note that all disciplines will be returned, sorted by their
                median predicted probability of being relevant to the text.
                
                Parameters
                ----------
                text : str
                    The text to be analyzed.
                num_samples : int
                    The number of samples to use in order to estimate the
                    fit of each discipline.
                    Higher numbers will result in less variance between calls,
                    but take more time.
                num_predictions : int
                    The number of predicted disciplines (sorted by relevance)
                    to return.
                interval_size : float (0, 1]
                    The size of the credibility interval for the probability
                    that a discipline is assigned to the given text.
                    E.g. at 0.8, there is a probability of 80% that the
                    predicted probability of the discipline belonging to the
                    text is within the returned interval.

                Returns
                -------
                disciplines : list of Discipline
                    The list of disciplines, see below.
                version : str
                    The version of the prediction tool.

                Discipline
                ----------
                id : str
                    The URI of the discipline.
                name : str
                    The name of the discipline.
                mean_prob : float [0, 1]
                    The mean of the predicted probabilities that this discipline
                    belongs to the given text.
                median_prob : float [0, 1]
                    The median of the above probabilities.
                prob_interval : 2-tuple of floats in [0, 1]
                    The credibility interval of the predicted probabilities above.
                """,
            )(self.predict_disciplines)

            return app

    webservice = Webservice(model, model.vocab)
    app = webservice.app

    print(f"running on device {model.device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
