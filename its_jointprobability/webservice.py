"""The webservice that allows for interaction with the Bayesian model."""
import argparse
from enum import Enum
from pathlib import Path

import pyro.ops.stats
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

import its_jointprobability.models.prodslda as model_module
from its_jointprobability._version import __version__
from its_jointprobability.models.model import Simple_Model


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
    model, metadata = model_module.import_model(data_dir, device=device)

    # collect the possible discipline values in an Enum
    Disciplines_Enum = Enum(
        "Disciplines_Enum",
        dict((discipline, discipline) for discipline in metadata.titles),
        type=str,
    )
    Disciplines_Enum_URI = Enum(
        "Disciplines_URI_Enum",
        dict((uri, uri) for uri in metadata.uris),
        type=str,
    )

    disciplines = [Disciplines_Enum(disc) for disc in metadata.titles]
    uris = [Disciplines_Enum_URI(uri) for uri in metadata.uris]

    class Prediction_Data(BaseModel):
        """Input to be used for prediction."""

        text: str
        num_samples: int = Field(
            default=100 if not debug else 2, gt=1, le=100000 if not debug else 10
        )
        num_predictions: int = Field(default=len(uris), gt=0, le=len(uris))
        interval_size: float = Field(default=0.8, gt=0.0, lt=1.0)

    # classes that define interfaces for the API
    class Discipline(BaseModel):
        """An individual prediction for a particular school discipline."""

        id: Disciplines_Enum_URI
        name: Disciplines_Enum
        mean_prob: float
        median_prob: float
        prob_interval: list[float] = Field(min_items=2, max_items=2)

    class Prediction_Result(BaseModel):
        """The output of the prediction."""

        disciplines: list[Discipline]
        version: str = __version__

    class Webservice:
        """The actual web service."""

        def __init__(self, model: Simple_Model, token_dict: dict[int, str]) -> None:
            self.model = model
            self.token_dict = token_dict

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
                        id=uri,
                        name=label,
                        mean_prob=float(mean_prob),
                        median_prob=float(median_prob),
                        prob_interval=interval,
                    )
                    for label, uri, mean_prob, median_prob, interval in zip(
                        Disciplines_Enum,
                        Disciplines_Enum_URI,
                        mean_probs,
                        median_probs,
                        intervals,
                    )
                ],
                key=lambda x: x.median_prob,
                reverse=True,
            )[: inp.num_predictions]

            return Prediction_Result(disciplines=disciplines)

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

    webservice = Webservice(model, metadata.word_id_meanings)
    app = webservice.app

    print(f"running on device {model.device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
