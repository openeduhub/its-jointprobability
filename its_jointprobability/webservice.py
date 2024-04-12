"""The webservice that allows for interaction with the Bayesian model."""
import argparse
from collections.abc import Sequence
import math
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from its_jointprobability._version import __version__
from its_jointprobability.data import load_model
from its_jointprobability.models.model import Prediction_Score
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
    model = load_model(ProdSLDA, data_dir, device=device)

    class Prediction_Data(BaseModel):
        """Input to be used for prediction."""

        text: str
        num_samples: int = Field(
            default=100 if not debug else 2, gt=1, le=100000 if not debug else 10
        )
        num_predictions: int = Field(default=10, gt=0)
        interval_size: float = Field(default=0.8, gt=0.0, lt=1.0)

    class Prediction_Result(BaseModel):
        """The output of the prediction."""

        predictions: dict[str, list[Prediction_Score]]
        version: str = __version__

    class Webservice:
        """The actual web service."""

        def __init__(self, model: ProdSLDA, tokens: Sequence[str]) -> None:
            self.model = model
            self.tokens = tokens

        @property
        def app(self) -> FastAPI:
            app = FastAPI()

            @app.get("/_ping")
            def _ping():
                pass

            @app.post(
                "/predict",
                summary="Predict the metadata fitting the given text.",
                description="""
                Note that all categories are not filtered out. Instead, they
                are sorted by their mean predicted probability of being
                relevant to the text.
                
                Parameters
                ----------
                text : str
                    The text to be analyzed.
                num_samples : int
                    The number of samples to use in order to estimate the fit
                    of each discipline. Higher numbers will result in less
                    variance between calls, but take more time.
                num_predictions : int
                    The number of predicted disciplines (sorted by relevance)
                    to return. This does not affect performance; it simply
                    serves as an initial filtering tool.
                interval_size : float (0, 1]
                    The size of the credibility interval for the probability
                    that a discipline is assigned to the given text.
                    E.g. at 0.8, there is a probability of 80% that the
                    predicted probability of the discipline belonging to the
                    text is within the returned interval.

                Returns
                -------
                predictions : dict[str, list[Prediction]]
                    Map from predicted metadatum to predictions for this
                    metadatum.
                version : str
                    The version of the prediction tool.

                Prediction
                ----------
                id : str
                    The URI of the category.
                name : str
                    The label of the category.
                mean_prob : float [0, 1]
                    The mean of the predicted probabilities that this category
                    belongs to the given text.
                median_prob : float [0, 1]
                    The median of the above probabilities.
                prob_interval : 2-tuple of floats in [0, 1]
                    The credibility interval of the predicted probabilities
                    above.
                """,
            )
            def predict(inp: Prediction_Data) -> Prediction_Result:
                predictions = next(
                    self.model.predict_from_texts(
                        inp.text,
                        tokens=self.tokens,
                        num_samples=inp.num_samples,
                        interval_size=inp.interval_size,
                    )
                )
                # sort the predictions and only keep the most relevant
                predictions = {
                    key: sorted(
                        value,
                        key=lambda x: x.mean_prob,
                        reverse=True,
                    )[: min(len(value), inp.num_predictions)]
                    for key, value in predictions.items()
                }
                return Prediction_Result(predictions=predictions)

            return app

    webservice = Webservice(model, model.vocab)
    app = webservice.app

    # initialize the baseline distributions by calling prediction on a dummy
    # text.
    # TODO: move this to post model training and save the baseline
    # distributions
    list(
        model.predict_from_texts(
            model.vocab[0] + model.vocab[1], tokens=model.vocab, num_samples=1
        )
    )

    print(f"running on device {model.device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
