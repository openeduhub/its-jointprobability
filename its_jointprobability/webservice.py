"""The webservice that allows for interaction with the Bayesian model."""
import argparse
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import pyro.ops.stats
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from its_jointprobability._version import __version__
from its_jointprobability.models.prodslda_sep import Classification, import_data
from its_jointprobability.utils import device, labels_to_tensor, texts_to_bow_tensor


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
    model, dictionary, disciplines_raw, uris_raw = import_data(data_dir)

    # collect the possible discipline values in an Enum
    Disciplines_Enum = Enum(
        "Disciplines_Enum",
        dict((discipline, discipline) for discipline in disciplines_raw),
        type=str,
    )
    Disciplines_Enum_URI = Enum(
        "Disciplines_URI_Enum",
        dict((uri, uri) for uri in uris_raw),
        type=str,
    )

    disciplines = [Disciplines_Enum(disc) for disc in disciplines_raw]
    uris = [Disciplines_Enum_URI(uri) for uri in uris_raw]

    class Prediction_Data(BaseModel):
        """Input to be used for prediction."""

        text: str
        num_samples: int = Field(
            default=100 if not debug else 2, gt=1, le=1000 if not debug else 10
        )
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

    class Update_Input_URI(BaseModel):
        """Input to be used for updating the model."""

        text: str
        classification: list[Disciplines_Enum_URI]
        learning_rate: float = Field(default=1.0, le=1.0, gt=0.0)
        gamma: float = Field(default=0.001, le=1.0, gt=0.0)
        num_repeats: int = Field(default=10, gt=0, le=1000 if not debug else 10)
        num_train_iterations: int = Field(
            default=250 if not debug else 1,
            ge=200 if not debug else 1,
            le=1000 if not debug else 50,
        )
        num_losses_head: int = Field(default=2, gt=0)
        num_losses_tail: int = Field(default=2, gt=0)

    class Update_Input_Label(BaseModel):
        """Input to be used for updating the model."""

        text: str
        classification: list[Disciplines_Enum]
        learning_rate: float = Field(default=1.0, le=1.0, gt=0.0)
        gamma: float = Field(default=0.001, le=1.0, gt=0.0)
        num_repeats: int = Field(default=10, gt=0, le=1000 if not debug else 10)
        num_train_iterations: int = Field(
            default=250 if not debug else 1,
            ge=200 if not debug else 1,
            le=1000 if not debug else 50,
        )
        num_losses_head: int = Field(default=2, gt=0)
        num_losses_tail: int = Field(default=2, gt=0)

    class Update_Output(BaseModel):
        """Some diagnostics to evaluate how well the update went."""

        losses_head: list[float]
        losses_tail: list[float]
        num_train_iterations: int

    class Webservice:
        """The actual web service."""

        def __init__(self, model: Classification, token_dict: dict[int, str]) -> None:
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
            )

            return Prediction_Result(disciplines=disciplines)

        def update_model(self, inp: Update_Input_URI) -> Update_Output:
            texts = [inp.text for _ in range(inp.num_repeats)]
            # ignore multiple identical labels
            labels = [list(set(inp.classification)) for _ in range(inp.num_repeats)]

            try:
                bows_tensor = texts_to_bow_tensor(*texts, token_dict=self.token_dict)
                labels_tensor = labels_to_tensor(
                    *labels, label_values=[e for e in Disciplines_Enum_URI]
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

        def update_model_label(self, inp: Update_Input_Label) -> Update_Output:
            classification_uris = [
                Disciplines_Enum_URI(uris[disciplines.index(label)])
                for label in inp.classification
            ]

            return self.update_model(
                Update_Input_URI(
                    text=inp.text,
                    classification=classification_uris,
                    learning_rate=inp.learning_rate,
                    gamma=inp.gamma,
                    num_repeats=inp.num_repeats,
                    num_train_iterations=inp.num_train_iterations,
                    num_losses_head=inp.num_losses_head,
                    num_losses_tail=inp.num_losses_tail,
                )
            )

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
            app.post(
                "/update_model",
                summary="Update the model with a text and discipline assignment.",
                description="""
                Note that the given assignment of disciplines for the text is
                assumed to be *fully* correct. This means that all listed
                disciplines are actually relevant to the text and that *all
                of the unlisted disciplines are not relevant*.
                After the update, the model will predict higher probabilities
                for the listed disciplines and lower probabilities for all
                other disciplines.

                Parameters
                ----------
                text : str
                    The text based on which the model is to be updated.
                classification : array of str
                    The URIs of school disciplines that fit to the given text.
                learning_rate : float (0, 1]
                    The speed with which to learn from the data.
                    The lower num_repeats is chosen, the lower this should be.
                gamma : float (0, 1]
                    The rate with which the learning rate will decrease
                    during training. The learning rate for the last iteration
                    will be gamma * learning_rate.
                num_repeats : int > 0
                    The number of times this text will be 'copied' for the update.
                    Higher values will result in stronger changes in the updated
                    model, but may negatively impact the performance on
                    different texts.
                    If the prediction on the text was relatively close to the
                    desired results, choose a lower number. If it is very far
                    off (e.g. almost 0), a higher number like 50 - 100 may be
                    necessary.
                num_train_iterations : int > 0
                    The number of iterations to run the updating process for.
                    Higher values will result in more consistent updates,
                    but will also take a longer amount of time.
                    Note that the update may be stopped earlier if
                    convergence has been detected. The update will always run
                    for at least 0.1 * num_train_iterations steps.
                num_losses_head : int > 0
                    The number of initial losses of the update to return.
                num_losses_tail : in > 0
                    The number of final losses of the update to return.

                Returns
                -------
                losses_head : list of float
                    The first few losses of the update.
                losses_tail : list of float
                    The last few losses of the update.
                    If these values are close to the initial ones,
                    you may need to increase the number of training iterations
                    or the learning rate.
                num_train_iterations : int
                    The number of iterations that the update process was
                    actually run for.
                    If the first and final losses are similar, and this
                    number is lower than the set number of training iterations,
                    the learning rate is very likely to be too low.
                """,
            )(self.update_model)

            app.post(
                "/update_model_label",
                summary="Update the model with a text and discipline assignment.",
                description="Like update_model, but with discipline names instead of URIs.",
            )(self.update_model_label)

            return app

    webservice = Webservice(model, dictionary)
    app = webservice.app

    print(f"running on device {device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
