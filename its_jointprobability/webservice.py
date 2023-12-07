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
        help="Whether to enable debug mode (more constrained allowed values)",
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
    model, dictionary, disciplines = import_data(data_dir)

    # collect the possible discipline values in an Enum
    Disciplines_Enum = Enum(
        "Disciplines_Enum",
        dict((discipline, discipline) for discipline in disciplines),
        type=str,
    )

    # classes that define interfaces for the API
    class Discipline(BaseModel):
        name: Disciplines_Enum
        mean_prob: float
        median_prob: float
        prob_interval: list[float] = Field(min_items=2, max_items=2)

    class Prediction_Data(BaseModel):
        text: str
        num_samples: int = Field(
            default=100 if not debug else 2, gt=1, le=1000 if not debug else 10
        )
        interval_size: float = Field(default=0.8, gt=0.0, lt=1.0)

    class Prediction_Result(BaseModel):
        disciplines: list[Discipline]
        version: str = __version__

    class Update_Input(BaseModel):
        text: str
        classification: list[Disciplines_Enum]
        learning_rate: float = Field(default=1.0, le=1.0, gt=0.0)
        gamma: float = Field(default=0.001, le=1.0, gt=0.0)
        num_repeats: int = Field(default=10, gt=0, le=1000 if not debug else 10)
        num_train_iterations: int = Field(
            default=100 if not debug else 1, gt=0, le=1000 if not debug else 50
        )
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

            app.get(
                "/disciplines",
                summary="The list of all supported disciplines.",
                description="""
                
                See <https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.html> for a list of all disciplines
                """,
            )(self.disciplines)
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
                    The school disciplines that fit to the given text.
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

            return app

    webservice = Webservice(model, dictionary, disciplines)
    app = webservice.app

    print(f"running on device {device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
