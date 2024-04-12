import argparse
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.nn
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from its_data.default_pipelines.data import (
    BoW_Data,
    balanced_split,
    publish,
    subset_data_points,
)
from its_data.defaults import Fields
from its_jointprobability.data import (
    Split_Data,
    import_data,
    load_model,
    make_data,
    save_model,
)
from its_jointprobability.models.model import Model, eval_samples
from its_jointprobability.utils import (
    Data_Loader,
    default_data_loader,
    sequential_data_loader,
)
from torch.distributions import constraints


class Parameters(NamedTuple):
    # parameters for the per-document topic mixture
    logtheta: tuple[torch.Tensor, torch.Tensor]
    # parameters for the global relationship between topics and target
    # categories
    nu: tuple[torch.Tensor, torch.Tensor]
    # the scale of the per-document target category applicability
    a_scale: torch.Tensor


class ProdSLDA(Model):
    """
    A modification of the ProdLDA model to support semi-supervised
    multi-assignment classification for arbitrarily many fields.
    """

    def __init__(
        self,
        # information about the inputs
        vocab: Sequence[str],
        target_names: Sequence[str],
        id_label_dicts: Sequence[dict[str, str]],
        # model settings
        num_topics: int = 500,
        nu_loc: float = -6.7,
        nu_scale: float = 0.85,
        mle_priors: bool = True,
        # variational auto-encoder settings
        hid_size: int = 1000,
        hid_num: int = 1,
        hid_size_factor: float = 0.5,
        dropout: float = 0.6,
        use_batch_normalization: bool = True,
        affine_batch_normalization: bool = False,
        # the torch device to run on
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        :param vocab: The ordered list of words that are contained within the
            bag-of-words representations. "Ordered" refers to the fact that the
            in-th entry in this sequence corresponds to the n-th column of the
            bag-of-words representations.
            We require this information in order to do predictions, as texts
            have to be mapped back to comparable bag-of-words representations.
        :param target_names: The names of the various fields that are being
            learned and predicted.
        :param id_label_dicts: The maps from id's to human readable labels for
            values of the various fields. Note that the order of these
            dictionaries must match the order of the fields in
            ``target_names``. While this information is not directly used in
            the model, at least the number of categories within each field must
            correspond to the number of entries within each dictionary.
        :param num_topics: Hyperparameter determining the number of latent
            topics to use for representing the bag-of-words documents as
            vectors. The default value was chosen through hyperparameter
            optimization, though it could, given sufficient data, be increased
            arbitrarily.
        :param nu_loc: The (initial) prior mean for the applicability
            coefficient that links topics to targets. The default value was
            chosen through hyperparameter optimization.
        :param nu_scale: The (initial) prior standard deviation for the
            applicability coefficient that links topics to targets. The default
            value was chosen through hyperparameter optimization.
        :param mle_priors: Whether to adjust the priors during optimization to
            match their corresponding maximum likelihood estimators.
            Note that setting this to True results in a different prior
            distribution for each category. The default value was chosen
            through hyperparameter optimization.
            Additionally, setting this to True makes the choice of ``nu_loc``
            and ``nu_scale`` largely irrelevant.
        :param hid_size: The size of the first hidden layer of the encoder and
            decoder neural networks.
            Note that this may be set arbitrarily high, without risking
            overfitting, as the auto-encoder is only responsible for estimating
            the variational parameters, NOT the actual predictions.
        :param hid_num: The number of hidden layers in the encoder neural
            network. Note that each subsequent hidden layer's size is
            multiplied with ``hid_size_factor``. The default was chosen through
            hyperparameter optimization.
        :param hid_size_factor: The factor to multiply the size of each
            subsequent hidden layer{s size with. The default was chosen
            arbitrarily.
        :param dropout: The dropout rate to use. This helps in combating
            component collapse in the neural networks. The default was chosen
            through hyperparameter optimization.
        :param use_batch_normalization: Whether to apply batch normalization to
            the results of the neural networks. This helps in combating
            component collapse in the neural networks. The default was chosen
            through hyperparameter optimization.
        :param affine_batch_normalization: Whether to use affine batch
            normalization, which increases the number of learnable parameters
            and expressiveness of the neural networks. The default was chosen
            through hyperparameter optimization.
        :param device: The PyTorch device to run this model on.
        """
        # save the given arguments so that they can be exported later.
        # this allows us to not have to export the entire model object,
        # but only its state and initialization parameters.
        self.args = locals().copy()
        del self.args["self"]
        del self.args["device"]
        del self.args["__class__"]

        # dynamically set the return sites
        vocab_size = len(vocab)
        target_sizes = [len(id_label_dict) for id_label_dict in id_label_dicts]

        self.return_sites = tuple(
            [f"target_{label}" for label in target_names]
            + [f"probs_{label}" for label in target_names]
            + ["nu", "a"]
        )
        self.return_site_cat_dim = (
            {f"target_{label}": -2 for label in target_names}
            | {f"probs_{label}": -2 for label in target_names}
            | {"nu": -4, "a": -2}
        )
        self.prediction_sites = {
            site[6:]: site for site in self.return_sites if "probs_" == site[:6]
        }
        self.baseline_data = [torch.zeros([vocab_size], device=device)]

        super().__init__()

        self.vocab = vocab
        self.vocab_size = vocab_size
        self.id_label_dicts = id_label_dicts
        self.target_names = list(target_names)
        self.target_sizes = target_sizes
        self.num_topics = num_topics
        self.hid_size = hid_size
        self.hid_num = hid_num
        self.hid_size_factor = hid_size_factor
        self.dropout = dropout
        self.nu_loc = float(nu_loc)
        self.nu_scale = float(nu_scale)
        self.mle_priors = mle_priors
        self.device = device

        # define the variational auto-encoder networks.

        # define the decoder, which takes topic mixtures
        # and returns word probabilities.
        self.decoder = nn.Sequential(
            nn.Linear(num_topics, hid_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, vocab_size),
        )
        if use_batch_normalization:
            self.decoder.append(
                nn.BatchNorm1d(
                    vocab_size,
                    affine=affine_batch_normalization,
                    track_running_stats=True,
                )
            )
        self.decoder.append(nn.Softmax(-1))

        # define the encoder, which takes word probabilities
        # and returns parameters for the distribution of topics
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hid_size),
        )
        cur_hid_size, prev_hid_size = (
            math.ceil(hid_size * self.hid_size_factor),
            hid_size,
        )
        for _ in range(hid_num - 1):
            self.encoder.append(nn.Tanh())
            self.encoder.append(nn.Linear(prev_hid_size, cur_hid_size))
            prev_hid_size = cur_hid_size
            cur_hid_size = math.ceil(prev_hid_size * self.hid_size_factor)
        self.encoder.append(nn.Tanh())
        self.encoder.append(nn.Dropout(dropout))
        self.encoder.append(nn.Linear(prev_hid_size, num_topics * 2))
        if use_batch_normalization:
            self.encoder.append(
                nn.BatchNorm1d(
                    num_topics * 2,
                    affine=affine_batch_normalization,
                    track_running_stats=True,
                )
            )

        self.to(device)

    def model(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor | None,
        obs_masks: Optional[Sequence[torch.Tensor | None]] = None,
        **kwargs,
    ):
        model_params = self.model_params(docs)
        nu = self.nu_dist(*model_params.nu)

        with pyro.plate("documents_plate", docs.shape[-2], dim=-1):
            theta = self.theta_dist(*model_params.logtheta)

            # draw from the prior distribution on the document contents, which
            # depend on the particular per-document topic mixture
            self.docs_dist(obs=docs, theta=theta)

            target_probs = self.target_probs_dist(
                theta=theta, nu=nu, a_scale=model_params.a_scale
            )

            # draw from the prior distribution on the targets
            self.targets_dist(
                *targets,
                target_probs=target_probs,
                obs_masks=obs_masks,
            )

        # log the drawn target probabilities
        for target_name, target_probs_local in zip(self.target_names, target_probs):
            pyro.deterministic(f"probs_{target_name}", target_probs_local)

    def guide(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor | None,
        obs_masks: Optional[Sequence[torch.Tensor | None]] = None,
    ):
        guide_params = self.guide_params(docs)

        nu_q = self.nu_dist(*guide_params.nu)

        with pyro.plate("documents_plate", docs.shape[-2], dim=-1):
            theta_q = self.theta_dist(*guide_params.logtheta)
            target_probs_q = self.target_probs_dist(
                theta=theta_q, nu=nu_q, a_scale=guide_params.a_scale
            )

            # draw from the variational distribution of the targets for
            # unobserved data
            targets_q = self.targets_dist(
                target_probs=target_probs_q,
                suffix="_unobserved",
            )

    def model_params(self, docs: torch.Tensor) -> Parameters:
        n = sum(self.target_sizes)

        nu_loc_fun = lambda: self.nu_loc * docs.new_zeros([self.num_topics, n])
        nu_scale_fun = lambda: self.nu_scale * docs.new_ones([self.num_topics, n])

        nu_loc = pyro.param("nu_loc", nu_loc_fun) if self.mle_priors else nu_loc_fun()
        nu_scale = (
            pyro.param("nu_scale", nu_scale_fun, constraint=constraints.positive)
            if self.mle_priors
            else nu_scale_fun()
        )

        # the prior on the topic mixture distribution is constant between
        # documents
        logtheta_loc_fun = lambda: docs.new_zeros(self.num_topics)
        logtheta_scale_fun = lambda: docs.new_ones(self.num_topics)

        logtheta_loc = (
            pyro.param("logtheta_loc", logtheta_loc_fun)
            if self.mle_priors
            else logtheta_loc_fun()
        )
        logtheta_scale = (
            pyro.param(
                "logtheta_scale",
                logtheta_scale_fun,
                constraint=constraints.positive,
            )
            if self.mle_priors
            else logtheta_scale_fun()
        )

        # the prior on the scale of the category applicability distribution is
        # constant between documents
        a_scale_fun = lambda: docs.new_ones([n])
        a_scale = (
            pyro.param("a_scale", a_scale_fun, constraint=constraints.positive)
            if self.mle_priors
            else a_scale_fun()
        )

        return Parameters(
            logtheta=(logtheta_loc, logtheta_scale),
            nu=(nu_loc, nu_scale),
            a_scale=a_scale,
        )

    def guide_params(self, docs) -> Parameters:
        n = sum(self.target_sizes)

        # use random initialization for the variational parameters of nu
        nu_q_loc = pyro.param(
            "nu_q_loc", lambda: torch.randn([self.num_topics, n], device=docs.device)
        )
        nu_q_scale = pyro.param(
            "nu_q_scale",
            lambda: torch.randn([self.num_topics, n], device=docs.device).abs() + 1e-3,
            constraint=constraints.positive,
        )

        # the variational parameters for (log)theta are given by the
        # variational autoencoder
        logtheta_q_params = self.logtheta_params(docs)

        # use random initialization for the variational scale of the category
        # applicability
        a_q_scale = pyro.param(
            "a_q_scale",
            lambda: torch.randn([n], device=docs.device).abs() + 1e-3,
            constraint=constraints.positive,
        )

        return Parameters(
            logtheta=logtheta_q_params, nu=(nu_q_loc, nu_q_scale), a_scale=a_q_scale
        )

    def docs_dist(self, obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # get each document's word distribution from the decoder.
        count_param = self.count_param(theta)

        # the distribution of the actual document contents.
        # Currently, PyTorch Multinomial requires `total_count` to be
        # homogeneous. Because the numbers of words across documents can
        # vary, we will use the maximum count accross documents here. This
        # does not affect the result, because Multinomial.log_prob does not
        # require `total_count` to evaluate the log probability.
        total_count = int(obs.sum(-1).max())

        return pyro.sample(
            "docs",
            dist.Multinomial(total_count, count_param),
            obs=obs,
        )

    def nu_dist(self, nu_loc: torch.Tensor, nu_scale: torch.Tensor) -> torch.Tensor:
        """
        nu is the matrix mapping the relationship between latent topic and
        targets
        """
        # this being a latent random variable, apply KL annealing
        with pyro.poutine.scale(scale=self.annealing_factor):
            nu = pyro.sample("nu", dist.Normal(nu_loc, nu_scale).to_event(2))

        if len(nu.shape) > 2 and nu.shape[-3] == 1:
            nu.squeeze_(-3)

        return nu

    def theta_dist(
        self, logtheta_loc: torch.Tensor, logtheta_scale: torch.Tensor
    ) -> torch.Tensor:
        """theta is each document's topic applicability"""
        # this being a latent random variable, apply KL annealing
        with pyro.poutine.scale(scale=self.annealing_factor):
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
            )

        return F.softmax(logtheta, -1)

    def target_probs_dist(
        self, theta: torch.Tensor, nu: torch.Tensor, a_scale: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        return the randomly drawn probabilities of target categories applying.

        Note that for convenience, these probabilities are already grouped by
        the targets, such that res[i].shape == [self.target_sizes[i]]
        for all i.
        """
        # this being a latent random variable, apply KL annealing
        with pyro.poutine.scale(scale=self.annealing_factor):
            a = pyro.sample(
                "a",
                dist.Normal(torch.matmul(theta, nu), a_scale).to_event(1),
            )
            if len(a.shape) > 2 and a.shape[-3] == 1:
                a.squeeze_(-3)

        return [
            F.sigmoid(a_local)
            for _, a_local in enumerate(a.split(self.target_sizes, -1))
        ]

    def targets_dist(
        self,
        *targets: torch.Tensor | None,
        target_probs: Iterable[torch.Tensor],
        obs_masks: Optional[Sequence[torch.Tensor | None]] = None,
        suffix: str = "",
    ) -> list[torch.Tensor]:
        # if no observations mask has been given, ignore any docs that
        # do not have any assigned labels for that given target
        # or any non-assigned labels (depending on the model's settings)
        if obs_masks is None:
            obs_masks = self._get_obs_mask(*targets)

        drawn_targets: list[torch.Tensor] = list()

        for i, target_probs_i in enumerate(target_probs):
            with pyro.plate(f"target_{i}_plate"):
                obs_i = targets[i] if len(targets) > i else None
                obs_masks_i = obs_masks[i] if len(obs_masks) > i else None

                drawn_targets.append(
                    pyro.sample(
                        f"target_{self.target_names[i]}{suffix}",
                        dist.Bernoulli(target_probs_i.swapaxes(-1, -2)),  # type: ignore
                        infer={"enumerate": "parallel"},
                        obs_mask=obs_masks_i,  # type: ignore
                        obs=obs_i.swapaxes(-1, -2) if obs_i is not None else None,
                    ).swapaxes(-1, -2)
                )

        return drawn_targets

    def count_param(self, theta: torch.Tensor) -> torch.Tensor:
        pyro.module("decoder", self.decoder)
        # if we have more than one batch dimension, combine them all into one,
        # as we can only work with 2D inputs in the neural networks
        theta_shape = theta.shape
        if len(theta_shape) > 2:
            batch_size = int(np.prod(theta_shape[:-1]))
            theta = theta.reshape(batch_size, theta_shape[-1])

        count_param = self.decoder(theta)

        # convert the outputs of the neural network back to the batch
        # dimensions we expect (if necessary)
        if len(theta_shape) > 2:
            count_param = count_param.reshape(*theta_shape[:-1], count_param.shape[-1])

        return count_param

    def logtheta_params(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pyro.module("encoder", self.encoder)

        # if we have more than one batch dimension, combine them all into one,
        # as we can only work with 2D inputs in the neural networks
        doc_shape = doc.shape
        if len(doc_shape) > 2:
            batch_size = int(np.prod(doc_shape[:-1]))
            doc = doc.reshape(batch_size, doc_shape[-1])

        logtheta_loc, logtheta_logvar = self.encoder(doc).split(self.num_topics, -1)
        logtheta_scale = F.softplus(logtheta_logvar) + 1e-7

        # convert the outputs of the neural network back to the batch
        # dimensions we expect (if necessary)
        if len(doc_shape) > 2:
            logtheta_loc = logtheta_loc.reshape(*doc_shape[:-1], logtheta_loc.shape[-1])
            logtheta_scale = logtheta_scale.reshape(
                *doc_shape[:-1], logtheta_scale.shape[-1]
            )

        return logtheta_loc, logtheta_scale

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # the following sites have a leading dummy-dimension during posterior
        # sampling. drop them before returning the samples
        if "nu" in posterior_samples:
            posterior_samples["nu"].squeeze_(-3)

        for prediction_site in self.prediction_sites.values():
            if prediction_site in posterior_samples:
                posterior_samples[prediction_site].squeeze_(-3)

        return posterior_samples

    def _get_obs_mask(self, *targets: torch.Tensor | None) -> list[torch.Tensor | None]:
        return [
            target.sum(-1) > 0 if target is not None else None for target in targets
        ]

    def draw_posterior_probs_samples(
        self, *data: torch.Tensor, num_samples: int = 100
    ) -> dict[str, torch.Tensor]:
        """
        Return samples of the probabilities for each predicted target's
        categories.

        Note that the returned tensors are on the CPU, regardless of the
        primary device used.
        """
        samples = self.draw_posterior_samples(
            data_loader=sequential_data_loader(
                *data,
                device=self.device,
                batch_size=512,
                dtype=torch.float,
            ),
            return_sites=[site for site in self.return_sites if "probs_" == site[:6]],
            num_samples=num_samples,
        )
        # re-key the samples such that only the target names are left
        samples = {key[6:]: value for key, value in samples.items()}

        return samples


class Torch_Data(NamedTuple):
    train_docs: torch.Tensor
    train_targets: dict[str, torch.Tensor]
    test_docs: torch.Tensor
    test_targets: dict[str, torch.Tensor]


def set_up_data(data: Split_Data) -> Torch_Data:
    def to_tensor(data: BoW_Data) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # use the from_numpy function, as this way, the two share memory
        docs: torch.Tensor = torch.from_numpy(data.bows)
        targets = {
            key: torch.from_numpy(value.arr) for key, value in data.target_data.items()
        }

        return docs, targets

    return Torch_Data(*to_tensor(data.train), *to_tensor(data.test))


def train_model(
    data_loader: Data_Loader,
    vocab: Sequence[str],
    id_label_dicts: Sequence[dict[str, str]],
    target_names: Sequence[str],
    min_epochs: int = 10,
    max_epochs: int = 1000,
    num_particles: int = 1,
    initial_lr: float = 0.09,
    gamma: float = 0.25,
    betas: tuple[float, float] = (0.3, 0.18),
    seed: int = 0,
    initial_annealing_factor: float = 0.012,
    device: Optional[torch.device] = None,
    **kwargs,
) -> ProdSLDA:
    pyro.set_rng_seed(seed)

    prodslda = ProdSLDA(
        vocab=vocab,
        id_label_dicts=id_label_dicts,
        target_names=target_names,
        device=device,
        **kwargs,
    )

    prodslda.annealing_factor = initial_annealing_factor

    prodslda.run_svi(
        data_loader=data_loader,
        elbo=pyro.infer.TraceEnum_ELBO(
            num_particles=num_particles,
            max_plate_nesting=2,
            vectorize_particles=False,
        ),
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        initial_lr=initial_lr,
        gamma=gamma,
        betas=betas,
    )

    return prodslda.eval()


def retrain_model_cli():
    """Add some CLI arguments to the retraining of the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="The path to the directory containing the training data",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The name to identify the trained model with; used for storing / loading its relevant files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for pseudo random number generation",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="The maximum number of training epochs per batch of data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="The amount of training data to use",
    )
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Whether to also include materials that have not been confirmed editorially.",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Whether to skip the cached train / test data, effectively forcing a re-generation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Whether to skip training and go directly to evaluation.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        help="The amount of available (V)RAM, in MB. Note that this directly affects the batch-size and thus the training duration. If training crashes due to too little memory, reduce this.",
        default=5 * 1024,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print various logs to the stdout during training.",
    )

    args = parser.parse_args()

    path = Path(args.path)

    try:
        if args.skip_cache:
            raise FileNotFoundError()
        data = import_data(path)
    except FileNotFoundError:
        print("Processed data not found. Generating it...")
        data = make_data(path)
        publish(data.train, path, name="train")
        publish(data.test, path, name="test")

    if not args.include_unconfirmed:
        train_data = subset_data_points(data.train, np.where(data.train.editor_arr)[0])
        data = Split_Data(train_data, data.test)

    if args.train_ratio is not None and not args.train_ratio == 1.0:
        train_data, _ = balanced_split(
            data.train, field=Fields.TAXONID.value, ratio=1 - args.train_ratio
        )
        data = Split_Data(train_data, data.test)

    train_docs, train_targets, _, _ = set_up_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the batch size such that the batch should fit into the given memory
    # size.
    batch_size = min(
        len(train_docs),
        int(
            args.memory
            * (1024**2)
            / (
                1.2
                * train_docs.shape[-1]
                * sum(
                    targets.arr.shape[-1] for targets in data.train.target_data.values()
                )
            )
        ),
    )
    data_loader = default_data_loader(
        train_docs,
        *train_targets.values(),
        device=device,
        dtype=torch.float,
        batch_size=batch_size,
    )

    suffix = (
        "_".join(sorted(list(train_targets.keys())))
        if args.model_name is None
        else args.model_name
    )

    if not args.eval_only:
        prodslda = train_model(
            data_loader,
            vocab=data.train.words.tolist(),
            id_label_dicts=[
                {
                    id: label
                    for id, label in zip(
                        data.train.target_data[field].uris,
                        data.train.target_data[field].labels,
                    )
                }
                for field in train_targets.keys()
            ],
            target_names=list(train_targets.keys()),
            device=device,
            seed=args.seed,
            max_epochs=args.max_epochs,
        )

        save_model(prodslda, path, suffix=suffix)
    else:
        # try:
        prodslda = load_model(ProdSLDA, path, device, suffix=suffix)
        # except FileNotFoundError:
        #     prodslda = load_model(ProdSLDA, path, device)

    eval_sites = {key: f"probs_{key}" for key in train_targets.keys()}

    run_evaluation(prodslda, data, eval_sites)


def run_evaluation(model: ProdSLDA, data: Split_Data, eval_sites: dict[str, str]):
    train_docs, train_targets, test_docs, test_targets = set_up_data(data)
    titles = {key: value.labels for key, value in data.train.target_data.items()}

    # evaluate the newly trained model
    # print()
    # print("------------------------------")
    # print("evaluating model on train data")
    # results = eval_samples(
    #     target_samples=model.draw_posterior_probs_samples(train_docs, num_samples=50),
    #     targets=train_targets,
    #     target_values=titles,
    #     cutoffs=None,
    #     # cutoff_compute_method="base-rate",
    # )
    # cutoffs = {key: result.cutoff for key, result in results.items()}

    if len(test_docs) > 0:
        print()
        print("-----------------------------")
        print("evaluating model on test data")
        eval_samples(
            target_samples=model.draw_posterior_probs_samples(
                test_docs, num_samples=150
            ),
            targets=test_targets,
            target_values=titles,
            cutoffs=None,
        )
