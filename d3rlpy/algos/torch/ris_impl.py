import math
from typing import Optional, Sequence, cast

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_discrete_imitator
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import compute_max_with_n_actions
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .ddpg_impl import DDPGBaseImpl
from .dqn_impl import DoubleDQNImpl


class RISImpl(DDPGBaseImpl):

    _imitator_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _lam: float
    _policy: Optional[DeterministicResidualPolicy]
    _targ_policy: Optional[DeterministicResidualPolicy]
    _imitator: Optional[Imitator]
    _imitator_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        lam: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._n_critics = n_critics
        self._lam = lam

        # initialized in build
        self._imitator = None
        self._imitator_optim = None

    def build(self) -> None:
        self._build_imitator()
        super().build()
        # setup optimizer after the parameters move to GPU
        self._build_imitator_optim()

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-6.0,
            max_logstd=0.0,
            use_std_parameter=True,
        )

    def _build_imitator(self) -> None:
        self._imitator = create_probablistic_regressor(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
                min_logstd=-4.0,
                max_logstd=15.0,
            )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None

        # compute log probability
        policy_dist = self._policy.dist(batch.observations)
        policy_log_probs = policy_dist.log_prob(batch.actions)

        # compute imitator log probability
        imitator_dist = self._imitator.dist(batch.observations)
        imitator_log_probs = imitator_dist.log_prob(batch.actions)

        # compute q values
        q_values = self._q_func(batch.observations, batch.actions, "min")

        return -((policy_log_probs-imitator_log_probs) * q_values).sum()

    @train_api
    @torch_api()
    def update_imitator(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._imitator_optim is not None
        assert self._imitator is not None

        self._imitator_optim.zero_grad()

        loss = self._imitator.compute_error(batch.observations, batch.actions)

        loss.backward()
        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            repeated_x = self._repeat_observation(batch.next_observations)
            actions = self._sample_repeated_action(repeated_x, True)

            values = compute_max_with_n_actions(
                batch.next_observations, actions, self._targ_q_func, self._lam
            )

            return values
