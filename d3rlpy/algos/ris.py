from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.ris_impl import RISImpl


class RIS(AlgoBase):
    r"""Regression IS with Q-function.

    References:
        * `Hanna et al., Importance Sampling Policy Evaluation with
        an Estimated Behavior Policy. <https://arxiv.org/abs/1806.01347>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        rl_start_step (int): step to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.ris_impl.RISImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _imitator_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _imitator_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _imitator_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _update_actor_interval: int
    _lam: float
    _rl_start_step: int
    _use_gpu: Optional[Device]
    _impl: Optional[RISImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        imitator_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        imitator_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        imitator_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        update_actor_interval: int = 1,
        lam: float = 0.75,
        rl_start_step: int = 0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[RISImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._imitator_learning_rate = imitator_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._imitator_optim_factory = imitator_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._update_actor_interval = update_actor_interval
        self._lam = lam
        self._rl_start_step = rl_start_step
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = RISImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            imitator_learning_rate=self._imitator_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            imitator_optim_factory=self._imitator_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            imitator_encoder_factory=self._imitator_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            lam=self._lam,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        imitator_loss = self._impl.update_imitator(batch)
        metrics.update({"imitator_loss": imitator_loss})

        if self._grad_step >= self._rl_start_step:
            critic_loss = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})

            if self._grad_step % self._update_actor_interval == 0:
                actor_loss = self._impl.update_actor(batch)
                metrics.update({"actor_loss": actor_loss})
                self._impl.update_actor_target()
                self._impl.update_critic_target()

        return metrics

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """RIS does not support sampling action."""
        raise NotImplementedError("RIS does not support sampling action.")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
