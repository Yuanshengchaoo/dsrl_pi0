"""Utilities for loading and sampling flow-matching priors for QC chunk policies."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import checkpoints

Params = Any
PRNGKey = jax.Array
Observation = Any


@dataclasses.dataclass
class FlowMatchingChunkPriorConfig:
    """Configuration for constructing a flow-matching prior sampler."""

    chunk_shape: Tuple[int, int]
    scale: float = 1.0
    checkpoint_path: Optional[str] = None
    model_def: Optional[Any] = None
    sample_observation: Optional[Observation] = None
    init_rng: Optional[PRNGKey] = None


class FlowMatchingChunkPrior:
    """Sampler that produces noise chunks conditioned on observations."""

    def __init__(
        self,
        chunk_shape: Tuple[int, int],
        *,
        apply_fn: Optional[Callable[..., Any]] = None,
        params: Optional[Params] = None,
        batch_stats: Optional[Any] = None,
        scale: float = 1.0,
    ) -> None:
        self._chunk_shape = tuple(chunk_shape)
        if len(self._chunk_shape) != 2:
            raise ValueError("chunk_shape must be a (chunk_len, action_dim) tuple.")
        self._apply_fn = apply_fn
        self._params = params
        self._batch_stats = batch_stats
        self._scale = scale

        if self._apply_fn is not None and self._params is None:
            raise ValueError("params must be provided when apply_fn is set.")

    @classmethod
    def from_config(cls, config: FlowMatchingChunkPriorConfig) -> "FlowMatchingChunkPrior":
        if config.model_def is None or config.checkpoint_path is None:
            return cls(config.chunk_shape, scale=config.scale)

        if config.sample_observation is None:
            raise ValueError("sample_observation must be provided when loading from a checkpoint.")

        init_rng = config.init_rng or jax.random.PRNGKey(0)
        variables = config.model_def.init(init_rng, config.sample_observation, training=False)
        params = variables.get("params")
        batch_stats = variables.get("batch_stats")

        restored = checkpoints.restore_checkpoint(
            config.checkpoint_path,
            target={"params": params, "batch_stats": batch_stats},
        )
        params = restored.get("params", params)
        batch_stats = restored.get("batch_stats", batch_stats)

        return cls(
            config.chunk_shape,
            apply_fn=config.model_def.apply,
            params=params,
            batch_stats=batch_stats,
            scale=config.scale,
        )

    def _prepare_inputs(self, observations: Observation, num_candidates: int) -> Observation:
        def _tile(x: jnp.ndarray) -> jnp.ndarray:
            x = jnp.asarray(x)
            batch = x.shape[0]
            x = jnp.repeat(x, num_candidates, axis=0)
            return x.reshape(batch * num_candidates, *x.shape[1:])

        return jax.tree_util.tree_map(_tile, observations)

    def _call_model(
        self,
        rng: PRNGKey,
        observations: Observation,
        num_candidates: int,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        batch = self.batch_size(observations)
        flat_obs = self._prepare_inputs(observations, num_candidates)
        input_collections: Dict[str, Any] = {"params": self._params}
        if self._batch_stats is not None:
            input_collections["batch_stats"] = self._batch_stats
        try:
            outputs = self._apply_fn(
                input_collections,
                flat_obs,
                training=False,
                rng=rng,
                num_samples=num_candidates,
            )
        except TypeError:
            outputs = self._apply_fn(
                input_collections,
                flat_obs,
                training=False,
                rng=rng,
            )
        if isinstance(outputs, tuple):
            samples = outputs[0]
        else:
            samples = outputs
        samples = jnp.asarray(samples)
        samples = samples.reshape(batch, num_candidates, *self._chunk_shape)
        return samples, {}

    def sample(
        self,
        rng: PRNGKey,
        observations: Observation,
        num_candidates: int,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive.")
        if observations is None:
            raise ValueError("observations must be provided.")

        observations = jax.tree_util.tree_map(jnp.asarray, observations)

        if self._apply_fn is None:
            batch = self.batch_size(observations)
            noise = jax.random.normal(
                rng,
                (batch, num_candidates, *self._chunk_shape),
            )
            return noise * self._scale, {}

        return self._call_model(rng, observations, num_candidates)

    @staticmethod
    def batch_size(observations: Observation) -> int:
        leaves, _ = jax.tree_util.tree_flatten(observations)
        if not leaves:
            raise ValueError("observations must contain at least one array.")
        return leaves[0].shape[0]

    def __call__(
        self,
        rng: PRNGKey,
        observations: Observation,
        num_candidates: int,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return self.sample(rng, observations, num_candidates)
