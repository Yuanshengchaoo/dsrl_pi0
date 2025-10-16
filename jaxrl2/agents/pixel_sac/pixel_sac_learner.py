"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.critic_updater import update_critic
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.agents.qc_chunk import FlowMatchingChunkPrior, FlowMatchingChunkPriorConfig


class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter',  'aug_next', 'num_cameras'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(key, actor, critic, target_critic, temp, batch, discount, critic_reduction=critic_reduction)
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, critic_reduction=critic_reduction)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class PixelSACLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter = True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 qc_use_best_of_n: bool = False,
                 qc_num_candidates: int = 1,
                 qc_eval_batch_size: int = 32,
                 qc_prior_scale: float = 1.0,
                 qc_prior_config: Optional[Dict[str, Any]] = None,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        policy_def = LearnedStdTanhNormalPolicy(hidden_dims, self.action_dim, dropout_rate=dropout_rate, low=-action_magnitude, high=action_magnitude)

        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        print(actor_def)
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)

        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        print(critic_def)
        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init['params']

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr),
                                   batch_stats=critic_batch_stats
                                   )
        target_critic_params = copy.deepcopy(critic_params)
        
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)


        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(self.critic_reduction)

        self._last_sample_info: Optional[Dict[str, Any]] = None
        self._qc_use_best_of_n = bool(qc_use_best_of_n and qc_num_candidates > 0)
        self._qc_num_candidates = int(max(1, qc_num_candidates))
        self._qc_eval_batch_size = int(max(1, qc_eval_batch_size))
        self._qc_prior: Optional[FlowMatchingChunkPrior] = None

        if self._qc_use_best_of_n:
            prior_cfg_kwargs = qc_prior_config.copy() if qc_prior_config else {}
            if 'chunk_shape' in prior_cfg_kwargs:
                prior_cfg_kwargs.pop('chunk_shape')
            if 'scale' not in prior_cfg_kwargs:
                prior_cfg_kwargs['scale'] = qc_prior_scale
            if 'sample_observation' not in prior_cfg_kwargs:
                prior_cfg_kwargs['sample_observation'] = observations
            allowed_keys = {'checkpoint_path', 'model_def', 'sample_observation', 'init_rng', 'scale'}
            unexpected = set(prior_cfg_kwargs.keys()) - allowed_keys
            if unexpected:
                raise ValueError(f"Unsupported qc_prior_config keys: {unexpected}")
            prior_config = FlowMatchingChunkPriorConfig(
                chunk_shape=self.action_chunk_shape,
                **prior_cfg_kwargs,
            )
            self._qc_prior = FlowMatchingChunkPrior.from_config(prior_config)

        def critic_apply(params, batch_stats, obs, actions):
            input_collections = {'params': params}
            if batch_stats is not None:
                input_collections['batch_stats'] = batch_stats
            return self._critic.apply_fn(input_collections, obs, actions)

        self._critic_eval_fn = jax.jit(critic_apply)
        

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params, self._temp, batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras
            )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        return info

    def _critic_batch_stats(self) -> Optional[Any]:
        return getattr(self._critic, 'batch_stats', None)

    def _evaluate_candidates(self, observations, candidates: jnp.ndarray) -> jnp.ndarray:
        batch_size = candidates.shape[0]
        num_candidates = candidates.shape[1]

        obs_batched = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x), num_candidates, axis=0),
            observations,
        )
        flat_candidates = candidates.reshape(batch_size * num_candidates, *candidates.shape[2:])

        total = flat_candidates.shape[0]
        chunk_qs = []
        for start in range(0, total, self._qc_eval_batch_size):
            end = min(start + self._qc_eval_batch_size, total)
            obs_slice = jax.tree_util.tree_map(lambda x: x[start:end], obs_batched)
            action_slice = flat_candidates[start:end]
            qs_slice = self._critic_eval_fn(
                self._critic.params,
                self._critic_batch_stats(),
                obs_slice,
                action_slice,
            )
            if qs_slice.ndim == 1:
                qs_slice = qs_slice[None, :]
            chunk_qs.append(qs_slice)

        qs = jnp.concatenate(chunk_qs, axis=1)
        qs = qs.reshape(qs.shape[0], batch_size, num_candidates)

        if self.critic_reduction == 'min':
            reduced = qs.min(axis=0)
        elif self.critic_reduction == 'mean':
            reduced = qs.mean(axis=0)
        else:
            raise ValueError(f'Unsupported critic reduction: {self.critic_reduction}')

        return reduced

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        if not self._qc_use_best_of_n or self._qc_prior is None:
            self._last_sample_info = None
            return super().sample_actions(observations)

        rng, key = jax.random.split(self._rng)
        self._rng = rng

        observations_jnp = jax.tree_util.tree_map(jnp.asarray, observations)
        candidates, prior_info = self._qc_prior(key, observations_jnp, self._qc_num_candidates)

        candidate_qs = self._evaluate_candidates(observations_jnp, candidates)
        best_indices = jnp.argmax(candidate_qs, axis=1)

        def select(cand, idx):
            return cand[idx]

        best_chunks = jax.vmap(select)(candidates, best_indices)
        best_q = jnp.take_along_axis(candidate_qs, best_indices[:, None], axis=1).squeeze(-1)

        self._last_sample_info = {
            'qc_candidates': np.asarray(candidates),
            'qc_candidate_q_values': np.asarray(candidate_qs),
            'qc_best_index': np.asarray(best_indices),
            'qc_best_q_value': np.asarray(best_q),
            'prior_info': prior_info,
        }

        return np.asarray(best_chunks)

    @property
    def last_sample_info(self) -> Optional[Dict[str, Any]]:
        return self._last_sample_info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)
        
    
@functools.partial(jax.jit)
def get_value(action, observation, critic):
    input_collections = {'params': critic.params}
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, rewards, masks, images):

    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image