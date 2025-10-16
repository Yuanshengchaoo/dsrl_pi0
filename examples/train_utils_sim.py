from tqdm import tqdm
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    action_chunk_shape = agent.action_chunk_shape
    diffusion_steps = getattr(variant, 'diffusion_steps', None)
    if diffusion_steps is None:
        raise ValueError('variant.diffusion_steps must be specified for trajectory collection.')
    if getattr(variant, 'query_freq', -1) <= 0:
        variant.query_freq = diffusion_steps
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(
                variant,
                agent,
                env,
                i,
                agent_dp,
                action_chunk_shape=action_chunk_shape,
                diffusion_steps=diffusion_steps,
            )
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.diffusion_steps
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    chunk_infos = traj.get('chunk_infos', [])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        if chunk_infos:
            info = chunk_infos[t] if t < len(chunk_infos) else None
            if info is not None and info.get('qc_candidate_q_values') is not None:
                candidate_qs = np.asarray(info['qc_candidate_q_values'])
                if candidate_qs.ndim > 1:
                    candidate_qs = candidate_qs.reshape(-1, candidate_qs.shape[-1])[0]
                candidate_qs = candidate_qs.astype(np.float32)
                best_index = int(np.asarray(info['qc_best_index']).reshape(-1)[0])
                best_q = float(np.asarray(info['qc_best_q_value']).reshape(-1)[0])
                insert_dict['qc'] = {
                    'candidate_q_values': candidate_qs,
                    'best_index': best_index,
                    'best_q': best_q,
                }
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None, action_chunk_shape=None, diffusion_steps=None):
    if action_chunk_shape is None:
        action_chunk_shape = agent.action_chunk_shape
    if diffusion_steps is None:
        diffusion_steps = variant.diffusion_steps

    noise_chunk_length = action_chunk_shape[0]

    def pad_noise_to_horizon(noise_seq):
        extra = diffusion_steps - noise_seq.shape[1]
        if extra > 0:
            pad = jax.numpy.repeat(noise_seq[:, -1:, :], extra, axis=1)
            noise_seq = jax.numpy.concatenate([noise_seq, pad], axis=1)
        elif extra < 0:
            noise_seq = noise_seq[:, :diffusion_steps, :]
        return noise_seq

    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []
    chunk_info_list = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)

        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if i == 0:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
                chunk_info_list.append(None)
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                chunk_info_list.append(agent.last_sample_info)

            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    query_steps = len(action_list)
    rewards = -np.ones(query_steps)
    masks = np.ones(query_steps)
    if is_success and query_steps > 0:
        rewards[-1] = 0
        masks[-1] = 0

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'chunk_infos': chunk_info_list,
        'env_steps': t + 1
    }

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    diffusion_steps = getattr(variant, 'diffusion_steps', None)
    if diffusion_steps is None:
        raise ValueError('variant.diffusion_steps must be specified for evaluation.')
    if getattr(variant, 'query_freq', -1) <= 0:
        variant.query_freq = diffusion_steps
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)
    action_chunk_shape = agent.action_chunk_shape
    noise_chunk_length = action_chunk_shape[0]

    def pad_noise_to_horizon(noise_seq):
        extra = diffusion_steps - noise_seq.shape[1]
        if extra > 0:
            pad = jax.numpy.repeat(noise_seq[:, -1:, :], extra, axis=1)
            noise_seq = jax.numpy.concatenate([noise_seq, pad], axis=1)
        elif extra < 0:
            noise_seq = noise_seq[:, :diffusion_steps, :]
        return noise_seq

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                rng, key = jax.random.split(rng)
                assert agent_dp is not None

                obs_pi_zero = obs_to_pi_zero_input(obs, variant)


                if i == 0:
                    base_noise = jax.random.normal(key, (1, *action_chunk_shape))
                    actions_noise = np.asarray(base_noise)[0, :noise_chunk_length, :]
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.asarray(actions_noise).reshape(action_chunk_shape)
                    base_noise = jax.numpy.asarray(actions_noise[None])

                noise = pad_noise_to_horizon(base_noise)

                actions = np.asarray(agent_dp.infer(obs_pi_zero, noise=np.asarray(noise))["actions"])
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
