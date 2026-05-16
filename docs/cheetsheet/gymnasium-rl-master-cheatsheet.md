---
title: Gymnasium RL Master Cheatsheet
sidebar_position: 21
---

# Gymnasium RL Master Cheatsheet

## Environments and spaces

| Method | Description | Code example |
|---|---|---|
| `gym.make()` | Creates an environment by ID. | `import gymnasium as gym`<br/>`env = gym.make("CartPole-v1")` |
| `env.reset()` | Starts a new episode. Returns observation and info. | `obs, info = env.reset(seed=42)` |
| `env.step()` | Runs one action. Returns observation, reward, terminated, truncated, info. | `obs, reward, terminated, truncated, info = env.step(action)` |
| `env.action_space` | Describes valid actions. | `action = env.action_space.sample()` |
| `env.observation_space` | Describes observation shape and bounds. | `print(env.observation_space)` |
| `env.close()` | Releases resources. | `env.close()` |

## Episode loops

| Method | Description | Code example |
|---|---|---|
| Random policy loop | Baseline interaction loop. | `obs, info = env.reset()`<br/>`done = False`<br/>`while not done:`<br/>`    action = env.action_space.sample()`<br/>`    obs, reward, terminated, truncated, info = env.step(action)`<br/>`    done = terminated or truncated` |
| Accumulate reward | Track return per episode. | `total_reward = 0`<br/>`obs, info = env.reset()`<br/>`while True:`<br/>`    action = policy(obs)`<br/>`    obs, reward, terminated, truncated, info = env.step(action)`<br/>`    total_reward += reward`<br/>`    if terminated or truncated: break` |
| Render | Visualize environment. | `env = gym.make("CartPole-v1", render_mode="human")` |
| Seed | Makes experiments more reproducible. | `obs, info = env.reset(seed=123)`<br/>`env.action_space.seed(123)` |
| Vector env | Runs multiple environments for faster collection. | `envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])`<br/>`obs, info = envs.reset()` |

## DQN and PPO scaffolding

| Method | Description | Code example |
|---|---|---|
| Replay buffer | Stores transitions for off-policy algorithms like DQN. | `transition = (obs, action, reward, next_obs, done)`<br/>`replay.append(transition)` |
| Epsilon-greedy | Balances exploration and exploitation. | `if random.random() < epsilon:`<br/>`    action = env.action_space.sample()`<br/>`else:`<br/>`    action = q_network.act(obs)` |
| Bellman target | DQN target for one-step TD learning. | `target = reward + gamma * next_q.max(dim=1).values * (1 - done)` |
| Policy gradient loss | PPO-style clipped objective skeleton. | `ratio = torch.exp(new_logprob - old_logprob)`<br/>`loss = -torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage).mean()` |
| Advantage estimate | Reward-to-go minus value baseline. | `advantage = returns - values.detach()` |

## Wrappers and Stable-Baselines3

| Method | Description | Code example |
|---|---|---|
| `ObservationWrapper` | Transforms observations. | `class NormalizeObs(gym.ObservationWrapper):`<br/>`    def observation(self, obs):`<br/>`        return obs / 255.0` |
| `RewardWrapper` | Transforms rewards. | `class ClipReward(gym.RewardWrapper):`<br/>`    def reward(self, reward):`<br/>`        return float(np.clip(reward, -1, 1))` |
| `TimeLimit` | Adds max episode length. | `env = gym.wrappers.TimeLimit(env, max_episode_steps=500)` |
| SB3 PPO | Train PPO with stable-baselines3. | `from stable_baselines3 import PPO`<br/>`model = PPO("MlpPolicy", "CartPole-v1", verbose=1)`<br/>`model.learn(total_timesteps=10000)` |
| SB3 evaluate | Evaluate trained model. | `obs, info = env.reset()`<br/>`action, state = model.predict(obs, deterministic=True)` |
| Save and load | Persist trained policy. | `model.save("ppo_cartpole")`<br/>`model = PPO.load("ppo_cartpole")` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Custom env | Implement `reset`, `step`, spaces, and optional render. | `class TradingEnv(gym.Env):`<br/>`    def __init__(self):`<br/>`        self.action_space = gym.spaces.Discrete(3)`<br/>`        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)` |
| Training loop skeleton | Collect transition, update policy, repeat. | `for episode in range(num_episodes):`<br/>`    obs, info = env.reset()`<br/>`    while True:`<br/>`        action = agent.act(obs)`<br/>`        next_obs, reward, terminated, truncated, info = env.step(action)`<br/>`        agent.update(obs, action, reward, next_obs, terminated or truncated)` |
| Evaluate policy | Run deterministic episodes and average return. | `returns = []`<br/>`for _ in range(10):`<br/>`    returns.append(run_episode(env, policy))`<br/>`print(np.mean(returns))` |
| Normalize observations | Important for neural policies. | `obs_mean, obs_std = obs_batch.mean(axis=0), obs_batch.std(axis=0) + 1e-8`<br/>`obs_norm = (obs - obs_mean) / obs_std` |
| Clip rewards | Stabilizes some Atari-style tasks. | `reward = np.sign(reward)` |
| Save video | Record episodes for debugging. | `env = gym.wrappers.RecordVideo(env, video_folder="videos")` |
| Track experiments | Log episode returns and lengths. | `wandb.log({"episode_return": total_reward, "episode_length": steps})` |
| Common done flag | Combine Gymnasium termination flags. | `done = terminated or truncated` |

## Senior RL engineering

| Method | Description | Code example |
|---|---|---|
| Separate terminated and truncated | Treat environment success/failure differently from time-limit cutoff. | `bootstrap = 0 if terminated else value(next_obs)`<br/>`target = reward + gamma * bootstrap` |
| Reproducibility bundle | Seed env, action space, NumPy, Python, and torch. | `random.seed(seed)`<br/>`np.random.seed(seed)`<br/>`torch.manual_seed(seed)`<br/>`obs, info = env.reset(seed=seed)`<br/>`env.action_space.seed(seed)` |
| Evaluation protocol | Evaluate without exploration noise and with fixed seeds. | `returns = [run_eval_episode(env, agent, seed=i) for i in range(20)]`<br/>`print(np.mean(returns), np.std(returns))` |
| Replay buffer design | Store arrays in preallocated buffers for performance. | `obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)`<br/>`act_buf = np.zeros(capacity, dtype=np.int64)` |
| Target network update | Stabilizes DQN training. | `if step % target_update_every == 0:`<br/>`    target_net.load_state_dict(q_net.state_dict())` |
| Entropy bonus | Encourages exploration in policy-gradient methods. | `loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()` |
| Reward normalization | Reduces scale instability. | `reward_norm = (reward - reward_mean) / (reward_std + 1e-8)` |
| Curriculum | Gradually increase task difficulty. | `env.unwrapped.set_difficulty(min(max_difficulty, episode // 1000))` |

## Failure modes and diagnostics

| Method | Description | Code example |
|---|---|---|
| Action distribution | Detect collapsed policies. | `counts = np.bincount(actions, minlength=env.action_space.n)`<br/>`print(counts / counts.sum())` |
| Value drift | Monitor critic predictions for explosion. | `wandb.log({"value_mean": values.mean(), "value_std": values.std()})` |
| Reward hacking | Log environment state and videos, not only reward. | `env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda i: i % 100 == 0)` |
| Off-policy freshness | Avoid training too long on stale replay without new collection. | `updates_per_step = gradient_updates / env_steps` |
| Exploration schedule | Decay epsilon with a floor. | `epsilon = max(eps_min, eps_start - step / decay_steps * (eps_start - eps_min))` |
| Gradient health | Track norm and clipping frequency. | `grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)` |
| Checkpoint policy and optimizer | Save enough to resume, not just evaluate. | `torch.save({"model": policy.state_dict(), "optim": optimizer.state_dict(), "step": step}, "ckpt.pt")` |
| Baselines first | Compare against random and simple heuristics before deep RL. | `random_return = np.mean([run_random_episode(env) for _ in range(100)])` |
