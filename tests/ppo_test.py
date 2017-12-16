import gym
from stats import RunningStats


env = ...
state_dim = ...
action_dim = ...
state_dim += 1
policy = ...
stats = RunningStats()
value_function = ValueFunction(state_dim)
policy = Policy(state_dim, action_dim, target_kl)
run_policy(env, policy, scaler, episodes=5)
episode = 0

while episode < num_episodes:
    trajectory = run_policy(env, policy, scaler, episodes=batch_size)
    episode += len(trajectory)
    values = value_function.predict(states)
    discounted_reward = 0
    target_values = []
    for r in rewards[::-1]:
        discounted_reward = r + gamma * discounted_reward
        target_values.append(discounted_reward)
    target_values = target_values[::-1]
    advantage = target_values - values
    normalized_advantage = (advantage - advantage.mean()) / advantage.std()
    policy.update(state, action, advantage)
    value_function.fig(states, target_values)


def runEpisode():
    state = env.reset()
    trajectory = []
    done = False
    step = .0
    scale, offset = scaler.get()
    scale[-1] = 1.
    offset[-1] = 0.
    while not done:
        state = np.append(state, step)
        norm_state = (state - offset) * scale
        action = policy.sample(state)
        state, reward, done, _ = env.step(action)
        trajectory.append([state, action, reward])
        step += 1e-3

    return zip(*trajectory)


def run_policy(env, policy, scaler, episodes):
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        states, actions, rewards = run_episode(env, policy_scaler)
        total_steps += len(states)
        trajectory.append([states, actions, rewards])
    scaler.update(unscaled)
