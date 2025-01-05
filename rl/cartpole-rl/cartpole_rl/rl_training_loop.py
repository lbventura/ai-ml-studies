from typing import Any
import torch
from torch.distributions import Bernoulli
from cartpole_rl.policy_net import PolicyNet
from cartpole_rl.training_utils import convert_to_torch, record_video
from cartpole_rl.data_types import TrainingParams


def rl_training_loop(
    policy_net: PolicyNet, training_params: TrainingParams, env: Any, monitored_env: Any
) -> list[float]:
    """Training loop for policy gradient algorithm."""
    # Set baseline to 0
    baseline = 0.0

    # Define optimizer
    optimizer = torch.optim.RMSprop(
        policy_net.parameters(), lr=training_params.learning_rate
    )

    # Collect trajectory rewards for plotting purpose
    traj_reward_history = []

    # training loop , iterate over episodes
    for ep_i in range(training_params.num_episodes):
        loss = 0.0

        # Record states, actions and discounted rewards of this episode
        states = []
        actions = []
        rewards = []
        cumulative_undiscounted_reward = 0.0

        for _ in range(training_params.batch_size):
            time_step = 0
            done = False

            # initialize environment
            cur_state = env.reset()

            cur_state = convert_to_torch(cur_state)

            discount_factor = 1.0
            discounted_rewards = []

            while not done:
                # Compute action probability using the current policy
                action_prob = policy_net(cur_state)

                # Sample action according to action probability
                action_sampler = Bernoulli(probs=action_prob)
                action_x = action_sampler.sample()
                action = action_x.numpy().astype(int)[0]

                # Record the states and actions -- will be used for policy gradient later
                states.append(cur_state)
                actions.append(action)

                # take a step in the environment, and collect data
                # see state details in the README.md file
                next_state, reward, done, _ = env.step(action)

                # Discount the reward, and append to reward list
                discounted_reward = reward * discount_factor
                discounted_rewards.append(discounted_reward)
                cumulative_undiscounted_reward += reward

                # Prepare for taking the next step
                cur_state = convert_to_torch(next_state)

                time_step += 1
                # the discount factor has a decay
                discount_factor *= training_params.gamma

            # Finished collecting data for the current trajectory.
            # Recall temporal structure in policy gradient.
            # Construct the "cumulative future discounted reward" at each time step.
            for time_i in range(time_step):
                # relevant reward is the sum of rewards from time t to the end of trajectory
                relevant_reward = sum(discounted_rewards[time_i:])
                rewards.append(relevant_reward)

        # Finished collecting data for this batch. Update policy using policy gradient.
        avg_traj_reward = cumulative_undiscounted_reward / training_params.batch_size
        traj_reward_history.append(avg_traj_reward)

        if (ep_i + 1) % 10 == 0:
            print(
                "Episode {}: Average reward per trajectory = {}".format(
                    ep_i + 1, avg_traj_reward
                )
            )

        if (ep_i + 1) % 100 == 0:
            record_video(monitored_env=monitored_env, policy_net=policy_net)

        optimizer.zero_grad()
        data_len = len(states)
        loss = 0.0

        # Compute the policy gradient
        for data_i in range(data_len):
            action_prob = policy_net(states[data_i])
            action_sampler = Bernoulli(probs=action_prob)

            loss -= action_sampler.log_prob(
                torch.tensor(actions[data_i], dtype=torch.float)
            ) * (rewards[data_i] - baseline)
        loss /= float(data_len)
        loss.backward()  # type: ignore
        optimizer.step()
    return traj_reward_history
