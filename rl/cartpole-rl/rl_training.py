from typing import Any
import gym

from cartpole_rl.policy_net import PolicyNet
from cartpole_rl.rl_training_loop import rl_training_loop
from cartpole_rl.data_types import TrainingParams
from cartpole_rl.training_utils import plot_reward_history
from pathlib import Path


def video_monitor_callable(*args: Any) -> bool:
    return True


if __name__ == "__main__":
    # Define environment
    env = gym.make("CartPole-v1")

    # Define the path to save the videos
    run_info_path = str(Path(__file__).parent / "cartpole_run_info")

    # Create environment monitor for video recording
    monitored_env = gym.wrappers.RecordVideo(
        env, run_info_path, episode_trigger=video_monitor_callable
    )

    # Define state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    bernoulli_action_dim = 1

    # Initialize policy network
    # The policy network is a simple neural network that takes the state as input and outputs the probability of taking action 1 (going to the right).
    # This is then used to sample the action from a Bernoulli distribution.
    # If, for example, the output of the policy network is 0.7, then the probability of taking action 1 is 0.7.
    policy_net = PolicyNet(input_dim=state_dim, output_dim=bernoulli_action_dim)

    # Train the policy network
    traj_reward_history = rl_training_loop(
        policy_net=policy_net,
        training_params=TrainingParams(num_episodes=500),
        env=env,
        monitored_env=monitored_env,
    )

    # Don't forget to close the environments.
    monitored_env.close()
    env.close()

    # Plot the reward history
    plot_reward_history(
        traj_reward_history=traj_reward_history, save_path=run_info_path
    )
