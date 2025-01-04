import torch
import numpy as np
from torch.distributions import Bernoulli
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt


def convert_to_torch(arr: np.ndarray) -> torch.Tensor:
    """Converts a numpy array to torch variable"""
    return torch.from_numpy(arr).float()


def record_video(monitored_env: RecordVideo, policy_net: torch.nn.Module) -> None:
    print("Recording video")
    recorder_cur_state = monitored_env.reset()
    recorder_cur_state = convert_to_torch(recorder_cur_state)
    recorder_done = False

    while not recorder_done:
        recorder_action = (
            Bernoulli(probs=policy_net(recorder_cur_state))
            .sample()
            .numpy()
            .astype(int)[0]
        )

        recorder_next_state, _, recorder_done, _ = monitored_env.step(recorder_action)
        recorder_cur_state = convert_to_torch(recorder_next_state)


def plot_reward_history(traj_reward_history: list[float], save_path: str) -> None:
    # Plot learning curve
    plt.figure()
    plt.plot(traj_reward_history)
    plt.title("Learning to Solve CartPole-v1 with Policy Gradient")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Trajectory")
    plt.savefig(f"{save_path}/CartPole-pg.png")
    plt.show()
    plt.close()
