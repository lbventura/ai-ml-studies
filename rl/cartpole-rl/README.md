# Reinforcement Learning for the Cartpole environment

The example displayed here is mostly derived from [Tutorial 10](https://nbviewer.jupyter.org/url/www.cs.toronto.edu/~rgrosse/courses/csc421_2019/tutorials/tut10/policy_gradient_cartpole.ipynb) of the Neural Networks and Deep Learning course of the University of Toronto. See more information in the course page [here](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/).

In this example, a policy gradient method is used to train an agent to balance a pole on a cart, by moving the cart to the left or to the right. The agent receives a reward of +1 for each time step that the pole is balanced. The agent is trained using the REINFORCE algorithm, which is a policy gradient method that uses the likelihood ratio trick to estimate the gradient of the expected return with respect to the policy parameters.

The data for this project is sourced from the [OpenAI Gym](https://gym.openai.com/) library, a comprehensive toolkit for developing and comparing reinforcement learning algorithms. See details below.

## Cartpole environment

It is important to note the return value of `env.step(action)`, where `env=gym.make("CartPole-v1")`. This is a 4-tuple: `observation, reward, done, info = env.step(action)`, composed by:

`observation`: Represents the new state of the environment after the action is taken. For `CartPole-v1`, it is a NumPy array containing four values:
    1. Cart Position: Position of the cart on the track.
    2. Cart Velocity: Velocity of the cart.
    3. Pole Angle: Angle of the pole with the vertical.
    4. Pole Velocity at Tip: Angular velocity of the pole.

`reward`: The immediate reward received after taking the action. For `CartPole-v1`, it is typically +1 for every step the pole remains upright. The goal is to maximize cumulative rewards by keeping the pole balanced for as long as possible.

`done`: Indicates whether the episode has ended. For `CartPole-v1`, it is `True` if the pole's angle exceeds ±12 degrees from vertical, or the cart's position exceeds ±2.4 units from the center or the episode length reaches 500 steps.

As one example, one can create an environment and move the cart to the right to watch the pole tip to the left:

![Example](figures/cartpole-movement.png)
