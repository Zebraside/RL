import torch
from torch.autograd import Variable
import numpy as np
import gym
from dqn_agent import DQNAgent
from preprocess import PreprocessAtari
from replay_buffer import ReplayBuffer


def compute_td_loss(states, actions, rewards, next_states, is_done, gamma=0.99, check_shapes=False):
    """ Compute td loss using torch operations only."""
    states = Variable(torch.FloatTensor(states))  # shape: [batch_size, state_size]
    actions = Variable(torch.IntTensor(actions))  # shape: [batch_size]
    rewards = Variable(torch.FloatTensor(rewards))  # shape: [batch_size]
    next_states = Variable(torch.FloatTensor(next_states))  # shape: [batch_size, state_size]
    is_done = Variable(torch.FloatTensor(is_done))  # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues = network(states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = torch.sum(predicted_qvalues.cpu() * to_one_hot(actions, n_actions), dim=1)

    # compute q-values for all actions in next states
    predicted_next_qvalues = network(next_states)

    # compute V*(next_states) using predicted next q-values
    next_state_values = predicted_next_qvalues.max()

    assert isinstance(next_state_values.data, torch.FloatTensor)

    # compute 'target q-values' for loss
    target_qvalues_for_actions = rewards + gamma * next_state_values

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = where(is_done, rewards, target_qvalues_for_actions).cpu()

    # Mean Squared Error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            'make sure you predicted q-values for all actions in next state'
        assert next_state_values.data.dim() == 1, \
            'make sure you computed V(s-prime) as maximum over just the actions axis and not all axes'
        assert target_qvalues_for_actions.data.dim() == 1, \
            'there is something wrong with target q-values, they must be a vector'

    return loss


if __name__ == '__main__':
    env = gym.make("BreakoutDeterministic-v0")  # create raw env
    env = PreprocessAtari(env)

    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state_dim = observation_shape
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    agent = DQNAgent(state_dim, n_actions, epsilon=0.5)
    target_network = DQNAgent(state_dim, n_actions)

    exp_replay = ReplayBuffer(10)
    for _ in range(30):
        exp_replay.add(env.reset(), env.action_space.sample(), 1.0, env.reset(), done=False)

    target_network.load_state_dict(agent.state_dict())
    # sanity checks
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(10)

    loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, gamma=0.99,
                           check_shapes=True)
    loss.backward()

    assert np.any(next(agent.parameters()).grad.data.numpy() != 0), "loss must be differentiable w.r.t. network weights"
    print("TD Loss OK")
