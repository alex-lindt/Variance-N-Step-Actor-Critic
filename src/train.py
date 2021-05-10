import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from .helpers import save_models, compute_q_val


def train_actor_critic(env, environment_name, models, optimizer, num_episodes, gamma, n_step=1,
                       model_type="Advantage", device="cpu"):
    """
    :param env:
    :param models:
    :param optimizer:
    :param num_episodes:
    :param gamma:
    :param n_step:
    :param model_names:
    :return:
    """
    episode_durations = []
    cumulative_reward = []
    actor_losses = []
    critic_losses = []
    running_average = []

    actor, critic = models
    opt_actor, opt_critic = optimizer

    # loop for each episode
    print("Training Started. Runs {} episodes with n-step: {} ...".format(num_episodes, n_step))
    for episode in range(num_episodes):
        states, actions, critic_values, rewards, log_probs, pi_entropy, done_list = run_episode(env, actor, critic,
                                                                                                model_type)
        cumulative_reward.append(np.sum(rewards))
        episode_durations.append(len(rewards))
        running_average.append(np.mean(cumulative_reward))

        # TODO: probably here a good place distinguish different critics and returns
        # Monte Carlo, otherwise n-step 1 ... N
        if type(n_step) == int:
            if len(rewards) < n_step:
                n_longer_than_r = True
            else:
                n_longer_than_r = False
        if n_step == "Monte Carlo" or n_longer_than_r == True:
            discounted_returns = get_cumulative_discounted_rewards(rewards, done_list, gamma)
            discounted_returns = (discounted_returns - discounted_returns.mean()) / discounted_returns.std()
            actor_loss, critic_loss = calculate_loss(discounted_returns, log_probs, critic_values, pi_entropy,
                                                     actions=actions, model_type=model_type)
        else:
            n_step_returns = compute_n_step_returns(rewards, critic_values, gamma, n_step, model_type, actions=actions,
                                                    done=done_list)
            n_step_returns = (n_step_returns - n_step_returns.mean()) / n_step_returns.std()

            actor_loss, critic_loss = calculate_loss(n_step_returns, log_probs, critic_values, pi_entropy,
                                                     actions=actions, model_type=model_type)

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        opt_actor.step()
        opt_critic.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        if episode % 50 == 0:
            print(
                "Episode: {:0>4d}/{:0>4d}:  current reward: {:.2f}, average reward: {:.2f},   current steps: {:.2f},   loss: {:.2f} ".format(
                    episode,
                    num_episodes,
                    np.sum(rewards),
                    np.mean(cumulative_reward),
                    len(rewards),
                    actor_loss))

    save_models((actor, critic), model_type, n_step, environment_name)

    return episode_durations, cumulative_reward, actor_losses, critic_losses, running_average


def run_episode(env, actor, critic, model_type):
    """

    :param env:
    :param actor:
    :param critic:
    :return:
    """
    states, actions, critic_values, rewards, log_probs, pi_entropies, done_list = [], [], [], [], [], [], []

    # only used if estimating q_values
    s = env.reset()
    while True:
        a, log_prob, pi_s_a = select_action(actor, s)

        if model_type == "Q":
            # v_s in this case all q-values
            _, v_s = compute_q_val(critic, s, a)
        else:
            v_s = critic(s)

        s_, reward, done, _ = env.step(a)

        # For better exploration ?!
        pi_entropy = pi_s_a.entropy()

        critic_values.append(v_s)
        states.append(s)
        actions.append(a)
        rewards.append(torch.FloatTensor([reward]))
        log_probs.append(log_prob)
        pi_entropies.append(pi_entropy)
        done_list.append(torch.FloatTensor([1 - done]))

        s = s_

        if done:
            if model_type == "Q":
                _, v_s = compute_q_val(critic, s, a)
            else:
                v_s = critic(s)
            critic_values.append(v_s)
            break

    return states, torch.tensor(actions, dtype=torch.int64), torch.stack(critic_values), rewards, torch.stack(
        log_probs), torch.stack(
        pi_entropies), torch.FloatTensor(done_list)


def select_action(model, state):
    """

    :param model:
    :param state:
    :return:
    """
    pi_s_a = model(state)
    a = pi_s_a.sample()
    log_prob = pi_s_a.log_prob(a).unsqueeze(0)

    return a.item(), log_prob, pi_s_a


def get_cumulative_discounted_rewards(rewards, done_list, gamma):
    """
    This calculates basically the full reinforce return

    :param rewards:
    :param done_list:
    :param gamma:
    :return:
    """
    cumulative_reward, _r = [], 0
    for i in np.arange(len(rewards))[::-1]:
        _r = done_list[i] * _r * gamma + rewards[i]
        cumulative_reward.insert(0, _r)
    return torch.stack(cumulative_reward)


# TODO: add action
def compute_n_step_returns(rewards, values, gamma, n_step, model_type, actions=None, done=None):
    """
    Should return n_step target ((γ^0) Rt+1 +  (γ^1)Rt+2 ...  (γ^t+n-1)Rt+n + (γ^t+n)V(St+n)) - V(S_2)
    :param rewards:
    :param values:
    :param gamma:
    :param n_step:
    :return:
    """

    T = len(rewards)
    returns = []
    for i in np.arange(len(rewards)):
        returns.append(torch.sum(
            torch.stack(rewards[i: np.min((i + n_step, T))]) * torch.FloatTensor(
                np.power(gamma, np.arange(len(rewards[i: np.min((i + n_step, T))]))))[:, None]))
    returns = torch.stack(returns)[:, None]

    if model_type == "Q":

        # Basically SARSA or Q-learning
        # max_q_values_ = torch.gather(values.detach()[:-1], 1, actions[:, None])
        # max_q_values_ = torch.cat((max_q_values_, values.detach()[-1:, 0][:, None]), dim=0)
        max_q_values_ = torch.max(values.detach(), dim=1)[0][:, None]

        if n_step == 1:
            returns += torch.FloatTensor(np.power(gamma, np.ones(T)))[:, None] * max_q_values_[1:] * done[:, None]

        else:
            returns[:T - (n_step - 1)] += torch.FloatTensor(np.power(gamma, np.ones(T - (n_step - 1)) * n_step))[:,
                                          None] * max_q_values_[n_step:] * done[:T - (n_step - 1), None]

            returns[T - (n_step - 1):] += torch.FloatTensor(np.power(gamma, np.ones(n_step - 1) * n_step))[:,
                                          None] * max_q_values_[T] * done[T - (n_step - 1):, None]

        return returns

    else:
        if n_step == 1:
            returns += torch.FloatTensor(np.power(gamma, np.ones(T)))[:, None] * values[1:]

        else:
            returns[:T - (n_step - 1)] += torch.FloatTensor(np.power(gamma, np.ones(T - (n_step - 1)) * n_step))[:,
                                          None] * values[n_step:]

            returns[T - (n_step - 1):] += torch.FloatTensor(np.power(gamma, np.ones(n_step - 1) * n_step))[:,
                                          None] * values[T]

        return returns


def calculate_loss(discounted_returns, log_prob, v_s, pi_entropy, actions=None, model_type="Advantage", batch=True):
    """

    :param discounted_returns:.unwrapped
    :param log_prob:
    :param v_s:
    :return:
    """

    if model_type == "Advantage" or model_type == "Reinforce":
        advantage = discounted_returns - v_s[:-1]

        loss_actor = (-log_prob * advantage.detach() - pi_entropy * 0.01).mean()
        loss_critic = advantage.pow(2).mean()

        return loss_actor, loss_critic * 0.5

    elif model_type == "Q":
        q_values = torch.gather(v_s[:-1], 1, actions[:, None])
        q_error = discounted_returns - q_values

        loss_actor = (-log_prob * q_error.detach() - pi_entropy * 0.01).mean()
        # loss_actor = (-log_prob * q_error.detach()).mean()
        # loss_actor = (-log_prob * q_values.detach()).mean()
        # Q learning loss
        # loss_critic = F.smooth_l1_loss(q_values, discounted_returns)
        loss_critic = q_error.pow(2).mean()

        return loss_actor, loss_critic * 0.5
