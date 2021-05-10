from .actor_critic import ActorCritic


def old_run(env, num_episodes, gamma, learn_rate, n_hidden=256):
    n_state_features = len(env.reset())
    n_actions = env.action_space.n

    model = ActorCritic(n_state_features, n_actions, n_hidden)

    optimizer = optim.Adam(model.parameters(), learn_rate)

    episode_durations = []

    # loop for each episode
    for episode in tqdm(range(num_episodes)):

        # initialize S (first state of episode)
        s = env.reset()
        I = 1
        step = 0

        while True:

            optimizer.zero_grad()

            v_s, pi_s_a = model.forward(s)

            # select action
            a = torch.multinomial(pi_s_a, 1).item()
            log_prob = torch.log(pi_s_a.squeeze(0)[a])

            # take action
            s_new, r, done, _ = env.step(a)

            # compute delta
            v_s_new = 0
            if not done:
                v_s_new, _ = model.forward(s_new)
                v_s_new = v_s_new.item()
            delta = r + gamma * v_s_new - v_s.item()

            # from another github where the whole thing works:
            # adv = r - v_s.item()
            # loss_a = -log_prob * adv
            # loss_c = torch.nn.functional.smooth_l1_loss(v_s, torch.Tensor([r]).float())

            # compute gradient
            loss_a = - delta * I * log_prob
            loss_c = - delta * v_s
            loss = loss_a + loss_c

            # backprop
            loss.backward()
            optimizer.step()

            I = gamma * I
            s = s_new
            step += 1

            # until s is a terminal state or we used the max steps
            if done:
                break

        if episode % 10 == 0:
            # print("E{0}- Steps:{1} Loss:{2}".format(episode, step, loss))
            pass

        episode_durations.append(step)

    return episode_durations


###

import torch.nn as nn
import torch
from torch import optim
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train(env, model, optimizer, num_episodes, gamma, n_step=1):
    episode_durations = []
    cumulative_reward = []

    # loop for each episode
    for episode in tqdm(range(num_episodes)):

        T = np.inf
        I = 1
        step, t = 0, 0
        s = env.reset()

        states = [s]  # S_0, S_1, ...
        rewards = []  # R_1, R_2, ...

        while True:

            if t < T:
                a, _ = select_action(model, s)
                # take action
                s_, r, done, _ = env.step(a)

                rewards.append(r)
                states.append(s_)

                if done:
                    T = t + 1

            tau = t - n_step + 1

            if tau >= 0:

                # compute G
                G = compute_G_n_step(rewards, gamma, tau, n_step, t, T)

                if tau + n_step < T:
                    with torch.no_grad():
                        v_s_new, _ = model.forward(s_)
                        G += (gamma ** n_step) * v_s_new.item()

                # update model for state s_tau
                state_tau = states[tau]
                v_tau, pi_s_a_tau = model.forward(state_tau)
                log_prob_tau = torch.log(pi_s_a_tau.squeeze(0)[a])

                # compute loss
                delta = G - v_tau
                loss_a = - delta * I * log_prob_tau
                loss_c = - delta * v_tau
                loss = loss_a + loss_c

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print('t:', t, 'v_s_t:',v_s.item(), 'v_s_t+1:', v_s_new.item(),
                #    'tau:', tau, 'G:', G, 'delta:', delta, 'done:', done)

            # update episode and records
            s = s_
            t += 1
            I = gamma * I

            if tau == T - 1:
                cumulative_reward.append(np.sum(rewards))
                break

        if episode % 10 == 0:
            # print("E{0}- Steps:{1} Loss:{2}".format(episode, t, loss))
            pass

        episode_durations.append(t)

    return episode_durations, cumulative_reward


# TODO: check as possible error source
def compute_G_n_step(rewards, gamma, tau, n_step, t, T):
    # print("Alex loss: ", np.sum(gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n_step, T))))
    # print("me loss: ", np.sum(rewards[tau:t + 1] * np.power(gamma, range(len(rewards[tau:t + 1])))))
    # return np.sum(rewards[tau:t + 1] * np.power(gamma, range(len(rewards[tau:t + 1]))))
    return np.sum(gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n_step, T)))


def select_action(model, s):
    with torch.no_grad():
        _, pi_s_a = model.forward(s)
    # select action
    a = torch.multinomial(pi_s_a, 1).item()
    log_prob = torch.log(pi_s_a.squeeze(0)[a])
    return a, log_prob


## EXPERIMENT WITH ONE STEP AGAIN
def train_one_step(env, models, optimizer, num_episodes, gamma, n_step=1, model_names=("actor_cartpole", "v_cartpole")):
    episode_durations = []
    cumulative_reward = []

    actor, critic = models
    opt_actor, opt_critic = optimizer

    # loop for each episode
    for episode in tqdm(range(num_episodes)):

        T = np.inf
        I = 1
        step, t = 0, 0
        s = env.reset()

        states = [s]  # S_0, S_1, ...
        rewards = []  # R_1, R_2, ...

        while True:

            if t < T:
                with torch.no_grad():
                    a, log_prob, pi_s_a = select_action(actor, s)

                # take action
                s_, r, done, _ = env.step(a)

                rewards.append(r)
                states.append(s_)

                if done:
                    T = t + 1

            tau = t - n_step + 1

            if tau >= 0:

                # compute G
                G = compute_G_n_step(rewards, gamma, tau, n_step, t, T)

                if tau + n_step < T:
                    with torch.no_grad():
                        v_s_new = critic(s_)
                        G += (gamma ** n_step) * v_s_new

                # update model for state s_tau
                state_tau = states[tau]
                a, log_prob_tau, pi_s_a_tau = select_action(actor, state_tau)
                v_s_tau = critic(s)

                # For better exploration ?!
                pi_entropy = pi_s_a_tau.entropy().mean()

                actor_loss, critic_loss = calculate_loss(G, log_prob_tau, v_s_tau, pi_entropy, False)

                # backprop
                opt_actor.zero_grad()
                opt_critic.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                opt_actor.step()
                opt_critic.step()

            # update episode and records
            s = s_
            t += 1
            I = gamma * I

            if tau == T - 1:
                cumulative_reward.append(np.sum(rewards))
                break

        if episode % 10 == 0:
            # print("E{0}- Steps:{1} Loss:{2}".format(episode, t, loss))
            pass

        episode_durations.append(t)

    save_models((actor, critic), model_names)

    return episode_durations, cumulative_reward


def compute_G_n_step(rewards, gamma, tau, n_step, t, T):
    # print("Alex loss: ", np.sum(gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n_step, T))))
    # print("me loss: ", np.sum(rewards[tau:t + 1] * np.power(gamma, range(len(rewards[tau:t + 1])))))
    # return np.sum(rewards[tau:t + 1] * np.power(gamma, range(len(rewards[tau:t + 1]))))
    return np.sum(gamma ** (i - tau) * rewards[i] for i in range(tau, min(tau + n_step, T)))
