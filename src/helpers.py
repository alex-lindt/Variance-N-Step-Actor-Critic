import time

import pickle
import torch
from torch import optim

import numpy as np

import matplotlib.pyplot as plt
import gym

from .actor_critic import ValueFunction, Actor, QValueFunction


def smooth(x, N):
    """

    :param x:
    :param N:
    :return:
    """  #

    # TODO: maybe misleading for some plots...
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# TODO: implement a def plot_results function ...
def plot_results(episode_durations, running_average, reward_across_episodes, actor_losses, critic_losses, n_step,
                 environment_name,
                 model_types):
    """
    :param episode_durations:
    :param reward_across_episodes:
    :return:
    """

    fig = plt.figure(figsize=(15, 6), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.suptitle('{} Actor-Critic in {} at n: {}'.format(model_types, environment_name, n_step), fontsize=24)

    ax1.plot(running_average, label="Running average", color="r")
    ax2.plot(reward_across_episodes, label="Average", color="g")

    ax1.legend()
    ax2.legend()

    result_dict = {'environment': environment_name,
                   'n_step': n_step,
                   'running_average': running_average,
                   'rewards_across_episodes': reward_across_episodes,
                   'actor_losses': actor_losses,
                   'critic_losses': critic_losses,
                   'model_type': model_types
                   }

    # save results in npy for altering plots later on etc.
    # np.save("Results/numpy/{}_n_step{}_{}.npy".format(environment_name, n_step, model_types), result_dict)
    with open("Results/numpy/{}_n_step{}_{}.pickle".format(environment_name, n_step, model_types), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save plt
    plt.savefig("Results/Plots/{}_n_step{}_{}.png".format(environment_name, n_step, model_types))
    # plt.show()


def init_model(model_type, env, learn_rates, device, n_hidden=(128, 256)):
    """

    :param model_type:
    :param env:
    :param learn_rate:
    :param device:
    :param n_hidden:
    :return:
    """
    lr_actor, lr_critic = learn_rates
    # print("Observation and action space: ")
    # print(env.observation_space.n)
    # quit()
    # print(env.action_space)
    n_state_features = env.observation_space.shape[0]
    # n_state_features = env.observation_space.n
    n_actions = env.action_space.n

    if model_type == "Advantage" or model_type == "Reinforce":
        actor = Actor(n_state_features, n_actions, n_hidden, device).to(device)
        critic = ValueFunction(n_state_features, n_hidden, device).to(device)
        # print("Actor Network: ")
        # print(actor)
        # print("Critic Network: ")
        # print(critic)
        opt_actor = optim.Adam(actor.parameters(), lr_actor)
        opt_critic = optim.Adam(critic.parameters(), lr_critic)

        return (actor, critic), (opt_actor, opt_critic)

    elif model_type == "Q":
        actor = Actor(n_state_features, n_actions, n_hidden, device).to(device)
        critic = QValueFunction(n_state_features, n_actions, n_hidden, device).to(device)
        # print("Actor Network: ")
        # print(actor)
        # print("Critic Network: ")
        # print(critic)
        opt_actor = optim.Adam(actor.parameters(), lr_actor)
        opt_critic = optim.Adam(critic.parameters(), lr_critic)

        return (actor, critic), (opt_actor, opt_critic)

    else:
        print("No matching procedure")


def save_models(models, model_type, n_step, environment_name):
    """

    :param models:
    :param model_names:
    :return:
    """
    # path = "models/{}.pth"
    filename = "Models/{}_n_step{}_{}".format(model_type, n_step, environment_name)
    actor, critic = models
    torch.save(actor, filename + '_actor.pth')
    torch.save(critic, filename + '_critic.pth')


def visualize_performance(environment_name, model_path, device):
    env_to_wrap = gym.make(environment_name)
    env = gym.wrappers.Monitor(env_to_wrap, 'Videos/{}/{}/'.format(environment_name, model_path.split('/')[-1]),
                               force=True, video_callable=lambda episode_id: True)
    actor = torch.load(model_path)

    rewards = []
    for episode in np.arange(5):
        s = env.reset()
        episode_reward = 0
        while True:
            env.render()
            pi_s_a = actor(s)
            a = pi_s_a.sample().item()
            s_, r, done, _ = env.step(a)
            episode_reward += r

            s = s_
            if done:
                rewards.append(episode_reward)
                print(episode_reward)
                break

    env.close()
    env_to_wrap.close()

    f = open('Videos/{}/{}/rewards.txt'.format(environment_name, model_path.split('/')[-1]), "w+")
    for i, r in enumerate(rewards):
        f.write(str(r) + "\n")
    f.write("Average reward over episodes: " + str(np.mean(rewards)))
    f.close()

    print("Average reward over episodes: ", np.mean(rewards))
    plt.plot(rewards)


'''
Now Q-learning helpers
'''


def compute_q_val(model, state, action):
    if isinstance(action, int):
        pred = model(state)
        return pred[action].unsqueeze(0), pred
    else:
        pred = model(state)
        return torch.gather(pred, 1, action[:, None]).reshape(-1), pred

# def select_q_from_action():
