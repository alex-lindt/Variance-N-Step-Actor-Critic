import random
import gym
import os
import argparse
import torch
import numpy as np
from src import train, helpers


def start_training(environment_name, model_type, n_step, max_episodes, device):
    env = gym.make(environment_name)

    # SETTINGS
    learn_rates = (7e-4, 7e-4)
    num_episodes = max_episodes
    gamma = 0.99
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # INIT
    models, optimizer = helpers.init_model(model_type, env, learn_rates, device, n_hidden=(64, 64))
    # TRAIN
    episode_durations, cumulative_reward, actor_losses, critic_losses, running_average = train.train_actor_critic(env,
                                                                                                                  environment_name,
                                                                                                                  models,
                                                                                                                  optimizer,
                                                                                                                  num_episodes,
                                                                                                                  gamma,
                                                                                                                  n_step,
                                                                                                                  model_type,
                                                                                                                  device)
    # PLOT
    helpers.plot_results(episode_durations, running_average, cumulative_reward, actor_losses, critic_losses, n_step,
                         environment_name,
                         model_type)


def run_experiments(max_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environments = ["CartPole-v0", "LunarLander-v2", "MountainCar-v0", "Taxi-v2"]  #
    for environment in environments:
        model_types = ["Advantage", "Q", "Reinforce"]  # "Reinforce"
        for model_type in model_types:
            if model_type == "Reinforce":
                print("\n --> Reinforce is automatically all N steps, so no n-step variations needed. \n")
                n_steps = ["Monte Carlo"]
            else:
                n_steps = [1, 2, 4, 8]
            for n_step in n_steps:
                print('**********************************************************************')
                print("Environment:", environment, 'Model type:', model_type, 'N_step:', n_step)
                print('**********************************************************************')
                start_training(environment, model_type, n_step, max_episodes, device)


if __name__ == "__main__":
    # MAKE FOLDERS (if dont exist)
    os.makedirs('Models', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    os.makedirs('Results/numpy', exist_ok=True)
    os.makedirs('Results/Plots', exist_ok=True)

    # PARS ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_experiments', default=False, type=bool,
                        help='whether experiments should be run')
    parser.add_argument('--visualise', default=False, type=bool,
                        help='whether trained models should be loaded and visualised')
    parser.add_argument('--model_type', default='Advantage', type=str,
                        help='Advantage, Q or Reinforce')
    parser.add_argument('--n_step', default='1', type=str,
                        help='"1", "2", "4", "8" OR "Monte Carlo" in combination with Reinforce')
    parser.add_argument('--environment', default='CartPole-v0', type=str,
                        help='"CartPole-v0", "MountainCar-v0", "LunarLander-v2", "Taxi-v2"')
    parser.add_argument('--max_episodes', default=1000, type=int,
                        help='For testing purposes sometimes you want to lower the number of episodes')
    ARGS = parser.parse_args()

    if ARGS.run_experiments == ARGS.visualise:
        print('***** You can not both run experiments and visualise in one run. You have to choose between the two options. \n \
                EITHER: \n \
                --run_experiments=True \n \
                OR eg: \n \
                --visualise=True \n \
                --model_type=Advantage \n \
                --n_step=1 \n \
                --environment=CartPole-v0')
        quit()

    if ARGS.run_experiments:
        print('Running experiments...')
        run_experiments(ARGS.max_episodes)
    elif ARGS.visualise:
        print("Visualising trained model's performance.")
        model_path = "Models_Claartje/{}/{}_n_step{}_{}_actor.pth".format(ARGS.environment, ARGS.model_type,
                                                                          ARGS.n_step, ARGS.environment)
        helpers.visualize_performance(ARGS.environment, model_path, device)
