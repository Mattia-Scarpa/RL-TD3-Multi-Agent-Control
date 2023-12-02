# RL for Multi Agent Environment
The main goal of this project is to develop a multi-agent system to a tennis environment 

### Introduction

This project is solving the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) is added, to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Traiining result

In this environment it has been implemented a TD3 algorithm that will be shown to well perform for its task, solving the environment in 873 episodes. The training it was continued up to 1000 episodes were it was possible to observe a interesting curve of improvement.

### Instructions

To set up your python environment to run the code in this repository, follow the instructions below.

The project has been developed exploiting a dedicated conda environment. In order to run it correctly it is strongly suggested to launch the command

'./install.sh'

to install the correct environment with all the required package.

The project is built on:

- test_continuous_control.py -> The main file, the executable to run to see the project
- agent.py -> Agent Class
- model.py -> the NN model for the TD3 algorithm
- checkpoint_actor.pth -> the weights of the best policy model found during training
- checkpoint_critic.pth -> the weights of the best action-value function found during training
- report.md -> Project report of the overall implementation