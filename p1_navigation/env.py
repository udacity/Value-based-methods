from unityagents import UnityEnvironment
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys 
from agent import Agent
from qn import QNetwork
import time

TRAIN = int(sys.argv[1])
print("Train : {}".format(TRAIN))

time.sleep(1)
env = UnityEnvironment(file_name="Banana.app", base_port=64738, worker_id=2, seed=1)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
    

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

if TRAIN == 1: 
    print("TRAINNG" )
    n_episodes=1600
    max_t=1000
    eps_start=1.0
    eps_end=0.01
    eps_decay=0.998
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_ = env.step(action)[brain_name]
            next_state = env_.vector_observations[0]   # get the next state
            reward = env_.rewards[0]                   # get the reward
            done = env_.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=17.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break


    from matplotlib.pylab import plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig('scores.png', dpi=fig.dpi)

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    
    env.close()

else: 
    print("Smart Agent")
    # Uncooment this for smart agent visuazliation 
    import time
    eps = 0.8
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    for i in range(3):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        for j in range(200):
            action = agent.act(state, eps)
            env_ = env.step(action)[brain_name]
            next_state = env_.vector_observations[0]   # get the next state
            reward = env_.rewards[0]                   # get the reward
            done = env_.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            time.sleep(0.1)
            if done:
                break 





