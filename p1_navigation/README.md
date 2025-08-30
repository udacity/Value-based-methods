# Deep Q-Learning Agent for Unity Banana Environment

This project implements a **Deep Q-Network (DQN)** to solve the [Unity ML-Agents Banana environment](https://github.com/Unity-Technologies/ml-agents).  
The agent learns to navigate the environment, collect yellow bananas (+1 reward), and avoid blue bananas (-1 reward).

---

## 📂 Project Structure

```
.
├── Navigation.ipynb      # Main notebook to interact with the Unity environment
├── model.py              # Defines the QNetwork (neural network model)
├── dqn_agent.py          # Defines the Agent, replay buffer, and learning updates
├── checkpoint.pth        # Saved trained model weights (after training)
```

---

## 🧠 Environment Details

- **State space size**: 37  
  (includes agent's velocity and ray-based perception of objects in front of it)
- **Action space size**: 4 (discrete)
  - `0` → Walk forward  
  - `1` → Walk backward  
  - `2` → Turn left  
  - `3` → Turn right  
- **Rewards**:  
  - `+1` for collecting a yellow banana  
  - `-1` for collecting a blue banana  

The task is episodic, and the environment is considered solved when the agent gets an **average score of +13 over 100 consecutive episodes**.

---

## 🏗️ Q-Network (model.py)

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

- Input: state vector (size 37)  
- Hidden layers: 2 fully-connected layers with 64 units each  
- Output: Q-values for each action (size 4)  

---

## 🤖 Agent (dqn_agent.py)

The agent uses:
- **Replay Buffer** (stores past experiences for learning)
- **Epsilon-Greedy Policy** (exploration vs exploitation)
- **Target Network** (stabilizes training)
- **Soft Updates** (controlled by `tau`)
- **Adam Optimizer** with learning rate `5e-4`

Key hyperparameters:
```python
BUFFER_SIZE = 1e5   # replay buffer size
BATCH_SIZE = 64     # minibatch size
GAMMA = 0.99        # discount factor
TAU = 1e-3          # soft update factor
LR = 5e-4           # learning rate
UPDATE_EVERY = 4    # update frequency
```

---

## 🚀 Training (Navigation.ipynb)

Run the training loop using:
```python
scores = dqn(n_episodes=2000)
```

- Episodes: up to 2000  
- Early stopping: when average score ≥ 13 over 100 episodes  
- Best checkpoint is saved automatically as `checkpoint.pth`  

Plot training results:
```python
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
```

---

## 🎥 Watching the Trained Agent

After training, watch the agent perform:
```python
watch_agent(n_episodes=5, render=True)
```

This will open a Unity window where you can see the agent collecting bananas!

---

## 📦 Installation

1. Clone the repo & install dependencies:
```bash
git clone https://github.com/your-username/dqn-banana.git
cd dqn-banana
pip install -r requirements.txt
```

2. Download the Unity Banana environment from Udacity:  
   - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana.app.zip)  
   - [Windows (x86)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana_Windows_x86.zip)  
   - [Windows (x86_64)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana_Windows_x86_64.zip)  
   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana_Linux.zip)  

3. Extract and update the path in the notebook:
```python
env = UnityEnvironment(file_name="Banana.app")
```

---

## 🏆 Results

- The agent successfully solves the Banana environment, reaching an average score ≥ 13.  
- Saved model weights are available in `checkpoint.pth`.

---

## 📖 References

- Udacity Deep Reinforcement Learning Nanodegree  
- Unity ML-Agents Toolkit  
- [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (Mnih et al., 2015)  

---

## ✍️ Author
Krushna Thakkar  
