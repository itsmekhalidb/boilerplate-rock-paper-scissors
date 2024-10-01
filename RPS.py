import numpy as np
import random

Q = np.zeros((3,3))

STATES = ["R", "P", "S"]
ROCK = 0
PAPER = 1
SCISSORS = 2

def encode_Move(letter):
    if letter == "R":
        return ROCK
    elif letter == "P":
        return PAPER
    else:
        return SCISSORS

def get_Reward(opp, action):
    if opp == action:
        return 0
    elif STATES[opp] == "R" and STATES[action] == "S":
        return -1
    elif STATES[opp] == "S" and STATES[action] == "P":
        return -1
    elif STATES[opp] == "P" and STATES[action] == "R":
        return -1
    else:
        return 1

def q_Learning(opp):

    EPISODES = 1500

    LEARNING_RATE = 0.05
    GAMMA = 0.96

    EPSILON = 0.9

    for e in range(EPISODES):
        reward = 0
        for x in range(len(STATES)):
            if np.random.uniform(0, 1) < EPSILON:
                action = encode_Move(random.choice(STATES))
            else:
                ind = np.unravel_index(np.argmax(Q[opp, :]), Q.shape)
                action = encode_Move(STATES[ind[1]])
            
            reward += get_Reward(opp, action)
            if opp == 2:
                next_state = 0
            else:
                next_state = opp + 1
            
            Q[opp, action] = Q[opp, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[opp, action])
            EPSILON -= 0.001
    ind = np.unravel_index(np.argmax(Q[opp, :]), Q.shape)
    action = encode_Move(STATES[ind[1]])
    return action

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)
    action = q_Learning(encode_Move(prev_play))
    guess = STATES[action]
    return guess

# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

# def player(prev_play, opponent_history=[]):
#     opponent_history.append(prev_play)

#     guess = "R"
#     if len(opponent_history) > 2:
#         guess = opponent_history[-2]

#     return guess
