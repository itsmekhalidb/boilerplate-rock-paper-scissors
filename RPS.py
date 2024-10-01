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

def decode_Move(number):
    return STATES[number]

def get_Reward(opp, action):
    if opp == action:
        return 0
    elif (opp == ROCK and action == SCISSORS) or \
         (opp == SCISSORS and action == PAPER) or \
         (opp == PAPER and action == ROCK):
        return -1
    else:
        return 1

def get_next_state(opp):
    return (opp + 1) % 3

def q_Learning(opp):
    EPISODES = 100
    LEARNING_RATE = .95
    GAMMA = 0.96
    EPSILON = 0.9

    for e in range(EPISODES):
        total_reward = 0
        
        for x in range(len(STATES)):
            if np.random.uniform(0, 1) < EPSILON:
                action = random.choice([ROCK, PAPER, SCISSORS])
            else:
                action = np.argmax(Q[opp, :])
            
            reward = get_Reward(opp, action)
            total_reward += reward

            next_state = get_next_state(opp)
            
            Q[opp, action] = Q[opp, action] + LEARNING_RATE * (total_reward + GAMMA * np.max(Q[next_state, :]) - Q[opp, action])
            EPSILON = max(0.1, EPSILON * 0.99)

    action = np.argmax(Q[opp, :])
    return action

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)
    action = q_Learning(encode_Move(prev_play))
    guess = decode_Move(action)
    return guess