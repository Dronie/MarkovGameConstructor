import utils as mg
import numpy as np
import copy as cp

def get_altruistic_game(state, alpha):
    state = cp.copy(state)
    altruistic_payoffs = state.get_payoffs()
    for index, i in enumerate(altruistic_payoffs):
        for jndex, j in enumerate(i):
            altruistic_payoffs[index][jndex] = j + alpha * sum(j)
    return cp.copy(mg.PayoffMatrix(altruistic_payoffs.T[0], altruistic_payoffs.T[1]))


def get_altruistic_game_automatic(state, alpha):
    state = cp.copy(state)
    altruistic_payoffs = state.get_payoffs()
    for index, i in enumerate(altruistic_payoffs):
        R = altruistic_payoffs[0][0][index]
        S = altruistic_payoffs[1][0][index]
        T = altruistic_payoffs[0][1][index]
        P = altruistic_payoffs[1][1][index]
        if (2 * R - S - T) == 0:
            alpha = 0
        else:
            alpha = (T - R)/(2 * R - S - T)
        for jndex, j in enumerate(i):
            altruistic_payoffs[index][jndex] = j + alpha * sum(j)
    return cp.copy(mg.PayoffMatrix(altruistic_payoffs.T[0], altruistic_payoffs.T[1]))

R = 3
T = 4
S = -10
P = 1

# Define payoff matrices for each player in state type 1
p1_s1 = np.array([[0, 0], [0, 0]])
p2_s1 = np.array([[0, 0], [0, 0]])

# Define payoff matrices for each player in state type 2
p1_s2 = np.array([[R, R], [R, R]])
p2_s2 = np.array([[R, R], [R, R]])

# Define payoff matrices for each player in state type 3
p1_s3 = np.array([[S, S], [S, S]])
p2_s3 = np.array([[T, T], [T, T]])

# Define payoff matrices for each player in state type 4
p1_s4 = np.array([[P, P], [P, P]])
p2_s4 = np.array([[P, P], [P, P]])

# Define payoff matrices for each player in state type 5
p1_s5 = np.array([[T, T], [T, T]])
p2_s5 = np.array([[S, S], [S, S]])

# ---- define states ----
# 0 reward (transitory) states
state_1 = mg.PayoffMatrix(p1_s1, p2_s1)
state_2 = mg.PayoffMatrix(p1_s1, p2_s1) 
state_6 = mg.PayoffMatrix(p1_s1, p2_s1)

# R state
state_7 = mg.PayoffMatrix(p1_s2, p2_s2)

# S, T states
state_3 = mg.PayoffMatrix(p1_s3, p2_s3)
state_8 = mg.PayoffMatrix(p1_s3, p2_s3)

# T, S states
state_5 = mg.PayoffMatrix(p1_s5, p2_s5)
state_9 = mg.PayoffMatrix(p1_s5, p2_s5)

# P state
state_4 = mg.PayoffMatrix(p1_s4, p2_s4)


game = mg.MarkovGame(state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9)

# ---- state transitions ----

# state 1
game.add_transition(state_from=1, state_to=2, action=(0, 0), prob=1)
game.add_transition(state_from=1, state_to=3, action=(0, 1), prob=1)
game.add_transition(state_from=1, state_to=4, action=(1, 1), prob=1)
game.add_transition(state_from=1, state_to=5, action=(1, 0), prob=1)

# state 2
game.add_transition(state_from=2, state_to=1, action=(1, 1), prob=1)
game.add_transition(state_from=2, state_to=6, action=(0, 0), prob=1)

# state 3
game.add_transition(state_from=3, state_to=1, action=(0, 0), prob=1)
game.add_transition(state_from=3, state_to=1, action=(1, 0), prob=1)

# state 4
game.add_transition(state_from=4, state_to=1, action=(0, 0), prob=1)

# state 5
game.add_transition(state_from=5, state_to=1, action=(0, 0), prob=1)
game.add_transition(state_from=5, state_to=1, action=(0, 1), prob=1)

# state 6
game.add_transition(state_from=6, state_to=2, action=(1, 1), prob=1)
game.add_transition(state_from=6, state_to=7, action=(0, 0), prob=1)
game.add_transition(state_from=6, state_to=8, action=(0, 1), prob=1)
game.add_transition(state_from=6, state_to=9, action=(1, 0), prob=1)

# state 7
game.add_transition(state_from=7, state_to=6, action=(1, 1), prob=1)

# state 8
game.add_transition(state_from=8, state_to=6, action=(1, 0), prob=1)
game.add_transition(state_from=8, state_to=6, action=(0, 0), prob=1)

# state 9
game.add_transition(state_from=9, state_to=6, action=(0, 1), prob=1)
game.add_transition(state_from=9, state_to=6, action=(0, 0), prob=1)

q_tables = game.q_learn(gamma = 0.99, alpha= 0.99, episodes= 1000, T = 1000, s0 = 1, eps = 0.25)

print('===== Standard Game =====')
print("Player 1 Q Table:")
print(f'State 1:{q_tables[0]["1"]}')
print(f'State 2:{q_tables[0]["2"]}')
print(f'State 6:{q_tables[0]["6"]}')

print("\nPlayer 2 Q Table:")
print(f'State 1:{q_tables[1]["1"]}')
print(f'State 2:{q_tables[1]["2"]}')
print(f'State 6:{q_tables[1]["6"]}')
print('=========================\n\n')