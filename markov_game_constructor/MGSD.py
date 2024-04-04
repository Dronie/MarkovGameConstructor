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

# PD payoffs
R_pd = 3
T_pd = 4
S_pd = -10
P_pd = 1

# CH payoffs
R_ch = 3
T_ch = 4
S_ch = -10
P_ch = -15

# SH payoffs
R_sh = 4
T_sh = 3
S_sh = -10
P_sh = 1

# ---- PD reward matricies ----
# Define payoff matrices for each player in state type 1
p1_s1 = np.array([[0, 0], [0, P_pd]])
p2_s1 = np.array([[0, 0], [0, P_pd]])

# Define payoff matrices for each player in state type 2
p1_s5 = np.array([[R_pd, R_pd], [R_pd, R_pd]])
p2_s5 = np.array([[R_pd, R_pd], [R_pd, R_pd]])

# Define payoff matrices for each player in state type 3
p1_s4 = np.array([[S_pd, S_pd], [S_pd, S_pd]])
p2_s4 = np.array([[T_pd, T_pd], [T_pd, T_pd]])

# Define payoff matrices for each player in state type 5
p1_s6 = np.array([[T_pd, T_pd], [T_pd, T_pd]])
p2_s6 = np.array([[S_pd, S_pd], [S_pd, S_pd]])

# ---- CH reward matricies ----
# Define payoff matrices for each player in state type 1
p1_s2 = np.array([[0, 0], [0, P_ch]])
p2_s2 = np.array([[0, 0], [0, P_ch]])

# Define payoff matrices for each player in state type 2
p1_s8 = np.array([[R_ch, R_ch], [R_ch, R_ch]])
p2_s8 = np.array([[R_ch, R_ch], [R_ch, R_ch]])

# Define payoff matrices for each player in state type 3
p1_s7 = np.array([[S_ch, S_ch], [S_ch, S_ch]])
p2_s7 = np.array([[T_ch, T_ch], [T_ch, T_ch]])

# Define payoff matrices for each player in state type 5
p1_s9 = np.array([[T_ch, T_ch], [T_ch, T_ch]])
p2_s9 = np.array([[S_ch, S_ch], [S_ch, S_ch]])

# ---- SH reward matricies ----
# Define payoff matrices for each player in state type 1
p1_s3 = np.array([[0, 0], [0, P_sh]])
p2_s3 = np.array([[0, 0], [0, P_sh]])

# Define payoff matrices for each player in state type 2
p1_s11 = np.array([[R_sh, R_sh], [R_sh, R_sh]])
p2_s11 = np.array([[R_sh, R_sh], [R_sh, R_sh]])

# Define payoff matrices for each player in state type 3
p1_s10 = np.array([[S_sh, S_sh], [S_sh, S_sh]])
p2_s10 = np.array([[T_sh, T_sh], [T_sh, T_sh]])

# Define payoff matrices for each player in state type 5
p1_s12 = np.array([[T_sh, T_sh], [T_sh, T_sh]])
p2_s12 = np.array([[S_sh, S_sh], [S_sh, S_sh]])

# ---- define states ----
# transitory states
state_1 = mg.PayoffMatrix(p1_s1, p2_s1)
state_2 = mg.PayoffMatrix(p1_s2, p2_s2) 
state_3 = mg.PayoffMatrix(p1_s3, p2_s3)

# R states
state_5 = mg.PayoffMatrix(p1_s5, p2_s5)
state_8 = mg.PayoffMatrix(p1_s8, p2_s8)
state_11 = mg.PayoffMatrix(p1_s11, p2_s11)

# S, T states
state_4 = mg.PayoffMatrix(p1_s4, p2_s4)
state_7 = mg.PayoffMatrix(p1_s7, p2_s7)
state_10 = mg.PayoffMatrix(p1_s10, p2_s10)

# T, S states
state_6 = mg.PayoffMatrix(p1_s6, p2_s6)
state_9 = mg.PayoffMatrix(p1_s9, p2_s9)
state_12 = mg.PayoffMatrix(p1_s12, p2_s12)

game = mg.MarkovGame(state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10, state_11, state_12)

# ---- state transitions ----

# state 1
game.add_transition(state_from=1, state_to=2, action=(1, 1), prob=1)
game.add_transition(state_from=1, state_to=4, action=(0, 1), prob=1)
game.add_transition(state_from=1, state_to=5, action=(0, 0), prob=1)
game.add_transition(state_from=1, state_to=6, action=(1, 0), prob=1)

# state 2
game.add_transition(state_from=2, state_to=3, action=(1, 1), prob=1)
game.add_transition(state_from=2, state_to=7, action=(0, 1), prob=1)
game.add_transition(state_from=2, state_to=8, action=(0, 0), prob=1)
game.add_transition(state_from=2, state_to=9, action=(1, 0), prob=1)

# state 3
game.add_transition(state_from=3, state_to=1, action=(1, 1), prob=1)
game.add_transition(state_from=3, state_to=10, action=(0, 1), prob=1)
game.add_transition(state_from=3, state_to=11, action=(0, 0), prob=1)
game.add_transition(state_from=3, state_to=12, action=(1, 0), prob=1)

# state 4
game.add_transition(state_from=4, state_to=1, action=(1, 0), prob=1)
game.add_transition(state_from=4, state_to=1, action=(0, 0), prob=1)

# state 5
game.add_transition(state_from=5, state_to=1, action=(1, 1), prob=1)

# state 6
game.add_transition(state_from=6, state_to=1, action=(0, 1), prob=1)
game.add_transition(state_from=6, state_to=1, action=(0, 0), prob=1)

# state 7
game.add_transition(state_from=7, state_to=2, action=(1, 0), prob=1)
game.add_transition(state_from=7, state_to=2, action=(0, 0), prob=1)

# state 8
game.add_transition(state_from=8, state_to=2, action=(1, 1), prob=1)

# state 9
game.add_transition(state_from=9, state_to=2, action=(0, 1), prob=1)
game.add_transition(state_from=9, state_to=2, action=(0, 0), prob=1)

# state 10
game.add_transition(state_from=10, state_to=3, action=(1, 0), prob=1)
game.add_transition(state_from=10, state_to=3, action=(0, 0), prob=1)

# state 11
game.add_transition(state_from=11, state_to=3, action=(1, 1), prob=1)

# state 12
game.add_transition(state_from=12, state_to=3, action=(0, 1), prob=1)
game.add_transition(state_from=12, state_to=3, action=(0, 0), prob=1)

print(game.get_transitions())

q_tables = game.q_learn(gamma = 0.99, alpha= 0.99, episodes= 1000, T = 1000, s0 = 1, eps = 0.25)

print('===== Standard Game =====')
print("Player 1 Q Table:")
print(f'State 1:{q_tables[0]["1"]}')
print(f'State 2:{q_tables[0]["2"]}')
print(f'State 6:{q_tables[0]["3"]}')

print("\nPlayer 2 Q Table:")
print(f'State 1:{q_tables[1]["1"]}')
print(f'State 2:{q_tables[1]["2"]}')
print(f'State 6:{q_tables[1]["3"]}')
print('=========================\n\n')