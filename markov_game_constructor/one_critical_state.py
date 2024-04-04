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

# Define payoff matrices for each player in state 1
p1_s1 = np.array([[0, 0], [0, 0]])
p2_s1 = np.array([[0, 0], [0, 0]])

# Define payoff matrices for each player in state 2
p1_s2 = np.array([[R, R], [R, R]])
p2_s2 = np.array([[R, R], [R, R]])

# Define payoff matrices for each player in state 3
p1_s3 = np.array([[S, S], [S, S]])
p2_s3 = np.array([[T, T], [T, T]])

# Define payoff matrices for each player in state 4
p1_s4 = np.array([[P, P], [P, P]])
p2_s4 = np.array([[P, P], [P, P]])

# Define payoff matrices for each player in state 5
p1_s5 = np.array([[T, T], [T, T]])
p2_s5 = np.array([[S, S], [S, S]])

state_1 = mg.PayoffMatrix(p1_s1, p2_s1) # 0
state_2 = mg.PayoffMatrix(p1_s2, p2_s2) # R
state_3 = mg.PayoffMatrix(p1_s3, p2_s3) # S, T
state_4 = mg.PayoffMatrix(p1_s4, p2_s4) # P
state_5 = mg.PayoffMatrix(p1_s5, p2_s5) # T, S

# Construct Markov Game called 'game' with previously defined states 1-4
game = mg.MarkovGame(state_1, state_2, state_3, state_4, state_5)

alpha = 1

game_a = mg.MarkovGame(get_altruistic_game(state_1, alpha), get_altruistic_game(state_2, alpha), get_altruistic_game(state_3, alpha), get_altruistic_game(state_4, alpha), get_altruistic_game(state_5, alpha))

# Describe transitions for Markov Game 'game'
game.add_transition(state_from=1, state_to=2, action=(0, 0), prob=1) # Add transition: s1 -> s2 via action (0, 0) with prob = 1
game.add_transition(state_from=1, state_to=3, action=(0, 1), prob=1) # Add transition: s1 -> s3 via action (0, 1) with prob = 1
game.add_transition(state_from=1, state_to=4, action=(1, 1), prob=1) # Add transition: s1 -> s4 via action (1, 1) with prob = 1
game.add_transition(state_from=1, state_to=5, action=(1, 0), prob=1) # Add transition: s1 -> s5 via action (1, 0) with prob = 1

game.add_transition(state_from=2, state_to=1, action=(1, 1), prob=1) # Add transition: s2 -> s1 via action (1, 1) with prob = 1

game.add_transition(state_from=3, state_to=1, action=(0, 0), prob=1) # Add transition: s3 -> s1 via action (0, 0) with prob = 1
game.add_transition(state_from=3, state_to=1, action=(1, 0), prob=1) # Add transition: s3 -> s1 via action (1, 0) with prob = 1

game.add_transition(state_from=4, state_to=1, action=(0, 0), prob=1) # Add transition: s4 -> s1 via action (0, 0) with prob = 1

game.add_transition(state_from=5, state_to=1, action=(0, 0), prob=1) # Add transition: s5 -> s1 via action (1, 1) with prob = 1
game.add_transition(state_from=5, state_to=1, action=(0, 1), prob=1) # Add transition: s5 -> s1 via action (1, 1) with prob = 1


# Describe transitions for Markov Game 'game_a'
game_a.add_transition(state_from=1, state_to=2, action=(0, 0), prob=1) # Add transition: s1 -> s2 via action (0, 0) with prob = 1
game_a.add_transition(state_from=1, state_to=3, action=(0, 1), prob=1) # Add transition: s1 -> s3 via action (0, 1) with prob = 1
game_a.add_transition(state_from=1, state_to=4, action=(1, 1), prob=1) # Add transition: s1 -> s4 via action (1, 1) with prob = 1
game_a.add_transition(state_from=1, state_to=5, action=(1, 0), prob=1) # Add transition: s1 -> s5 via action (1, 0) with prob = 1

game_a.add_transition(state_from=2, state_to=1, action=(1, 1), prob=1) # Add transition: s2 -> s1 via action (1, 1) with prob = 1

game_a.add_transition(state_from=3, state_to=1, action=(0, 0), prob=1) # Add transition: s3 -> s1 via action (0, 0) with prob = 1
game_a.add_transition(state_from=3, state_to=1, action=(1, 0), prob=1) # Add transition: s3 -> s1 via action (1, 0) with prob = 1

game_a.add_transition(state_from=4, state_to=1, action=(0, 0), prob=1) # Add transition: s4 -> s1 via action (0, 0) with prob = 1

game_a.add_transition(state_from=5, state_to=1, action=(0, 0), prob=1) # Add transition: s5 -> s1 via action (1, 1) with prob = 1
game_a.add_transition(state_from=5, state_to=1, action=(0, 1), prob=1) # Add transition: s5 -> s1 via action (1, 1) with prob = 1


q_tables = game.q_learn(gamma = 0.99, alpha= 0.99, episodes= 1000, T = 1000, s0 = 1, eps = 0.25)
q_tables_a = game_a.q_learn(gamma = 0.99, alpha= 0.99, episodes= 1000, T = 1000, s0 = 1, eps = 0.25)


print('===== Standard Game =====')
print("Player 1 Q Table:")
print(q_tables[0])

print("\nPlayer 2 Q Table:")
print(q_tables[1])
print('=========================\n\n')

print('===== Altruistic Game =====')
print("Player 1 Q Table:")
print(q_tables_a[0])

print("\nPlayer 2 Q Table:")
print(q_tables_a[1])
print('===========================')