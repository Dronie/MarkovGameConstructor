import numpy as np
import copy as cp
import random
from typing import Union, List, Tuple
from numbers import Number
from itertools import product
import fnmatch


class PlayerMatrix():
    ''' 
    Player Matrix is a 2D tensor collecting the utilities for every possible
    strategy profile for a single agent.
    Args:
        num_actions (default: 2) : Sets the action space for the respective agent

    Methods:
        get_payoffs: returns the payoff matrix as a nested numpy array
        set_payoffs: sets the payoff values for player matrix
        reset_payoffs: sets all payoffs to zero
    '''
    def __init__(self, num_actions: int = 2, initial_matrix: Union[np.ndarray, None] = None) -> None:
        self.num_actions = num_actions
        
        if list(initial_matrix) == None:
            self.matrix = self.reset_payoffs()
        else:
            if isinstance(initial_matrix,(list, np.ndarray)):
                self.matrix = cp.copy(initial_matrix)
            else:
                raise Exception("Specified initial matrix not of type \'list\' or \'np.ndarray\'")
        
        self.action_set = set()
        for i in range(self.num_actions):
            self.action_set.add(i)
    
    def get_payoffs(self) -> np.ndarray:
        """
        returns the payoff matrix

        Returns:
            np.ndarray: _description_
        """
        return cp.copy(self.matrix)
    
    def set_payoffs(
        self,
        input_matrix: Union[List[None], np.ndarray] = [None], 
        loc: Union[None, Tuple[int]] = None, 
        new_value: Union[int, float] = None
    ) -> None:
        """
        sets the payoff values for player matrix

        Args:
            input_matrix (Union[List[None], np.ndarray], optional): _description_. Defaults to [None].
            loc (Union[None, Tuple[int]], optional): _description_. Defaults to None.
            new_value (Union[int, float], optional): _description_. Defaults to None.

        """
        if None not in input_matrix:
            if isinstance(input_matrix, (list, np.ndarray)):
                self.matrix = cp.copy(input_matrix)
            else:
                raise Exception("Specified input matrix not of type \'list\' or \'np.ndarray\'")
        else:
            if loc != None:
                if isinstance(loc, tuple):
                    if isinstance(new_value, (int, float)):
                        self.matrix[loc] = cp.copy(new_value)
                    else:
                        raise Exception("Specified value not of type \'int\' or \'float\'")
                else:
                    raise Exception("Specified input location not a tuple")

    def reset_payoffs(self) -> np.ndarray:
        """
        sets all payoffs to zero

        Returns:
            np.ndarray: _description_
        """
        self.matrix = cp.copy(np.ones(self.num_actions * self.num_actions).reshape(self.num_actions, self.num_actions))
        return cp.copy(self.matrix)

class PayoffMatrix:
    '''
    Payoff Matrix is a collection of PlayerMatrix's.
    Automatically handles the payoff matrix for a state.\n
    ARGS:
     - *player_matrices (PlayerMatrix or numpy.ndarray) : input for an unspecified number of player matrices (no implementation for >2 yet)
     - num_players (int) : the number of players in the game (maybe redundant if given the above)

    METHODS:
     - get_payoffs : returns the payoff matrix as an numpy.ndarray
     - set_payoffs : set the payoffs for one or more player matrices
     - get_social_welfare : return the social welfare of all profiles (or a specific one based on loc)
     - get_social_optimum : 
     - find_nash : 
    '''

    def __init__(self, *player_matrices, num_players: Union[None, int] = None):
        if num_players == None and len(player_matrices) >= 2:
            self.num_players = len(player_matrices)
        elif isinstance(num_players, int) and num_players >= 2:
            self.num_players = num_players
        else:
            raise Exception("Number of players cannot be < 2")
        
        self.payoffs = [None] # needs to be iterable for check in self._init_payoff_matrix()
        
        num_actions = None
        
        self.player_matrices = {}
        for i, player_matrix in enumerate(player_matrices):
            if not isinstance(player_matrix, (PlayerMatrix, np.ndarray)):
                raise Exception("All arguments (other than num_players) MUST be either of type \'PlayerMatrix\' or numpy.ndarray")
            if isinstance(player_matrix, PlayerMatrix):
                if num_actions == None:
                    num_actions = player_matrix.num_actions
                else:
                    if num_actions != player_matrix.num_actions:
                        raise Exception("Action set cardinality not equal amounst all players.\nSupport for different action sets per player not implemented!")
                self.player_matrices[f"Player {i + 1}"] = cp.copy(player_matrix)
            elif isinstance(player_matrix, np.ndarray):
                if num_actions == None:
                    num_actions = len(player_matrix)
                elif num_actions != len(player_matrix):
                    raise Exception("Action set cardinality not equal amounst all players.\nSupport for different action sets per player not implemented!")
                self.player_matrices[f"Player {i + 1}"] = PlayerMatrix(num_actions=num_actions, initial_matrix=player_matrix)
        self.num_actions = num_actions
        pm_action_sets = [list(self.player_matrices[i].action_set) for i in self.player_matrices]
        self.action_set = set(product(*pm_action_sets))

        self._init_payoff_matrix(self.player_matrices)
        self.values = np.zeros_like(self.payoffs)
    
    def __str__(self) -> str:
        string = '---------------------\n'
        for i in range(len(self.payoffs)):
            for j in range(len(self.payoffs[i])):
                string += '| '
                for k in range(len(self.payoffs[i][j])):
                    string += f'{str(self.payoffs[i,j,k])} '
            string += '|\n---------------------\n'

        return string

    def _init_payoff_matrix(self, player_matrices: Tuple[PlayerMatrix]) -> None:
        """
        Method to handle the payoff matrix initialization at object creation
        based on the PlayerMatrix objects given at PayoffMatrix call
        (support for >2 players not yet implemented)

        Args:
            player_matrices (_type_): _description_
            
        """
        for player_matrix in player_matrices:
            matrix = player_matrices[player_matrix].get_payoffs()
            if None in self.payoffs:
                self.payoffs = cp.copy(matrix)
                continue
            else:
                if self.num_players == 2:
                    '''
                    Not a very nice implementation for 2 players - but will do for now
                    '''
                    new_payoffs = np.ones(shape=(self.num_players, self.num_players, self.num_actions))
                    for row in range(len(self.payoffs)):
                        for i in range(len(self.payoffs[row])):
                            new_payoffs[row][i] = cp.copy([self.payoffs[row][i], matrix[row][i]])
                    self.payoffs = cp.copy(new_payoffs)
                else:
                    raise Exception("payoff matrices for >2 players not yet implemented...")
    
    def get_payoffs(self) -> np.ndarray:
        """
        Simply returns the \'payoffs\' attribute

        Returns:
            _type_: _description_
        """
        return cp.copy(self.payoffs)
    
    def set_payoffs(
        self, 
        player: int, 
        input_matrix: Union[List[Number], List[None], np.ndarray] = [None],
        loc: Union[None, Tuple[int]] = None,
        new_value: Union[int, float] = None
    ) -> None:
        """
        Allows for editing of stored payoff matrices.
        The type of the \'players\' argument specifies the modification logic:
         - if \'players\' == int
            - modify payoff matrix for that player only
         - if \'players\' == list/tuple/numpy.ndarray
            - modify payoff matrices for all players specified

        Args:
            players (_type_): _description_
            input_matrix (list, optional): _description_. Defaults to [None].
            loc (_type_, optional): _description_. Defaults to None.
            new_value (_type_, optional): _description_. Defaults to None.

        """
        if isinstance(player, int):
            if None not in input_matrix:
                if isinstance(input_matrix, (list, np.ndarray)):
                    self.player_matrices[f'Player {player}'].set_payoffs(input_matrix=input_matrix)
                else:
                    raise Exception("Specified input matrix not of type \'list\' or \'np.ndarray\'")
            else:
                if loc != None:
                    if isinstance(loc, tuple):
                        if isinstance(new_value, (int, float)):
                            self.player_matrices[f'Player {player}'].set_payoffs(loc=loc, new_value=new_value)
                        else:
                            raise Exception("Specified value not of type \'int\' or \'float\'")
                    else:
                        raise Exception("Specified input location not a tuple")
        elif isinstance(player, (list, tuple, np.ndarray)): 
            for i in player:
                assert isinstance(i, int), 'Non-integer in list of specified players'
                if None not in input_matrix:
                    if isinstance(input_matrix, (list, np.ndarray)):
                        self.player_matrices[f'Player {i}'].set_payoffs(input_matrix=input_matrix)
                    else:
                        raise Exception("Specified input matrix not of type \'list\' or \'np.ndarray\'")
                else:
                    if loc != None:
                        if isinstance(loc, tuple):
                            if isinstance(new_value, (int, float)):
                                self.player_matrices[f'Player {i}'].set_payoffs(loc=loc, new_value=new_value)
                            else:
                                raise Exception("Specified value not of type \'int\' or \'float\'")
                        else:
                            raise Exception("Specified input location not a tuple")
        self._refresh_matrix()
    
    def _refresh_matrix(self) -> None:
        """
        Refreshes the payoffs attribute to be up-to-date with the player matrices

        """
        for player_matrix in enumerate(self.player_matrices):
            if self.num_players == 2:
                '''
                Not a very nice implementation for 2 players - but will do for now
                '''
                for row in range(len(self.payoffs)):
                    for i in range(len(self.payoffs[row])):
                        self.payoffs[row][i][player_matrix[0]] = cp.copy(self.player_matrices[player_matrix[1]].get_payoffs()[row][i])
            else:
                raise Exception("payoff matrices for >2 players not yet implemented...")

    def get_social_welfare(
        self, 
        loc: Union[List[Number], List[None]] = [None]
    ) -> Union[Number, np.ndarray]:
        """
        _summary_

        Args:
            loc (list, optional): _description_. Defaults to [None].

        Returns:
            _type_: _description_
        """
        if None in loc:
            return sum(self.get_payoffs())
        else:
            return self.get_payoffs()[loc].sum()
    
    def get_social_optimum(self, val: str = 'value') -> np.ndarray:
        """
        _summary_

        Args:
            val (str, optional): _description_. Defaults to 'value'.

        Returns:
            _type_: _description_
        """
        if val == 'value':
            return self.get_payoffs().sum().max()
        elif val == 'loc':
            return self.get_payoffs().sum().argmax()
    
    def find_nash(self, val : str = 'value') -> np.ndarray:
        #TODO: will be in an endless loop if no NE present - fix this
        """
        Detects a Nash Equilibrium in the PayoffMatrix

        Args:
            val (str): _description_

        Returns:
            _type_: _description_
        """
        nash_locs = set()
        
        for _ in range(1000):
            p1_cstore = [None, None]
            p2_cstore = [None, None]

            p1_sum = None
            p2_sum = None

            p1_choice = np.random.choice(self.num_actions)
            while (p1_sum == None and p2_sum == None) or (p1_sum == 1 and p2_sum == 1):
                p2_choice = self.player_matrices['Player 2'].get_payoffs()[p1_choice].argmax()
                p2_cstore.append(p2_choice)
                p2_cstore.pop(0)

                p1_choice = self.player_matrices['Player 1'].get_payoffs().T[p2_choice].argmax()
                p1_cstore.append(p1_choice)
                p1_cstore.pop(0)

                if None in p1_cstore or None in p2_cstore:
                    continue
                else:
                    p1_sum = sum(p1_cstore)
                    p2_sum = sum(p2_cstore)
            
            nash_locs.add((p1_choice, p2_choice))

        if val == 'loc':
            return np.array(list(nash_locs))
        elif val == 'value':
            return np.array([self.payoffs[i] for i in nash_locs])
        
class MarkovGame():
    def __init__(self, *states: PayoffMatrix, transitions: Union[None, np.ndarray] = [None]):
        self.states = {}
        self.trans_dict = {}
        
        self.num_players = None
        self.num_states = len(states)
        self.num_actions = []
        
        for i, state in enumerate(states):
            if isinstance(state, PayoffMatrix):
                self.states[f'State {i + 1}'] = cp.copy(state)
                self.num_actions.append(self.states[f'State {i + 1}'].num_actions)
                if self.num_players == None:
                    self.num_players = state.num_players
                elif self.num_players != state.num_players:
                    raise Exception("Number of players not consistent between states!")
            else:
                raise Exception("States must be of type \'PayoffMatrix\'")

        if None in transitions:
            self._reset_transitions()
        else:
            self.transitions = cp.copy(transitions)
    
    def _reset_transitions(self) -> None:
        self.trans_dict = {}
        self.transitions = np.zeros(shape=(len(self.states), len(self.states)))
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions[i])):
                if i == j:
                    self.transitions[i, j] = 1
                    self.trans_dict[str((i, j))] = [self.states[f'State {i + 1}'].action_set, 1]
    
    def _check_transitions(self) -> None:
        prob_sums = [sum(i) for i in self.transitions] # this is not the actual prob sums as it is just looking through the adjacency matrix
        for i, sum in enumerate(prob_sums):
            if sum != 1:
                print(f"Probabilities for state {i} do not sum to 1!")
            else:
                print("All seems fine!")
            
    def get_transitions(self) -> None:
        return self.transitions
    
    def set_transitions(self, transtions: np.ndarray) -> None:
        for i in transtions:
            for j in i:
                if j not in (0, 1):
                    raise Exception("Transition matrices can only contain \'0\' or \'1\' elements")
        
        self.transitions = cp.copy(transtions)
        #self._check_transitions()

    def add_state(self, *state: Union[PayoffMatrix, np.ndarray]) -> None:
        try:
            if isinstance(*state, PayoffMatrix):
                self.states[f'State {len(self.states) + 1}'] = cp.copy(state[0])
        except:
            new_states_arrays = []
            for i in state:
                new_states_arrays.append(i)
            self.states[f'State {len(self.states)}'] = PayoffMatrix(*new_states_arrays)
        finally:
            self.num_states += 1
            self._reset_transitions()
    
    def remove_state(self, state: Union[int, str]) -> None:
        #TODO: reset the keys in state after a state has been removed
        if isinstance(state, int):
            self.states.pop(f'State {state}')
            self._reset_transitions()
        elif isinstance(state, str):
            self.states.pop(state)
            self._reset_transitions()
        else:
            raise Exception('State should be of type \'integer\' or \'string\'')

    def get_trans_prob(
        self, 
        state_from: int, 
        state_to: int, 
        action: Tuple[int]
    ) -> Union[int, float]:
        trans_data = self.trans_dict[f'({state_from}, {state_to})']
        if action in trans_data[0]:
            return trans_data[1]
        else:
            return 0

    def add_transition(
        self, 
        state_from: int, 
        state_to: int, 
        action: Tuple[int], 
        prob: Union[int, float] = 1
    ) -> None:
        if action in self.trans_dict[str((state_from - 1, state_from - 1))][0]:
            self.trans_dict[str((state_from - 1, state_from - 1))][0].remove(action)
        for i, list in enumerate(self.transitions):
            for j, value in enumerate(list):
                if (i, j) == (state_from-1, state_to-1):
                    self.transitions[i, j] = 1
                    if str((state_from-1, state_to-1)) in self.trans_dict:
                        if action in self.trans_dict[str((state_from-1, state_to-1))][0]:
                            print(f'{action} already in transition from state {state_from} to state {state_to}!')
                            self.trans_dict[str((state_from-1, state_to-1))][0].add((action))
                        else:
                            self.trans_dict[str((state_from-1, state_to-1))][0].add((action))
                    else:
                        self.trans_dict[str((state_from-1, state_to-1))] = [{action}, prob]
        
        #self._check_transitions()
        
    def run_vi(self, gamma: float, iterations: int) -> None: # Change to tabular Q-learning
        prev_val_sum = 0
        for iteration in range(iterations):
            for i in range(len(self.states)):
                state_transition_key_buffer = []
                for j in self.trans_dict:
                    if j[1] == str(i):
                        state_transition_key_buffer.append(j)
                for k in range(self.num_players):
                    for key in state_transition_key_buffer:
                        for action in self.trans_dict[key][0]:
                            s = self.states[f'State {i + 1}']
                            s_prime = self.states[f'State {int(key[4]) + 1}']
                            p = self.trans_dict[key][1]
                            r = s.get_payoffs()[action]
                            v_s_prime = s_prime.values.max()
                            
                            s.values[*action, k] = (p * (r[k] + (gamma * v_s_prime)))
    
    def q_learn(self, gamma: float, alpha: float, episodes: int, T: int, s0: int, eps: float):
        # Initialise Q(s, a) for all s in S, a in A, arbitrarily except that Q(terminal, .) = 0
        q_tables = [{} for i in range(self.num_players)]
        next_state = None
        for player, q_table in enumerate(q_tables):
            for i in range(episodes):
                # Initialise S
                current_state = str(s0)
                for t in range(T):
                    # Choose A from S using policy derived from Q (e.g., eps-greedy)
                    if current_state not in q_table:
                        # initialise q_table entry with key 'current_state' and value as a dict with each action as key and 0 as initial value
                        print(f'intialising state {current_state} in q_table')
                        q_table[current_state] = dict.fromkeys( [ str(i) for i in list(product(list(range(self.num_players)), list(range(self.num_players)))) ] , 0)

                    # eps-greedy policy action selection
                    if np.random.rand() < eps:
                        action = max(q_table[current_state], key=q_table[current_state].get)
                    else:
                        action = random.choice([i for i in q_table[current_state]])
                    
                    # Take action A, observe R, S'
                    reward = self.states[f'State {current_state}'].player_matrices[f"Player {player + 1}"].get_payoffs()[eval(action)]
        
                    state_action_pairs = [(transitions, self.trans_dict[transitions][0]) for transitions in fnmatch.filter([key for key in self.trans_dict], f'({int(current_state) - 1}, ?)')]
                    #print(state_action_pairs)
                    for state, action_set in state_action_pairs:
                        if eval(action) in action_set:
                            next_state = str(eval(state)[1] + 1)
                            break
                
                    if next_state == None:
                        print(f"Action {action} in state {current_state} not associated with state!")
                        current_state = None # throw spanner in works to crash program
    
                    # Q(S, A) <- Q(S, A) + alpha[R + gamma * max_a Q(S', a) - Q(S, A)]
                    if str(next_state) not in q_table:
                        # initialise q_table entry with key 'current_state' and value as a dict with each action as key and 0 as initial value
                        print(f'intialising state {next_state} in q_table')
                        q_table[str(next_state)] = dict.fromkeys( [ str(i) for i in list(product(list(range(self.num_players)), list(range(self.num_players)))) ] , 0)
                    
                    q_S_A = cp.copy(q_table[current_state][action])
                    max_a_Q_Sp_A = cp.copy(q_table[next_state][max(q_table[next_state], key=q_table[next_state].get)])

                    q_table[current_state][action] = round(q_S_A + alpha * (reward + gamma * max_a_Q_Sp_A - q_S_A), 2)

                    # S <- S'
                    current_state = next_state
                    next_state = None
        
        return q_tables

if __name__ == '__main__':

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

    # ---- single critical state game ----
    # Construct state objects for each par of payoff matrices
    '''
    state_1 = PayoffMatrix(p1_s1, p2_s1) # 0
    state_2 = PayoffMatrix(p1_s2, p2_s2) # R
    state_3 = PayoffMatrix(p1_s3, p2_s3) # S, T
    state_4 = PayoffMatrix(p1_s4, p2_s4) # P
    state_5 = PayoffMatrix(p1_s5, p2_s5) # T, S
    '''
    
    # 0 reward (transitory) states
    state_1 = PayoffMatrix(p1_s1, p2_s1)
    state_2 = cp.copy(state_1)
    state_6 = cp.copy(state_1)

    # R state
    state_7 = PayoffMatrix(p1_s2, p2_s2)

    # S, T states
    state_3 = PayoffMatrix(p1_s3, p2_s3)
    state_8 = cp.copy(state_3)
    
    # T, S states
    state_5 = PayoffMatrix(p1_s5, p2_s5)
    state_9 = cp.copy(state_5)

    # P state
    state_4 = PayoffMatrix(p1_s4, p2_s4)


    def get_altruistic_game(state, alpha):
        state = cp.copy(state)
        altruistic_payoffs = state.get_payoffs()
        for index, i in enumerate(altruistic_payoffs):
            for jndex, j in enumerate(i):
                altruistic_payoffs[index][jndex] = j + alpha * sum(j)
        return cp.copy(PayoffMatrix(altruistic_payoffs.T[0], altruistic_payoffs.T[1]))


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
        return cp.copy(PayoffMatrix(altruistic_payoffs.T[0], altruistic_payoffs.T[1]))

    
    # Construct Markov Game called 'game' with previously defined states 1-4
    game = MarkovGame(state_1, state_2, state_3, state_4, state_5)
    
    alpha = 1

    game_a = MarkovGame(get_altruistic_game(state_1, alpha), get_altruistic_game(state_2, alpha), get_altruistic_game(state_3, alpha), get_altruistic_game(state_4, alpha), get_altruistic_game(state_5, alpha))

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