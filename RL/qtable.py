import numpy as np

from enum import Enum
from configs import MU, GAMMA, PMODE_ID
from board import IBoard

class Action(Enum):
    INC_CL, DEC_CL, DO_NOTHING = range(3)

class QTable():
    def __init__(self, client: IBoard, states, actions):
        self.client = client
        self.mu = MU
        self.gamma = GAMMA
        self.states = states
        self.actions = actions
        self.init_table()
        self.init_prohibited_states()
    
    def init_table(self):
        self.table = {}
        for state in self.states:
            self.table[str(state)] = np.zeros(len(self.actions))
    
    def init_prohibited_states(self):
        self.prohibited_states = []
        for _ in range(len(PMODE_ID)):
            self.prohibited_states.append({})
    
    def get_next_state(self, current_state_index, action):
        cl_index = np.array(current_state_index[1:])
        cl_size = len(self.client.CONCURRENCY)

        if action == Action.INC_CL:
            cl_index = min(cl_index + 1, cl_size - 1)
        elif action == Action.DEC_CL:
            cl_index = max(0, cl_index - 1)
        elif action == Action.DO_NOTHING:
            pass
        next_state_index = [current_state_index[0], cl_index]
        err = 1 if action != Action.DO_NOTHING and next_state_index == current_state_index else 0
        return next_state_index, err
    
    def available_actions(self, current_state_index: np.array):
        available_actions = []
        for action_index, action in enumerate(Action):
            _, err = self.get_next_state(current_state_index, action)
            if err == 1:
                continue
            available_actions.append(action_index)
        return available_actions
    
    def get_action(self, current_state: np.array, epsilon: float, current_state_index):
        if not (self.prohibited_states[current_state[0]][str(current_state[1:])] if str(current_state[1:]) in self.prohibited_states[current_state[0]].keys() else False):
            rand_num = np.random.rand()
            if rand_num < epsilon:
                available_actions = self.available_actions(current_state_index)
                if len(available_actions) == 0:
                    return None
                action = np.random.choice(available_actions)
            else:
                action = self.get_largest_q_action(str(current_state))
                return Action(action)
    
    def get_largest_q_action(self, state):
        return np.argmax(self.table[state])
    
    def update_sarsa(self, state, action, new_state, new_action, reward):
        action = action.value
        new_action = new_action.value
        state, new_state = str(state), str(new_state)
        
        qsa = self.table[state][action]
        qnewsa = self.table[new_state][new_action]
        
        # SARSA update rule (on-policy)
        self.table[state][action] = qsa + self.mu * (reward + self.gamma * qnewsa - qsa)
        
        return self.table[state][action]

    
    def update_qlearning(self, state, action, new_state, reward):
        action = action.value
        state, new_state = str(state), str(new_state)
        
        new_state_action = self.get_largest_q_action(new_state)
        qsa = self.table[state][action]

        qnewsa = self.table[new_state][new_state_action]

        # Q-learning update rule (off-policy)
        self.table[state][action] = (1 - self.mu) * qsa + self.mu * (reward + self.gamma * qnewsa)
        return self.table[state][action]
    
class QTableVisualizer:
    def __init__(self, q_table):
        self.q_table = q_table

    def visualize(self):
        for state, actions in self.q_table.items():
            print(f"State: {state}")
            print("Actions:")
            for action, q_value in enumerate(actions):
                print(f"  Action {action}: Q-value = {q_value}")
            print()