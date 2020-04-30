from State import State
from utils import read_file

class CustomerHMM(object):
    def __init__(self):
        self.states = {0:'ZERO', 1:'AWARE', 2:'CONSIDERING', 3:'EXPERIENCING', 4:'READY', 5:'LOST', 6:'SATISFIED'}
        self.state_transition_matrix = [[0]*7 for i in range(7)]
        self.state_transition_matrix[0][0] = 0.6
        self.state_transition_matrix[0][1] = 0.4
        self.state_transition_matrix[1][1] = 0.49
        self.state_transition_matrix[1][2] = 0.3
        self.state_transition_matrix[1][4] = 0.01
        self.state_transition_matrix[1][5] = 0.2
        self.state_transition_matrix[2][2] = 0.48
        self.state_transition_matrix[2][3] = 0.2
        self.state_transition_matrix[2][4] = 0.02
        self.state_transition_matrix[2][5] = 0.3
        self.state_transition_matrix[3][3] = 0.4
        self.state_transition_matrix[3][4] = 0.3
        self.state_transition_matrix[3][5] = 0.3
        self.state_transition_matrix[4][4] = 0.8
        self.state_transition_matrix[4][5] = 0.2
        self.state_transition_matrix[5][5] = 1
        self.state_transition_matrix[6][6] = 1

        self.emissions_list = {}
        self.emissions = {0:'DEMO', 1:'VIDEO', 2:'TESTIMONIAL', 3:'PRICING', 4:'BLOG', 5:'PAYMENT'}
        self.emission_transition_matrix = [[0.1, 0.01, 0.05, 0.3, 0.5, 0.0], [0.1, 0.01, 0.15, 0.3, 0.4, 0.0], [0.2, 0.3, 0.05, 0.4, 0.4, 0.0], [
            0.4, 0.6, 0.05, 0.3, 0.4, 0.0], [0.05, 0.75, 0.35, 0.2, 0.4, 0.0], [0.01, 0.01, 0.03, 0.05, 0.2, 0.0], [0.4, 0.4, 0.01, 0.05, 0.5, 1.0]]

    def get_observation_list(self, file_name):
        self.emissions_list = read_file(file_name)

    def get_hidden_states(self):
        category_of_states = 6
        category_of_emissions = 5
        number_of_emissions = len(self.emissions_list)
        # create states
        states = [[0]*category_of_states for i in range(number_of_emissions)]
        for i in range(number_of_emissions):
            for j in range(category_of_states):
                states[i][j] = State()
        # set probability of initial state
        states[0][0].probability = 1
        states[0][1].probability = 0
        states[0][2].probability = 0
        states[0][3].probability = 0
        states[0][4].probability = 0
        states[0][5].probability = 0

        # Viterbi algorithm starts
        # the first layer
        for i in range(category_of_states):
            p_ob = 1
            for em in range(category_of_emissions):
                ob = self.emissions[em]
                if ob in self.emissions_list[0]:
                    p_ob *= self.emission_transition_matrix[i][em]
                else:
                    p_ob *= 1 - self.emission_transition_matrix[i][em]
            states[0][i].probability *= p_ob
        # the rest layers
        for i in range(1,number_of_emissions):
            for right in range(category_of_states):
                max_probability = -1
                max_index = 0
                for left in range(category_of_states):
                    p_ob = 1
                    for em in range(category_of_emissions):
                        ob = self.emissions[em]
                        if ob in self.emissions_list[i]:
                            p_ob *= self.emission_transition_matrix[right][em]
                        else:
                            p_ob *= 1 - self.emission_transition_matrix[right][em]
                    if max_probability <= self.state_transition_matrix[left][right] * p_ob * float(states[i-1][left].probability):
                        max_probability = self.state_transition_matrix[left][right] * p_ob * float(states[i-1][left].probability)
                        max_index = left
                states[i][right].probability = max_probability
                states[i][right].father = max_index

        # backtrack from the last layer
        max_p = -1
        max_index = 0
        last_layer = number_of_emissions - 1
        flag_payment = False
        if 'PAYMENT' in self.emissions_list[number_of_emissions-1]:
            last_layer -= 1
            flag_payment = True
        for i in range(category_of_states):
            if states[last_layer][i].probability > max_p:
                max_p = states[last_layer][i].probability
                max_index = i
        explanation_states = [self.states[max_index]]
        tmp_father = states[last_layer][max_index].father
        while last_layer > 0:
            last_layer -= 1
            explanation_states.append(self.states[tmp_father])
            tmp_father = states[last_layer][tmp_father].father
        explanation_states = explanation_states[::-1]

        # Print the most likely sequence of states
        print('The most likely sequence of states: ')
        for val in explanation_states:
            print(val)
        if flag_payment:
            print('SATISFIED')
