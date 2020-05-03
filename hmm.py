import os

class HMM(object):

    def __init__(self, file_path):

        self.file_path = file_path
        self.initial_proba = {}
        self.transition_proba = {}
        self.emission_proba = {}
        self.states_list = [1, 2, 3]
        self.emission_list = [1, 2, 3]
        self.emissions = []
        self.real_states = []
        self.get_input_values()

    def load_file(self):
        current_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(current_path, 'dice', self.file_path)
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                c = line.split('#')[0].replace(
                    '\n', '').replace(',', ' ').split(' ')
                if c == ['']:
                    continue
                else:
                    lines.append([eval(v) for v in c])
        return lines

    def get_input_values(self):
        l = self.load_file()

        # Initial probabilities
        self.initial_proba = {s:0.33333333 for s in self.states_list}

        # Transition probabilities
        for i in self.states_list:
            dict_i = {}
            for j in self.states_list:
                if i == j:
                    dict_i[j] = l[0][0]
                else:
                    dict_i[j] = (1 - l[0][0]) / 2.
            self.transition_proba[i] = dict_i

        # Emission probabilities
        for i, s in enumerate(self.states_list):
            self.emission_proba[s] = {j+1:v for j, v in enumerate(l[i+1])}

        # Emissions
        self.emissions = l[4]

        print('-' * 70)
        print('Input File:', self.file_path)
        print('-' * 70)
        print('States List:\n\t', self.states_list)
        print('Emissions List:\n\t', self.emission_list)
        print('Initial Probabilities:\n\t', self.initial_proba)
        print('Transition Probabilities:\n\t', self.transition_proba)
        print('Emission Probabilities:\n\t', self.emission_proba)
        print('Emissions:\n\t', self.emissions)

        # Real states
        if len(l) > 5:
            self.real_states = l[5]
            print('Real States:\n\t', self.real_states)

    def print_path(self, v_path):
        # Print a table of steps from dictionary
        yield 't:' + ' '.join('{:>7d}'.format(i) for i in range(len(v_path)))
        yield 'e:' + ' '.join('{:>7d}'.format(i) for i in self.emissions)
        for s in v_path[0]:
            yield "{}:  ".format(s) + " ".join(
                '{:.5f}'.format(v[s]['proba']) for v in v_path)
   
    def viterbi(self):

        # Init t_0
        v_path = [{}]
        for s in self.states_list:
            v_path[0][s] = {
                'proba': self.initial_proba[s] * \
                    self.emission_proba[s][self.emissions[0]],
                'pre_state': None
            }

        # Calculate the viterbi path
        for t, e in enumerate(self.emissions[1:]):
            dict_t = {}
            # Current state
            for s in self.states_list:
                proba_state = 0.
                pre_state = self.states_list[0]
                # Previous state
                for pre_s in self.states_list:
                    proba = v_path[t][pre_s]['proba'] * \
                        self.transition_proba[pre_s][s]
                    # Track the max proba and the previous state
                    if proba > proba_state :
                        proba_state = proba
                        pre_state = pre_s
                # Times the emission proba
                proba_state *= self.emission_proba[s][e]
                dict_t[s] = {'proba': proba_state, 'pre_state': pre_state}
            # Add current state to viterbi path
            v_path.append(dict_t)

        # print('-' * 70)
        # print('Viterbi Path:')
        # for line in self.print_path(v_path):
        #     print(line)

        # Choose the final state
        max_proba = 0.
        state_selected = None
        for s, value in v_path[-1].items():
            if value['proba'] > max_proba:
                max_proba = value['proba']
                state_selected = s

        # Trace back the state path
        back_path = [state_selected]
        for v_t in v_path[::-1]:
            state_selected = v_t[state_selected]['pre_state']
            back_path.append(state_selected)

        # The most possible state path
        state_path = back_path[-2::-1]

        return state_path, max_proba

    def run(self):

        state_path, proba = self.viterbi()
        print('-' * 70)
        print('The steps of states are:')
        print('\t', state_path)
        print('with the highest probability of:\n\t', proba)


if __name__ == '__main__':

    HMM('hmm_dice_1586654480227.txt').run()
