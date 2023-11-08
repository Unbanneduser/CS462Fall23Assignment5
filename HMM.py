

import random
import argparse
import codecs
import os
import numpy

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        
        # Read transition probabilities from the .trans file
        with open(basename + '.trans', 'r') as trans_file:
            for line in trans_file:
                state1, state2, probability = line.strip().split()
                # Update the transition probabilities in self.transitions dictionary
                if state1 not in self.transitions:
                    self.transitions[state1] = {}
                self.transitions[state1][state2] = float(probability)

        # Read emission probabilities from the .emit file
        with open(basename + '.emit', 'r') as emit_file:
            for line in emit_file:
                state, output, probability = line.strip().split()
                # Update the emission probabilities in self.emissions dictionary
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][output] = float(probability)


   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""

        # Initialize the state and output sequences
        state_seq = []
        output_seq = []

        # Get the initial state
        initial_state = '#'
        state_seq.append(initial_state)


        # Generate the rest of the state and output sequences
        for i in range(n):
            # Get the next state
            next_state = numpy.random.choice(list(self.transitions[state_seq[-1]].keys()), p=list(self.transitions[state_seq[-1]].values()))
            state_seq.append(next_state)

            # Get the next output
            next_output = numpy.random.choice(list(self.emissions[next_state].keys()), p=list(self.emissions[next_state].values()))
            output_seq.append(next_output)

        # Return the observation
        return Observation(state_seq[1:], output_seq)
    
    def forward(self, observation):
        '''
        Next, implement the forward algorithm. This tells us, for a sequence of observations, the most likely
final state. You should be able to run this like so:
'''
        # use dynamic programming to compute the forward matrix

        states = list(self.transitions.keys())
        states.remove('#')

        # Initialize the forward matrix
        forward_matrix = numpy.zeros((len(self.transitions), len(observation)))
        # forward_matrix[i][j] = probability of being in state i after seeing the first j observations
        # answer is forward_matrix[:, -1].argmax()

        # Initialize the first column of the forward matrix
        for i in range(len(states)):
            forward_matrix[i][0] = self.transitions['#'][states[i]]
            forward_matrix[i][0] *= self.emissions[states[i]][observation[0]] if observation[0] in self.emissions[states[i]] else 1e-8

        # Fill in the rest of the forward matrix
        for j in range(1, len(observation)):
            for i in range(len(states)):
                forward_matrix[i][j] = numpy.max([forward_matrix[k][j - 1] * self.transitions[states[k]][states[i]] for k in range(len(states))])
                forward_matrix[i][j] *= self.emissions[states[i]][observation[j]] if observation[j] in self.emissions[states[i]] else 1e-8
        
        return states[forward_matrix[:, -1].argmax()]

    

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        
        # use dynamic programming to compute the viterbi matrix

        states = list(self.transitions.keys())
        states.remove('#')

        # Initialize the viterbi matrix
        viterbi_matrix = numpy.zeros((len(self.transitions), len(observation)))
        prev_state_matrix = numpy.zeros((len(self.transitions), len(observation)))
        # viterbi_matrix[i][j] = probability of being in state i after seeing the first j observations
        # answer is viterbi_matrix[:, -1].argmax()

        # Initialize the first column of the viterbi matrix
        for i in range(len(states)):
            viterbi_matrix[i][0] = self.transitions['#'][states[i]]
            viterbi_matrix[i][0] *= self.emissions[states[i]][observation[0]] if observation[0] in self.emissions[states[i]] else 1e-8

        # Fill in the rest of the viterbi matrix
        for j in range(1, len(observation)):
            for i in range(len(states)):
                viterbi_matrix[i][j] = numpy.max([viterbi_matrix[k][j - 1] * self.transitions[states[k]][states[i]] for k in range(len(states))])
                viterbi_matrix[i][j] *= self.emissions[states[i]][observation[j]] if observation[j] in self.emissions[states[i]] else 1e-8
                prev_state_matrix[i][j] = numpy.argmax([viterbi_matrix[k][j - 1] * self.transitions[states[k]][states[i]] for k in range(len(states))])
        
        # Backtrack to find the most likely state sequence
        state_seq = []
        state_seq.append(states[viterbi_matrix[:, -1].argmax()])
        for j in range(len(observation) - 1, 0, -1):
            state_seq.append(states[int(prev_state_matrix[states.index(state_seq[-1])][j])])

        return state_seq[::-1]
        
        


def handle_generate(train_file, generate_count):
    model = HMM()
    model.load(train_file)
    print(model.generate(generate_count))


def handle_forward(train_file, obs_filename):
    model = HMM()
    model.load(train_file)
    with open(obs_filename, 'r') as obs_file:
        for line in obs_file:
            line = line.strip()
            if line == '':
                continue
            observation = line.split()
            last_state = model.forward(observation)
            print('observation: %s, last state: %s' % (observation, last_state))


def handle_viterbi(train_file, obs_filename):
    model = HMM()
    model.load(train_file)
    file_parts = obs_filename.split('.')
    gold_file = '.'.join([file_parts[0], 'tagged', file_parts[-1]])
    with open(gold_file, 'r') as gold_file:
        is_gold = True
        for line in gold_file:
            line = line.strip()
            if is_gold:
                golds = line.split()
            else:
                observation = line.split()
                pred = model.viterbi(observation)
                assert len(golds) == len(pred)
                print('observation: %s, gold: %s, pred: %s, acc: %f' % (observation, golds, pred, sum([1 for g, p in zip(golds, pred) if g == p]) / len(golds)))
            is_gold = not is_gold









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMM Text Generation')
    parser.add_argument('train_file', type=str, help='HMM train file')
    parser.add_argument('--generate', type=int, help='Number of texts to generate')
    parser.add_argument('--forward', type=str, help='File of observations to run forward algorithm on')
    parser.add_argument('--viterbi', type=str, help='File of observations to run viterbi algorithm on')
    args = parser.parse_args()
    if args.generate:
        handle_generate(args.train_file, args.generate)
    elif args.forward:
        handle_forward(args.train_file, args.forward)
    elif args.viterbi:
        handle_viterbi(args.train_file, args.viterbi)




# run this like below:
'''
1. To test generate
python hmm.py partofspeech.browntags.trained --generate 20
2. To test forward
python hmm.py partofspeech.browntags.trained --forward ambiguous_sents.obs
3. To test viterbi
python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs
'''