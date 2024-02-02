import numpy as np
from music21 import metadata, note, stream
import json


class MarkovChainMelodyGenerator:
    """
    Represents a Markov Chain model for melody generation.
    """

    def __init__(self, states):
        """
        Initialize the MarkovChain with a list of states.

        Parameters:
            states (list of tuples): A list of possible (pitch, duration)
                pairs.
        """

        self.states = states
        self.initial_probabilities = np.zeros(len(self.states))
        self.transition_matrix = np.zeros((len(self.states), len(self.states)))
        self._state_indexes = {state: i for (i, state) in enumerate(self.states)}

    def train(self, training_data):
        """
        Train the model based on a list of notes.

        Parameters:
            notes (list): A list of music21.note.Note objects.
        """
        for melody in training_data:
            self.initial_probabilities[self._state_indexes[melody[0]]]+=1
            previous_note=(melody[0])
            
            for note_duration in melody[1:]:
                self.transition_matrix[self._state_indexes[previous_note],self._state_indexes[note_duration]]+=1
                previous_note=(note_duration)

        self._normalize_initial_probabilities()
        self._normalize_transition_matrix()


    def generate(self, length,starting_sequence=[],mode="note",time_signature="2/4"): #,clean_measure=False
        """
        Genere une melodie de la longueur voulue.

        Parametres:
            length (int): La longueur de la melodie en note ou measure selon mode.
            starting_sequence (liste de tuple (string,float)) : Sequence de notes placees au debut de la melodie.
            mode (string) : Permet de definir la longueur de la melodie en note ou en mesure. Doit valoir "note" ou "measure".
            time_signature (string) : Choix de la signature rythmique. Doit etre de la forme "int/int"

        Sortie:
            melody (liste de tuples (string,float)): La liste d'etats generee.
        """
        if starting_sequence==[]:
            melody = [self._generate_starting_state()]
        else:
            melody = starting_sequence
        '''
        if clean_measure:
            measureDuration = float(time_signature.split('/')[0])
            measureCount = 0
            measureTime = 0.0
            for note_duration in melody:
                measureTime+=note_duration[1]
                if measureTime==measureDuration:
                    measureTime = 0.0
                    measureCount+=1
                elif measureTime>measureDuration:
                    measureTime = measureTime-measureDuration
                    measureCount+=1
            for _ in range(length):
                if measureTime==measureDuration:
                    measureTime = 0.0
                    measureCount+=1
                
                
                melody.append(self._generate_next_state_clean(melody[-1],measureDuration-measureTime))
        ''' 
        if mode=="note":
            for _ in range(len(melody), length):
                melody.append(self._generate_next_state(melody[-1]))
        elif mode=="measure":
            measureDuration = float(time_signature.split('/')[0])
            totalTime = 0.0
            for note_duration in melody:
                duration=float(note_duration.split("-")[-1])
                totalTime+=duration
            while totalTime<measureDuration*length:
                next_state=self._generate_next_state(melody[-1])
                melody.append(next_state)
                totalTime+=float(next_state.split("-")[-1])
            if totalTime>measureDuration*length:
                last_state=melody[-1].split("-")
                last_state="-".join(n for n in last_state[:-1])+"-"+str(float(last_state[-1])-(totalTime-(measureDuration*length)))
                melody[-1]=last_state
        else:
            print("mode doit être égal à 'note' ou 'melody'")
            return None

        return melody

    def _normalize_initial_probabilities(self):
        """
        Normalize the initial probabilities array such that the sum of all
        probabilities equals 1.
        """
        total = np.sum(self.initial_probabilities)
        if total:
            self.initial_probabilities /= total
        self.initial_probabilities = np.nan_to_num(self.initial_probabilities)


    def _normalize_transition_matrix(self):
        """
        This method normalizes each row of the transition matrix so that the
        sum of probabilities in each row equals 1. This is essential for the rows
        of the matrix to represent probability distributions of
        transitioning from one state to the next.
        """

        # Calculate the sum of each row in the transition matrix.
        # These sums represent the total count of transitions from each state
        # to any other state.
        row_sums = self.transition_matrix.sum(axis=1)

        # Use np.errstate to ignore any warnings that arise during division.
        # This is necessary because we might encounter rows with a sum of 0,
        # which would lead to division by zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            # Normalize each row by its sum. np.where is used here to handle
            # rows where the sum is zero.
            # If the sum is zero (no transitions from that state), np.where
            # ensures that the row remains a row of zeros instead of turning
            # into NaNs due to division by zero.
            self.transition_matrix = np.where(
                row_sums[:, None],  # Condition: Check each row's sum.
                # True case: Normalize if sum is not zero.
                self.transition_matrix / row_sums[:, None],
                0,  # False case: Keep as zero if sum is zero.
            )

    def _generate_starting_state(self):
        """
        Generate a starting state based on the initial probabilities.

        Returns:
            A state from the list of states.
        """
        initial_index = np.random.choice(
            list(self._state_indexes.values()), p=self.initial_probabilities
        )
        return self.states[initial_index]

    def _generate_next_state(self, current_state):
        """
        Generate the next state based on the transition matrix and the current
        state.

        Parameters:
            current_state: The current state in the Markov Chain.

        Returns:
            The next state in the Markov Chain.
        """
        if self._does_state_have_subsequent(current_state):
            index = np.random.choice(
                list(self._state_indexes.values()),
                p=self.transition_matrix[self._state_indexes[current_state]],
            )
            return self.states[index]
        return self._generate_starting_state()

    def _generate_next_state_clean(self, current_state,remaining_time):
        p_next_state=self.transition_matrix[self._state_indexes[current_state]][:]
        for next_state in self._state_indexes.keys():
            if next_state[1]<remaining_time:
                p_next_state[self._state_indexes[next_state]]=0
        if p_next_state.sum() > 0:
            index = np.random.choice(
                list(self._state_indexes.values()),
                p=p_next_state,
            )
            return self.states[index]
        
        return ("rest",remaining_time)

    def _does_state_have_subsequent(self, state):
        """
        Check if a given state has a subsequent state in the transition matrix.

        Parameters:
            state: The state to check.

        Returns:
            True if the state has a subsequent state, False otherwise.
        """
        return self.transition_matrix[self._state_indexes[state]].sum() > 0

def extract_states(training_data):
    states=set()
    for melody in training_data:
        for note_duration in melody:
            states.add(note_duration)

    return list(states)

def generated_to_json(title,generated_melodies,key="F",part="All"):
    '''for melody in generated_melodies:
        for i in range(len(melody)):

            melody[i]=(melody[i][0],melody[i][1])'''
    generated = [ {'Title' : f"Markov {i+1}" , 'Part': part, 'Key' : key , 'Generated' : melody} for i, melody in enumerate(generated_melodies) ]
    with open(f"Generated/{title}.json","w") as f:
        json.dump(generated,f,indent=2)
