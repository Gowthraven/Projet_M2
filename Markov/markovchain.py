import numpy as np
from music21 import metadata, note, stream


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
            
            melody=melody.split(", ")
            
            splitted=melody[0].split('-')
            if len(splitted)==2:
                note_name=splitted[0]
            else:
                note_name="-".join( substring for substring in splitted[:-1])
            duration=float(splitted[-1])
            self.initial_probabilities[self._state_indexes[(note_name,duration)]]+=1
            
            previous_note=(note_name,duration)
            
            for note_duration in melody[1:]:
                
                splitted=note_duration.split('-')
                if len(splitted)==2:
                    note_name=splitted[0]
                else:
                    note_name="-".join( substring for substring in splitted[:-1])
                duration=float(splitted[-1])
                
                self.transition_matrix[self._state_indexes[previous_note],self._state_indexes[(note_name,duration)]]+=1
                previous_note=(note_name,duration)

        self._normalize_initial_probabilities()
        self._normalize_transition_matrix()


    def generate(self, length):
        """
        Generate a melody of a given length.

        Parameters:
            length (int): The length of the sequence to generate.

        Returns:
            melody (list of tuples): A list of generated states.
        """
        melody = [self._generate_starting_state()]
        for _ in range(1, length):
            melody.append(self._generate_next_state(melody[-1]))
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
    states=[]
    for melody in training_data:
        melody=melody.split(", ")
        for note_duration in melody:
            splitted=note_duration.split('-')
            if len(splitted)==2:
                note_name=splitted[0]
            else:
                note_name="-".join( substring for substring in splitted[:-1])
            duration=float(splitted[-1])
            states.append((note_name, duration))

    return list(set(states))

def generated_to_string(generated_melody):
    melody=generated_melody[0][0]+"-"+str(generated_melody[0][1])
    for note_duration in generated_melody[1:]:
        melody+=", "+note_duration[0]+"-"+str(note_duration[1])

    return melody