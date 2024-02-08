import numpy as np
from music21 import metadata, note, stream
import json


class MarkovChainMelodyGenerator:
    """
    Représente un modèle de chaîne de Markov pour la génération de mélodies.
    """

    def __init__(self, states):
        """
        Initialiser la chaîne de Markov avec une liste d'états.

        Parametres:
            states (liste de string): une liste de string unique de la forme "note-duree" qui represente chaque etat.
        """

        self.states = states+["end"]
        self.initial_probabilities = np.zeros(len(self.states))
        self.transition_matrix = np.zeros((len(self.states)+1, len(self.states)))
        self._state_indexes = {state: i for (i, state) in enumerate(self.states)}

    def train(self, training_data,end_transition=False):
        """
        Entraine le modèle sur des melodies.

        Parameters:
            training_data (liste de liste de string): Une liste de melodie qui sont elle même sous forme de liste de string "note-duree".
            end_transition (bool) : Si il vaut True alors le modele prend en compte la transition vers l'etat "fin" qui interromp la generation.
        """
        for melody in training_data:
            self.initial_probabilities[self._state_indexes[melody[0]]]+=1
            previous_state=(melody[0])
            
            for next_state in melody[1:]:
                self.transition_matrix[self._state_indexes[previous_state],self._state_indexes[next_state]]+=1
                previous_state=(next_state)

            if end_transition:    
                self.transition_matrix[self._state_indexes[melody[-1]],-1]+=1

        self._normalize_initial_probabilities()
        self._normalize_transition_matrix()


    def generate(self, length,starting_sequence=[],mode="note",time_signature="2/4"):
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
        if mode=="note":
            for _ in range(len(melody), length):
                next_state=self._generate_next_state(melody[-1])
                if next_state=="end":
                    return melody
                melody.append(next_state)
        elif mode=="measure":
            measureDuration = float(time_signature.split('/')[0])
            totalTime = 0.0
            for note_duration in melody:
                duration=float(note_duration.split("-")[-1])
                totalTime+=duration
            while totalTime<measureDuration*length:
                next_state=self._generate_next_state(melody[-1])
                if next_state=="end":
                    return melody
                melody.append(next_state)
                totalTime+=float(next_state.split("-")[-1])
            if totalTime>measureDuration*length:
                last_state=melody[-1].split("-")
                last_state="-".join(n for n in last_state[:-1])+"-"+str(float(last_state[-1])-(totalTime-(measureDuration*length)))
                melody[-1]=last_state
        else:
            print("mode doit etre egal à 'note' ou 'measure'")
            return None

        return melody

    def _normalize_initial_probabilities(self):
        """
        Normalise le tableau des probabilites initiales de maniere a ce que la somme de toutes les probabilites soit égale à 1.
        """
        total = np.sum(self.initial_probabilities)
        if total:
            self.initial_probabilities /= total
        self.initial_probabilities = np.nan_to_num(self.initial_probabilities)


    def _normalize_transition_matrix(self):
        """
        Normalise la matrice des probabilites de transition de maniere à ce que la somme des valeurs de chaque ligne soit égale à 1.
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
        Genere un etat de depart a partir du vecteur de probabilites initiales.

        Sortie:
            Un etat string de la forme "note-duree".
        """
        initial_index = np.random.choice(
            list(self._state_indexes.values()), p=self.initial_probabilities
        )
        return self.states[initial_index]

    def _generate_next_state(self, current_state):
        """
        Genere l'etat suivant a partir de la matrice de transition et de l'etat actuel.

        Parametres:
            current_state: L'etat actuel de la chaine de markov.

        Sortie:
            Un etat string de la forme "note-duree".
        """
        if current_state in self.states:
            if self._does_state_have_subsequent(current_state):
                index = np.random.choice(
                    list(self._state_indexes.values()),
                    p=self.transition_matrix[self._state_indexes[current_state]],
                )
                return self.states[index]
            return self._generate_starting_state()
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
        Verifie si un etat donne a un etat suivant dans la matrice de transition.

        Parametres:
            state: L'etat a verifier.

        Sortie:
            Vrai si l'etat donne a un etat suivant dans la matrice de stransition, faux sinon.
        """
        return self.transition_matrix[self._state_indexes[state]].sum() > 0

def extract_states(training_data):
    """
    Permet d'extraire une liste d'etat unique contenu dans des melodies.

    Parametres:
        training_data(liste de liste de string): Une liste de melodie qui sont elle même sous forme de liste de string "note-duree".

    Sortie:
        states (liste de string): une liste de string unique de la forme "note-duree" qui represente chaque etat.
    """
    states=set()
    for melody in training_data:
        for note_duration in melody:
            states.add(note_duration)

    return list(states)

def generated_to_json(file_name,generated_melodies,key="F"):
    """
    Permet d'enregistrer une liste de melodies generees dans un fichier .json situe dans Data/Generated.

    Parametres:
        file_name(string): Nom du fichier .json cree.
        generated_melodies(liste de liste de string): Une liste de melodie generee qui sont elle même sous forme de liste de string "note-duree".
        key(string): Cle globale de la melodie. "F" par defaut.
    """
    generated = [ {'Title' : f"Markov {i+1}" , 'Key' : key , 'Generated' : melody} for i, melody in enumerate(generated_melodies) ]
    with open(f"Generated/{file_name}.json","w") as f:
        json.dump(generated,f,indent=2)
