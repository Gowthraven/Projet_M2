"""
melody_generator.py

This script defines the MelodyGenerator class, which is responsible for generating
melodies using a trained Transformer model. The class offers functionality to produce
a sequence of musical notes, starting from a given seed sequence and extending it
to a specified maximum length.

The MelodyGenerator class leverages the trained Transformer model's ability to
predict subsequent notes in a melody based on the current sequence context. It
achieves this by iteratively appending each predicted note to the existing sequence
and feeding this extended sequence back into the model for further predictions.

This iterative process continues until the generated melody reaches the desired length
or an end-of-sequence token is predicted. The class utilizes a tokenizer to encode and
decode note sequences to and from the format expected by the Transformer model.

Key Components:
- MelodyGenerator: The primary class defined in this script, responsible for the
  generation of melodies.

Usage:
The MelodyGenerator class can be instantiated with a trained Transformer model
and an appropriate tokenizer. Once instantiated, it can generate melodies by
calling the `generate` method with a starting note sequence.

Note:
This class is intended to be used with a Transformer model that has been
specifically trained for melody generation tasks.
"""

import tensorflow as tf
import numpy as np


class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, tokenizer, max_length=100):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence, forcing=0, melody=[],mode=2, temperature=1, k=20 , first_proba=0.8):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        proba=[]
        input_tensor = self._get_input_tensor(start_sequence)
        true_tensor = self._get_input_tensor(melody)
        
        
        num_notes_to_generate = self.max_length - len(input_tensor[0])

        if forcing > 0:
            random = np.random.rand(num_notes_to_generate)
        else:
            random = None
        
        
        for i in range(1,num_notes_to_generate+1):
            if (random is None) or (random is not None and i+1 < len(random) and random[i] > forcing):
                predictions = self.transformer(
                        input_tensor, input_tensor, False, None, None, None
                        )
            else:
                predictions = self.transformer(
                        true_tensor[:i-1+len(start_sequence)], true_tensor[:i-1+len(start_sequence)], False, None, None, None
                        )
                      
            
            
            if (predictions.shape[1]==0):
                print(start_sequence)
            if mode==0:
              predicted_note = self._get_note_with_highest_score(predictions,proba)
            elif mode==1:
              predicted_note = self._get_note_with_proba_temperature(predictions,temperature,proba)
            elif mode==2:
              predicted_note = self._get_note_with_k_sampling(predictions,k,proba,first_proba)


            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)



        return generated_melody,[1]*len(start_sequence)+proba

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor


    def _get_note_with_highest_score(self,predictions,proba):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]
        predicted_note_index = tf.argmax(latest_predictions, axis=1)
        probas = np.exp(latest_predictions) / np.sum(np.exp(latest_predictions))
        proba.append(probas[0][predicted_note_index])
        predicted_note = tf.argmax(latest_predictions, axis=1)
        predicted_note = predicted_note.numpy().item()
        return predicted_note

    def _get_note_with_proba_temperature(self, predictions,T,proba):
        """
        Gets the note with the categorical score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]/T
        probas = np.exp(latest_predictions) / np.sum(np.exp(latest_predictions))
        sorted_indices = np.argsort(probas[0])[::-1]
        s=np.sum(probas[0][sorted_indices[:3]])


        notes=tf.convert_to_tensor([sorted_indices[:3]], dtype=tf.int64)
        notes= self._decode_generated_sequence(notes)

        predicted_note= tf.random.categorical(latest_predictions, num_samples=1)
        predicted_note = predicted_note.numpy().item()
        proba.append(probas[0][predicted_note])

        return predicted_note

    def _get_note_with_k_sampling(self, predictions,k,proba,first_proba):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.
            k (int) : top K sampling
        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]
        probas = np.exp(latest_predictions) / np.sum(np.exp(latest_predictions))
        sorted_indices = np.argsort(probas[0])[::-1]
        
        s=sum(probas[0][sorted_indices[:1]])
        T=1
        while not( s < first_proba) and T<3:
            latest_predictions = predictions[:, -1, :]/T
            probas = np.exp(latest_predictions) / np.sum(np.exp(latest_predictions))
            sorted_indices = np.argsort(probas[0])[::-1]
            s=np.sum(probas[0][sorted_indices[:1]])
            T+=0.1

        notes=tf.convert_to_tensor([sorted_indices], dtype=tf.int64)
        notes= self._decode_generated_sequence(notes)
        
        top_k_indices= sorted_indices[:k]
        top_k_proba= probas[0][top_k_indices]/np.sum(probas[0][top_k_indices])

        predicted_note= np.random.choice(top_k_indices, p=top_k_proba, replace=False)

        selected_index= np.where(top_k_indices == predicted_note)[0]

        proba.append(top_k_proba[selected_index][0])
        return predicted_note

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.numpy()
        generated_melody = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_melody
    
def generate_seq(original,lg_debut,lg_predict):
    original_size= len(original)
    print(f'Nombre de mesures dans l\'original : {original_size}')
    size=lg_debut+lg_predict
    debuts=[]
    fins=[]
    for i in range(size,original_size+1):
        seq_debut= [ n for  m in original[i-size:i-lg_predict] for n in m]
        seq_originale= [ n for  m in original[i-size:i] for n in m if i<original_size]
        debuts.append(seq_debut)
        fins.append(seq_originale)
    return debuts,fins,original_size

def generate_decalage(melody_generator,original,lg_debut,lg_predict,mode=2,k=40,first_proba=0.8,time_signature="2/4"):
    '''Genere une partie 
      Parametres:
            original (liste de string) : Partie originale
            lg_debut (int) : Longueur sequence de départ en mesure
            lg_fin (int) : Longueur sequence prédite en mesure
            ***param du melody_generator.generate()
    '''
    melodie=[]
    debuts,fins,original_size=generate_seq(original,lg_debut,lg_predict)
    melodie.append(debuts[0])
    probas= [[ 1 for m in debuts[0] for _ in m ]]
    size=lg_debut+lg_predict
    for i in range(len(debuts)):
      new_melodie,p=melody_generator.generate(debuts[i],mode=mode,k=k)
      new_melodie=new_melodie.split(' ')[len(debuts[i]):]
      new_melodie=n_measure(new_melodie,lg_predict,time_signature)
      melodie.append(new_melodie)
      p= p[len(debuts[i]):len(debuts[i])+len(new_melodie)]
      #print(new_melodie)
      probas.append(p)
      print(f'Mesure {size+i} : generated')
    melodie = [ n for measure in melodie for n in measure ]
    print("Melodie generated")
    return melodie,probas

def n_measure(melody,n,time_signature):
    measureDuration = float(time_signature.split('/')[0])
    totalTime = 0.0
    for i in range(len(melody)):
        duration=float(melody[i].split("-")[-1])
        totalTime+=duration
        if totalTime==measureDuration*n:
            return melody[:i+1]
        elif totalTime>measureDuration*n:
            last_state=melody[i].split("-")
            last_state="-".join(last_state[:-1])+"-"+str(float(last_state[-1])-(totalTime-(measureDuration*n)))
            melody[i]=last_state
            return melody[:i+1]
    return melody
    
