import tensorflow as tf
import numpy as np
import sys


class MelodyGenerator:
    """
    Class to generate melodies using a trained LSTM model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, lstm, tokenizer, max_length=100):
        """
        Initializes the MelodyGenerator.

        Parameters:
            lstm (LSTM): The trained LSTM model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.lstm = lstm
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence, teacher_forcing=False, melody=[], mode=2, temperature=1, k=20):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        proba=[]
        input_tensor = self._get_input_tensor(start_sequence)

        if teacher_forcing:
            if melody==[]:
                print("The given melody is empty while teacher forcing activated.")
                sys.exit()
            else:
                true_tensor = self._get_true_tensor(melody[:len(start_sequence)+k])

        num_notes_to_generate = self.max_length - len(input_tensor[0])

        for i in range(num_notes_to_generate):

            if teacher_forcing:
                predictions = self.lstm(true_tensor[:i+len(start_sequence)])
            else:
                predictions = self.lstm(input_tensor)

            if mode==0:
              predicted_note = self._get_note_with_highest_score(predictions,proba)
            elif mode==1:
              predicted_note = self._get_note_with_proba_temperature(predictions,temperature,proba)
            elif mode==2:
              predicted_note = self._get_note_with_k_sampling(predictions,k,proba)
            
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody,[1]*len(start_sequence)+proba

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the LSTM model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_true_tensor(self, true_sequence):

        true_sequence = self.tokenizer.texts_to_sequences([true_sequence])
        true_tensor = tf.convert_to_tensor(true_sequence, dtype=tf.int64)
        return true_tensor

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

    def _get_note_with_k_sampling(self, predictions,k,proba):
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
        s=np.sum(probas[0][sorted_indices[:3]])
        T=1
        while not(0.5< s < 0.6) and T<3:
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
