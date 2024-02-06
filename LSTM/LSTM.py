import tensorflow as tf

class LSTMModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, max_length_melody):
        super(LSTMModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length_melody)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x = self.lstm(x)
        output = self.dense(x)
        return output