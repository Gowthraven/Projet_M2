import tensorflow as tf

class BiLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, max_length_melody):
        super(BiLSTM, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length_melody)
        self.batch_norm = tf.keras.layers.BatchNormalization() 
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.bilstm(x)
        output = self.dense(x)
        return output
