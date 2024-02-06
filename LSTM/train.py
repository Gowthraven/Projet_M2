import tensorflow as tf
from LSTM import *

sparse_categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
optimizer = tf.keras.optimizers.Adam()

def train(train_dataset,lstm_model,epochs):
    print("Training the model ...")
    iters,losses = [],[]
    i = 0
    for epoch in range(epochs):
        total_loss = 0
        for (batch,(input,target)) in enumerate(train_dataset):
            batch_loss = train_step(input,target,lstm_model)
            total_loss = batch_loss
            print(f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}")

            iters.append(i)
            losses.append(batch_loss)

            i += 1
    return iters,losses

def train_step(input,target,lstm_model):
    with tf.GradientTape() as tape:
        predictions = lstm_model(input)
        loss = calculate_loss(target, predictions)
    gradients = tape.gradient(loss, lstm_model.trainable_variables)
    gradient_variable_pairs = zip(gradients, lstm_model.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)
    return loss

def calculate_loss(real, pred):
    loss = sparse_categorical_crossentropy(real, pred)
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    total_loss = tf.reduce_sum(loss)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements
    return average_loss