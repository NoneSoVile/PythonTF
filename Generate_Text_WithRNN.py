import tensorflow as tf

import numpy as np
import os
import time
import argparse

def get_directory_part(path):
    """
    Returns the directory part of the specified path.
    
    :param path: str, the full or relative path
    :return: str, the directory part of the path
    """
    directory_part = os.path.dirname(path)
    return directory_part
  
parser = argparse.ArgumentParser(description='rnn text predict')

EPOCHS = 2
saved_weights_path = './rnn_saved_weights/rnn_test_weights.h5'
saved_model_path = 'rnn_saved_model'
parser.add_argument('--epochs', default=EPOCHS, type=int)
parser.add_argument('--savedweights', default=saved_weights_path, type=str)
parser.add_argument('--savedmodel', default=saved_model_path, type=str)
args = parser.parse_args()
EPOCHS = int(args.epochs)
saved_weights_path = args.savedweights
saved_model_path = args.savedmodel

# Check if the directory already exists
weightspath = get_directory_part(saved_weights_path)
if not os.path.exists(weightspath):
    # Create the directory
    os.makedirs(weightspath)
    print(f'Directory "{weightspath}" created.')
else:
    print(f'Directory "{weightspath}" already exists.')
    
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
# Define your relative file path
relative_file_path = './shakespeare.txt'

# Convert the relative path to an absolute path
absolute_file_path = os.path.abspath(relative_file_path)
path_to_file = tf.keras.utils.get_file(absolute_file_path, 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')
# Take a look at the first 250 characters in text
print(text[:250])
# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

ids = ids_from_chars(vocab)
print(ids)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
chars = chars_from_ids(ids)
print(chars)
restore_chars = tf.strings.reduce_join(chars, axis=-1).numpy()
print(restore_chars)
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
print(all_ids[:100])
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))

for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

print(dataset)

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                      return_sequences=True,
                                      return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
      

    def call(self, inputs, states=None, return_state=False, training=False):
      x = inputs
      x = self.embedding(x, training=training)
      if states is None:
        states = self.gru.get_initial_state(x)
      x, states = self.gru(x, initial_state=states, training=training)
      x = self.dense(x, training=training)

      if return_state:
        return x, states
      else:
        return x
      
  
if os.path.exists(saved_model_path):
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'MyModel': MyModel})
elif os.path.exists(saved_weights_path):
    model = MyModel()
    model.load_weights(saved_weights_path)
else:
  model = MyModel()
print(f"model configuration.. vocab_size : {vocab_size}, embedding_dim : {embedding_dim}, rnn_units: {rnn_units}")

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print(f"predication sampled_indices: {sampled_indices}")


print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", example_batch_mean_loss)
tf.exp(example_batch_mean_loss).numpy()
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './rnn_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)





history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
model.save_weights(saved_weights_path)
model.save(saved_model_path)
print(f"weights saved to {saved_weights_path}.")
print(f"model saved to {saved_model_path}.")

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
  

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)



result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)







