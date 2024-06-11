import tensorflow as tf
import tensorflow_text as tf_text
import unicodedata
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.framework import dtypes
example_text = tf.constant('¿Todavía está en casa?')
example_text_str = example_text.numpy().decode('UTF-8')
tf_text.normalize_utf8(example_text_str, 'NFKD')
input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        example_text, dtype=dtypes.string)
# Perform NFKD normalization using Python's unicodedata module
normalized_text_str = unicodedata.normalize('NFKD', example_text_str)

# Convert the normalized string back to a TensorFlow tensor
normalized_text_tensor = tf.constant(normalized_text_str)

print("Original text:", example_text.numpy())
print("Normalized text (unicodedata):", normalized_text_tensor.numpy())