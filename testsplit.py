import tensorflow as tf
example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(example_texts), mask_token=None)
ids = ids_from_chars(chars)
print(ids)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

print(split_input_target(list("Tensorflow")))