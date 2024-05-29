import tensorflow as tf
import argparse
import os

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to a specific checkpoint to restore')
parser.add_argument('--max_checkpoints', type=int, default=3, help='Maximum number of checkpoints to keep')
parser.add_argument('--save_model_path', type=str, default='./saved_model', help='Path to save the trained model')
parser.add_argument('--save_weights_path', type=str, default='./model_weights', help='Path to save the trained model weights')
parser.add_argument('--load_weights_path', type=str, help='Path to load the model weights')
parser.add_argument('--load_model_path', type=str, help='Path to load the entire model')
args = parser.parse_args()

print("TensorFlow version:", tf.__version__)

from keras.layers import Dense, Flatten, Conv2D
from keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
if args.load_model_path:
    model = tf.keras.models.load_model(args.load_model_path)
    print(f"Model loaded from {args.load_model_path}")
else:
    model = MyModel()

if args.load_weights_path and not args.load_model_path:
    model.load_weights(args.load_weights_path)
    print(f"Weights loaded from {args.load_weights_path}")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Define a checkpoint manager
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=args.max_checkpoints)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
  
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

def save_checkpoint():
    checkpoint_manager.save()
    print("Checkpoint saved.")

def load_checkpoint(checkpoint_path=None):
    if checkpoint_path:
        checkpoint.restore(checkpoint_path)
    else:
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
    print(f"Checkpoint restored from {checkpoint_path if checkpoint_path else latest_checkpoint}.")

def save_model(save_path):
    model.save(save_path)
    print(f"Model saved to {save_path}.")

def save_model_weights(save_path):
    model.save_weights(save_path)
    print(f"Model weights saved to {save_path}.")

# Load the checkpoint if provided
if args.checkpoint:
    load_checkpoint(args.checkpoint)
else:
    load_checkpoint()

EPOCHS = 2

for epoch in range(EPOCHS):
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():0.2f}, '
        f'Accuracy: {train_accuracy.result() * 100:0.2f}, '
        f'Test Loss: {test_loss.result():0.2f}, '
        f'Test Accuracy: {test_accuracy.result() * 100:0.2f}'
    )

    save_checkpoint()

# Save the entire model after training
save_model(args.save_model_path)

# Save only the model weights after training
save_model_weights(args.save_weights_path)

print("end")
