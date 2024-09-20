# This code was written as part of the research documented here: (https://doi.org/10.1109/ICPHM61352.2024.10627589)
# *** Gradient penalty is commented and not in this code but provided for reference. ***


import tensorflow as tf
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dense, LSTM, Reshape, Flatten, ReLU, LeakyReLU, BatchNormalization, Bidirectional, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import spectrogram, istft
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import random
from scipy.io import savemat

# import the datasets

# My dataset's structure was an [m,n] np array  in which m is the individual 1D time-series or signals or sequence counts
# and n was the length of the sequence. Please refer to (https://doi.org/10.1109/ICPHM61352.2024.10627589) for more information.


# *** Load your 1D time-series data here ***
# data =   


latent_dim = 100  # Adjust this to your needs (Controls the generation diversity to some degree)
seq_length = data.shape[1]  # The length of your time series sequences
number_of_synthetic_samples = case_0_i_z.shape[0]-data.shape[0]
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Function for calculating the gradient penalty
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py


def gradient_penalty(real_samples, fake_samples):
    alpha = tf.random.normal([real_samples.shape[0], 1, 1], 0.0, 1.0)
    diff = fake_samples - real_samples
    interpolated = real_samples + alpha * diff

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    gradients = tape.gradient(pred, [interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty

def build_generator(latent_dim, seq_length):
    model = Sequential()

    # Initial sequence length and feature map calculations might require adjustment
    initial_seq_length = 10  # Starting length, which we will upscale from
    initial_feature_maps = 128  # The number of feature maps for the first layer

    # Start with a Dense layer to create a starting sequence
    model.add(Dense(initial_feature_maps * initial_seq_length, activation='relu', input_dim=latent_dim))
    model.add(Reshape((initial_seq_length, initial_feature_maps)))

    # First Conv1DTranspose layer
    model.add(Conv1DTranspose(128, kernel_size=4, strides=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Subsequent Conv1DTranspose layers

    model.add(Conv1DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Depending on the exact sequence length needed, you might adjust the number of feature maps,
    # kernel size, and stride for the following layers
    # More Conv1DTranspose layers to get closer to the target output size
    model.add(Conv1DTranspose(32, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1DTranspose(16, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Last Conv1DTranspose layer to reach the final output shape
    model.add(Conv1DTranspose(1, kernel_size=45, strides=4, padding='valid', activation='linear'))

    return model

def build_discriminator(seq_length):
    model = Sequential()

    # Input is a sequence of length `seq_length`
    model.add(Conv1D(32, kernel_size=5, strides=2, padding='same', input_shape=(seq_length, 1)))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Conv1D(64, kernel_size=5, strides=2, padding='same'))
    model.add(ZeroPadding1D(padding=((0,1))))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Conv1D(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    # Additional Conv1D layer
   # model.add(Conv1D(512, kernel_size=5, strides=2, padding='same'))
   # model.add(BatchNormalization())
   # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1))

    return model

# Define the GAN model

def compile_models(generator, discriminator):
    discriminator.compile(loss=wasserstein_loss, optimizer=discriminator_optimizer, metrics=['accuracy'])
    z = tf.keras.Input(shape=(latent_dim,))
    fake_time_series = generator(z)
    discriminator.trainable = False
    valid = discriminator(fake_time_series)
    combined = tf.keras.Model(z, valid)
    combined.compile(loss=wasserstein_loss, optimizer=generator_optimizer)
    return combined

discriminator_losses = []
generator_losses = []

def train(generator, discriminator, combined, data, epochs, batch_size, latent_dim, seq_length):
    half_batch = int(batch_size / 2)
    n_critic = 4  # Number of discriminator updates per generator update
    clip_value = 0.01  # Clip value for discriminator weights
    gradient_penalty_weight = 5.0  # Weight for gradient penalty

    for epoch in range(epochs):
        for _ in range(n_critic):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of real samples
            idx = np.random.randint(0, data.shape[0], half_batch)
            real_time_series = data[idx]
            real_time_series = np.expand_dims(real_time_series, axis=-1)

            # Generate a half batch of fake samples
            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            fake_time_series = generator.predict(noise)

            # Labels for real and fake samples
            real_labels = -np.ones((half_batch, 1))
            fake_labels = np.ones((half_batch, 1))

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(real_time_series, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_time_series, fake_labels)

            # Calculate the gradient penalty   
            # (This is where I mentioned GP is commented. It is generally recommended that you do not use GP and Weight Clipping
            # at the same time since their design logics are conflicting. It works here though.)
            # gp = gradient_penalty(real_time_series, fake_time_series)

            # Calculate the total discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # + gradient_penalty_weight * gp

            # Clip discriminator weights
            for l in discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = -np.ones((batch_size, 1))  # The generator wants the discriminator to label the generated samples as real

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid_labels)

        # Plot the progress
        print(f"{epoch}/{epochs}, D Loss: {d_loss[0]:.10f}, G Loss: {g_loss:.10f}")
        # Store the losses
        discriminator_losses.append(d_loss)
        generator_losses.append(g_loss)

    print("Training complete.")

generator = build_generator(latent_dim, seq_length)
discriminator = build_discriminator(seq_length)
# Apply the custom weight initialization
# for layer in generator.layers:
#     custom_weights_init(layer)
# for layer in discriminator.layers:
#     custom_weights_init(layer)
combined = compile_models(generator, discriminator)
start_time = time.time()
train(generator, discriminator, combined, data, epochs=40000, batch_size=64, latent_dim=latent_dim, seq_length=seq_length)
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"Training took {elapsed_minutes:.2f} minutes")

# To generate synthetic data
noise = np.random.normal(0, 1, (number_of_synthetic_samples, latent_dim))
synthetic_data = generator.predict(noise)
synthetic_data = np.squeeze(synthetic_data, axis=2)


# Augmenting the dataset
data = np.append(data, synthetic_data, axis=0)

# Deleting the outlier generated signals.
# As every training signal was feature-scaled to be between -1 and 1, the generated signals that exhibit values beyond this range are removed (with a grace leeway).

# It is important to mention that if the WGAN is not carefully designed, the signals that are generated in the following loop
# will never be in the [-1.01,1.01] range

while True:
  rows_to_delete = []
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      if abs(data[i, j]) > 1.01:
        rows_to_delete.append(i)
        break

  data = np.delete(data, rows_to_delete, axis=0)

  if len(rows_to_delete) != 0:
    noise = np.random.normal(0, 1, (len(rows_to_delete), latent_dim))
    synthetic_data = generator.predict(noise)
    synthetic_data = np.squeeze(synthetic_data, axis=2)
    data = np.append(data, synthetic_data, axis=0)
  elif len(rows_to_delete) == 0:
    break

np.random.shuffle(data)


print(f'Final number of the sequences in the augmented dataset: {data.shape[0]}')

discriminator_losses_np = np.array(discriminator_losses)
generator_losses_np = np.array(generator_losses)

## Plotting the losses:

# Directory where you want to save plots
folder_name = 'saved_plots'

# Create the directory, if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

plt.rcParams['axes.titlesize'] = 20  # Title
plt.rcParams['axes.labelsize'] = 18  # X and Y labels
plt.rcParams['xtick.labelsize'] = 16  # X tick labels
plt.rcParams['ytick.labelsize'] = 16  # Y tick labels

# Plotting both losses together:
plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.plot(generator_losses, label='Generator Loss')
plt.title("WGAN-GP Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder_name}/both_losses.svg')
plt.show()

# Plotting Discriminator Loss
plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label='Discriminator Loss', color='blue')
plt.title("Discriminator Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder_name}/disc_losses_mode.svg')
plt.show()

# Plotting Generator Loss
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss', color='red')
plt.title(f"Generator Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder_name}/gen_losses_mode.svg')
plt.show()

sample_index =99
sample = synthetic_data[sample_index]

plt.rcParams['axes.titlesize'] = 20  # Title
plt.rcParams['axes.labelsize'] = 18  # X and Y labels
plt.rcParams['xtick.labelsize'] = 20  # X tick labels
plt.rcParams['ytick.labelsize'] = 20  # Y tick labels
# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the sample
plt.plot(sample)

# Set the title and labels
plt.title(f'Sample {sample_index}')
plt.xlabel('Time steps')
plt.ylabel('Value')
# Setting font size for tick labels
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show the plot
plt.show()

