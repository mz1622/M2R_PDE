import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def sine_activation(x):
    return tf.sin(math.pi*x)

NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(units = 32, activation = sine_activation),
    tf.keras.layers.Dense(units = 32, activation = sine_activation),
    tf.keras.layers.Dense(units = 32, activation = sine_activation),
    tf.keras.layers.Dense(units = 1)
])

NN.summary()

optm = tf.keras.optimizers.Adam(learning_rate = 0.001)


def ode_system(t, net):
    t = tf.constant(t.reshape(-1, 1), dtype=tf.float32)
    t_0 = tf.zeros((1, 1))
    t_1 = tf.ones((1, 1))
    zeros = tf.zeros((1, 1))
    const = tf.constant(1.0, dtype=tf.float32)

    # Nested gradient tape for higher-order derivatives
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(t)
        with tf.GradientTape() as tape2:
            tape2.watch(t)
            u = net(t)
            # First derivative within inner tape
        u_t = tape2.gradient(u, t)

    # Second derivative outside inner but within outer tape
    u_tt = tape1.gradient(u_t, t)

    # Derivative of u*u with respect to t
    u_t2 = 2 * u * u_t  # Chain rule

    # Compute losses
    ode_loss = - u_tt + u_t2
    IC_loss = net(t_0) - zeros
    EC_loss = net(t_1) - zeros
    BC_loss = net(t_1) - net(t_0)
    square_loss = tf.square(ode_loss) + tf.square(BC_loss) + tf.square(IC_loss) + tf.square(EC_loss)
    total_loss = tf.reduce_mean(square_loss)

    # Clean up the persistent tape
    del tape1

    return total_loss


train_t = np.linspace(0, 1, 100).reshape(-1, 1)
train_loss_record = []

for itr in range(6000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN)
        train_loss_record.append(train_loss)

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())




# Plot and save the training loss
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record, label='Training Loss')
plt.xlabel('Iteration', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(fontsize=15)
plt.title('Training Loss Over Iterations', fontsize=15)
plt.savefig("training_loss_plot.png")  # Save as PNG file

# Generate predictions and plot
test_t = np.linspace(0, 1, 100)
pred_u = NN.predict(test_t).ravel()

plt.figure(figsize=(10, 8))
plt.plot(test_t, pred_u, '--r', label='Prediction')
plt.plot(test_t, 0 * test_t, '-k', label='True')
plt.legend(fontsize=15)
plt.xlabel('t', fontsize=15)
plt.ylabel('u', fontsize=15)
plt.title('Prediction Plot', fontsize=15)
plt.savefig("sine_wave_plot.png")  # Save this plot as well