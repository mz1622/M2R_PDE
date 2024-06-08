import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# Set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

def sine_activation(x):
    return tf.sin(2*math.pi*x)

NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(units=32, activation=sine_activation),
    tf.keras.layers.Dense(units=32, activation=sine_activation),
    tf.keras.layers.Dense(units=1)
])

NN.summary()

optm = tf.keras.optimizers.Adam(learning_rate=0.001)

def ode_system(t, net):
    t = tf.constant(t.reshape(-1, 1), dtype=tf.float64)
    t_0 = tf.zeros((1, 1), dtype=tf.float64)
    t_1 = tf.ones((1, 1), dtype=tf.float64)
    zeros = tf.zeros((1, 1), dtype=tf.float64)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(t)
        with tf.GradientTape() as tape2:
            tape2.watch(t)
            u = net(t)
        u_t = tape2.gradient(u, t)

    u_tt = tape1.gradient(u_t, t)
    u_t2 = 2 * u * u_t  # Chain rule for derivative of u*u with respect to t

    ode_loss = - u_tt + u_t2
    IC_loss = net(t_0) - zeros
    EC_loss = net(t_1) - zeros
    BC_loss = net(t_1) - net(t_0)

    square_loss = tf.square(ode_loss) + tf.square(BC_loss) + tf.square(IC_loss) + tf.square(EC_loss)
    total_loss = tf.reduce_mean(square_loss)

    del tape1  # Clean up

    return total_loss

train_t = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float64)
train_loss_record = []

for itr in range(2000):
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN)
        train_loss_record.append(train_loss.numpy())

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())

# Plot training loss
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Iterations')
plt.savefig("training_loss_plot.png")  # Save plot

# Generate and plot predictions
test_t = np.linspace(0, 1, 100).astype(np.float64)
pred_u = NN.predict(test_t).ravel()

## TODO find the bound for the derivative of F_approx
## TODO Write the above Lipshitz methos



plt.figure(figsize=(10, 8))
plt.plot(test_t, pred_u, '--r', label='Prediction')
plt.plot(test_t, 0 * test_t, '-k', label='True')
plt.legend()
plt.xlabel('t')
plt.ylabel('u')
plt.title('Prediction Plot')
plt.savefig("sine_wave_plot.png")  # Save plot
