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

## TODO For the Neural Network above, find the bound for the derivative of F_approx (using Summation)
def spectral_norm(matrix):
    singular_values = tf.linalg.svd(matrix, compute_uv=False)
    return tf.reduce_max(singular_values)

# Function to calculate the Lipschitz constant for the neural network
def calculate_lipschitz_constant(model):
    max_derivative = 2 * np.pi  # Maximum absolute value of derivative of sine(2πx)
    lipschitz_constant = 1.0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()[0]
            norm = spectral_norm(weights)
            lipschitz_constant *= norm * max_derivative
    return lipschitz_constant

# Calculate the Lipschitz constant
lipschitz_constant = calculate_lipschitz_constant(NN)
print(f"Lipschitz constant: {lipschitz_constant}")


## TODO Lipshitz methods
# Define the points where we need to evaluate the error
x_points = np.linspace(0, 1, 1001).astype(np.float64)
x_i = x_points[::100]  # For simplicity, taking every 100th point as x_i

# Calculate the difference bound
error_bounds = []
for x in x_points:
    nearest_x_i = min(x_i, key=lambda xi: abs(x - xi))  # Find the nearest x_i
    error_bound = lipschitz_constant * abs(x - nearest_x_i)
    error_bounds.append(error_bound)

# Convert to numpy array for easier manipulation
error_bounds = np.array(error_bounds)

# Evaluate the model at these points
predictions = NN.predict(x_points).ravel()

# Calculate the total error bound
total_errors = np.abs(predictions) + error_bounds

# Print or plot the error bounds and predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(x_points, predictions, '--r', label='Prediction')
plt.plot(x_points, error_bounds, ':b', label='Error Bound')
plt.plot(x_points, total_errors, '-g', label='Total Error')
plt.xlabel('x')
plt.ylabel('F(f_approx)(x)')
plt.legend()
plt.title('Prediction and Error Bound')
plt.show()


NN.save('best_model.h5')


plt.figure(figsize=(10, 8))
plt.plot(test_t, pred_u, '--r', label='Prediction')
plt.plot(test_t, 0 * test_t, '-k', label='True')
plt.legend()
plt.xlabel('t')
plt.ylabel('u')
plt.title('Prediction Plot')
plt.savefig("PINN_result.png")  # Save plot
