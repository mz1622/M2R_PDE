import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# Set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

def sine_activation(x):
    return tf.sin(2 * math.pi * x)

NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(units=32, activation=sine_activation),
    tf.keras.layers.Dense(units=1)
])

NN.summary()

optm = tf.keras.optimizers.Adam(learning_rate=0.001)

def ode_system(t, net):
    t = tf.reshape(t, [-1, 1])
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

    ode_loss = - u_tt + u_t2 + tf.sin(t/10)
    IC_loss = net(t_0) - zeros
    EC_loss = net(t_1) - zeros
    BC_loss = net(t_1) - net(t_0)

    square_loss = tf.square(ode_loss) + tf.square(BC_loss)
    total_loss = tf.reduce_mean(square_loss)

    del tape1  # Clean up

    return total_loss, ode_loss

train_t = np.linspace(0, 1, 1001).reshape(-1, 1).astype(np.float64)
train_loss_record = []

for itr in range(3000):
    with tf.GradientTape() as tape:
        train_loss, _ = ode_system(tf.constant(train_t), NN)
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
def NN_derivative(t, net):
    """
    Compute the derivative of the neural network 
    """
    # Ensure t is a tensor and has the correct shape
    t = tf.convert_to_tensor(t, dtype=tf.float64)
    t = tf.reshape(t, [-1, 1])
    
    with tf.GradientTape() as tape:
        tape.watch(t)
        u = net(t)
    
    u_t = tape.gradient(u, t)
    return u_t

def F_derivative(t, net):
    """
    Compute the value of the derivative function:
    - d^3f/dx^3 + 2 * ((df/dx)^2 + f * d^2f/dx^2)
    """
    t = tf.convert_to_tensor(t, dtype=tf.float64)
    t = tf.reshape(t, [-1, 1])
    
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(t)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                f = net(t)
            df_dx = tape1.gradient(f, t)
        d2f_dx2 = tape2.gradient(df_dx, t)
    d3f_dx3 = tape3.gradient(d2f_dx2, t)
    
    result = -d3f_dx3 + 2 * ((df_dx ** 2) + f * d2f_dx2)
    
    del tape1, tape2, tape3  # Clean up
    
    return result

def Lipschitz_constant(t, net):
    constant = NN_derivative(t, net) * F_derivative(t, net)
    return constant


## TODO Lipshitz methods

# Function to calculate the residuals
def calculate_residuals(t, net):
    t_tensor = tf.constant(tf.reshape(t, [-1, 1]), dtype=tf.float64)
    _, residuals = ode_system(t_tensor, net)
    return residuals



# Define the points where we need to evaluate the error
x_points = np.linspace(0, 1, 1001).astype(np.float64)
x_i = x_points

# Calculate the Lipschitz constant
lipschitz_constant = Lipschitz_constant(x_i, NN)
print(f"Lipschitz constant: {lipschitz_constant}")

# Calculate the difference bound
error_bounds = []
for i,x in enumerate(x_points):
    error_bound = lipschitz_constant[i][0] * 1/1000
    error_bounds.append(error_bound)

error_bounds = np.array(error_bounds)

residuals = calculate_residuals(x_i, NN).numpy().ravel()

# |F(f_approx)(x)| < | F(f)(x_i)| (Known) + | F(f)(x) - F(f)(x_i) |
total_errors = np.abs(residuals) + error_bounds[:len(residuals)]


plt.figure(figsize=(10, 8))
plt.plot(x_points[:len(residuals)], residuals, '--r', label='Residuals')
plt.plot(x_points[:len(residuals)], error_bounds[:len(residuals)], ':b', label='Error Bound')
plt.plot(x_points[:len(residuals)], total_errors, '-g', label='Total Error')
plt.xlabel('x')
plt.ylabel('F(f_approx)(x)')
plt.legend()
plt.title('Prediction and Error Bound')
plt.savefig("Error.png")

NN.save('best_model.h5')

plt.figure(figsize=(10, 8))
plt.plot(test_t, pred_u, '--r', label='Prediction')
plt.plot(test_t, 0 * test_t, '-k', label='zero_solution')
plt.legend()
plt.xlabel('t')
plt.ylabel('u')
plt.title('Prediction Plot')
plt.savefig("PINN_result.png")  # Save plot
