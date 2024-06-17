import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
import math
import torch

tf.keras.backend.set_floatx('float64')

def sine_activation(x):
    return tf.sin(2 * math.pi * x)
    

class SineActivation(Layer):
    def __init__(self):
        super(SineActivation, self).__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2*math.pi*inputs), tf.cos(2*math.pi*inputs)], 1)

NN = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        SineActivation(),
        tf.keras.layers.Dense(units=64, activation='sigmoid'),
        tf.keras.layers.Dense(units=1),
    ])

NN.summary()

optm = tf.keras.optimizers.Adam(learning_rate=0.001)

def rho_sin(t):
    return tf.sin(2*math.pi*t)




def ode_system(t, rho, net):
    t = tf.reshape(t, [-1, 1])
    t_0 = tf.zeros((1, 1), dtype=tf.float64)
    t_1 = tf.ones((1, 1), dtype=tf.float64)
    zeros = tf.zeros((1, 1), dtype=tf.float64)

    with tf.GradientTape() as tape1:
        tape1.watch(t)
        with tf.GradientTape() as tape2:
            tape2.watch(t)
            u = net(t)
        u_t = tape2.gradient(u, t)

    u_tt = tape1.gradient(u_t, t)
    u_t2 = 2 * u * u_t  # Chain rule for derivative of u*u with respect to t
    '''
    indices = [0, 10, 20]
    for idx in indices:
        print(f"u_tt[{t[idx,0].numpy()}]: {u_tt[idx,0].numpy()}, u_t2[{t[idx,0].numpy()}]: {u_t2[idx,0].numpy()}")
    '''

    ode_loss = - u_tt + u_t2  - rho(t)   # for non-trivial solution, add rho(t)
    IC_loss = net(t_0) - 0  # Initial condition
    square_loss = tf.square(ode_loss) 
    total_loss = tf.reduce_mean(square_loss)

    del tape1, tape2

    return total_loss, ode_loss






def t_function(t):
    return (1/280) * tf.sin(2 * np.pi * t)
test_t = np.linspace(0, 1, 100).astype(np.float64)
total, ode = ode_system(test_t, rho_sin, t_function)
#print mean of ode.numpy()






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

    del tape
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
    _, residuals = ode_system(t, rho_sin, net)
    return residuals



## TODO Bound for the integral
def lower_bound_int(net):
    
    x_values = np.linspace(0, 1, 1000).astype(np.float64)  # Using left endpoints for lower sum
    f_values = net.predict(x_values).ravel()
    k = NN_derivative(x_values, net)

    # Calculate the sum
    lower_sum = -np.sum(k)/1000000 + np.sum(f_values) /1000
    
    return lower_sum

def upper_bound_int(net):
    

    x_values = np.linspace(0,1,1000).astype(np.float64)  # Using left endpoints for lower sum
    k = NN_derivative(x_values, net)

    # Evaluate model at these points
    f_values = net.predict(x_values).ravel()
    
    # Calculate the sum
    upper_sum = np.sum(k)/1000000 + np.sum(f_values) /1000
    
    return upper_sum


# Start training
train_t = np.linspace(0, 1, 1001).reshape(-1, 1).astype(np.float64)
train_loss_record = []

optm = tf.keras.optimizers.Adam(learning_rate=0.001)

for itr in range(1001):
    with tf.GradientTape() as tape:
        train_loss, _ = ode_system(tf.constant(train_t),rho_sin, NN)
        train_loss_record.append(train_loss.numpy())

        grad_w = tape.gradient(train_loss, NN.trainable_variables)
        optm.apply_gradients(zip(grad_w, NN.trainable_variables))

    if itr % 1000 == 0:
        print(train_loss.numpy())


# Save the model
NN.save('best_model.h5')

# Plot training loss
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Iterations')
plt.savefig("training_loss_plot.png")  # Save plot


# Error Analysis
x_i = np.linspace(0, 1, 1001).astype(np.float64)

lipschitz_constant = Lipschitz_constant(x_i, NN) # Calculate the Lipschitz constant

error_bounds = []   #Find the error bounds
for i,x in enumerate(x_i):
    error_bound = lipschitz_constant[i][0] * 1/1000
    error_bounds.append(error_bound)

error_bounds = np.array(error_bounds)

residuals = calculate_residuals(x_i, NN).numpy().ravel()

# |F(f_approx)(x)| < | F(f)(x_i)| (Known) + | F(f)(x) - F(f)(x_i) |
total_errors = np.abs(residuals) + error_bounds[:len(residuals)]

errors_in_l2 = np.sum(total_errors ** 2) * 1/1000
print(f"Error in L2: {errors_in_l2}")

#Plot the error bounds
plt.figure(figsize=(10, 8))
plt.plot(x_i[:len(residuals)], residuals, '--r', label=r'bound for |F($f_{approx} (x_k)$)|')
plt.plot(x_i[:len(residuals)], error_bounds[:len(residuals)], ':b', label=r'bound for |$F(f_{approx})(x_k) - F(f_{approx})(x)$|')
plt.plot(x_i[:len(residuals)], total_errors, '-g', label=r'Total Error: $\delta_k$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Error Analysis(n = 1000)')
plt.savefig("Error.png")


# Generate and plot predictions
test_t = np.linspace(0, 1, 100).astype(np.float64)
pred_u = NN.predict(test_t).ravel()

int_bound =  lower_bound_int(NN)
print(int_bound)
plt.figure(figsize=(10, 8))
plt.plot(test_t, pred_u, '-.r', label=r'$f_{approx}(x)$')
plt.plot(test_t,  pred_u - int_bound, '--g', label=r'$\tilde{f} (x)$')
#plt.plot(test_t, 0 * test_t, '--k', label='zero_solution')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximate solution Plot')
plt.savefig("PINN_result.png")  # Save plot


plt.figure(figsize=(10, 8))
#plt.plot(test_t, 0 * test_t , '-k', label='Zero Solution', linewidth=3, alpha=0.8)
plt.plot(test_t, pred_u, '--r', label=r'$\tilde{f}(x)$', linewidth=1, alpha=1)
plt.ylim(-1, 1)
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\tilde{f} (x)$')
plt.title('Approximate solution Plot')
plt.savefig("result_axis.png")

