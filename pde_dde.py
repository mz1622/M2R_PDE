import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deepxde.backend import tf
 
# Define the PDE
def pde(x, f):
    df_x = dde.grad.jacobian(f, x)
    d2f_xx = dde.grad.hessian(f, x)
    rho = tf.zeros_like(x)
    return -d2f_xx + dde.grad.jacobian(f**2, x) - rho
 
# Define the boundary condition
def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)
 
def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)
 
geom = dde.geometry.Interval(0, 1)
bc = dde.icbc.PeriodicBC(geom, 0, boundary_r)
 

data = dde.data.PDE(
    geom,
    pde,
    [bc],
    num_domain=200,
    num_boundary=2,
    num_test=100
)
 
#Nueral network model
layer_size = [1] + [50] * 3 + [1]
activation = "sin"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
 
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
 
losshistory, train_state = model.train(iterations=10000)
 

X_test = geom.uniform_points(100, True)
y_test = np.zeros((100, 1))
 

y_pred = model.predict(X_test)
 

# Save and Plot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
 
x = geom.uniform_points(500)
y_pred = model.predict(x)
plt.figure()
plt.plot(x, y_pred, label="NN solution")
plt.legend()
plt.show()
 
 