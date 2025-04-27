import numpy as np
import matplotlib.pyplot as plt

# Parameters
A, B, C = 10, 25, 8/3
dt = 0.01
t_end = 30
steps = int(t_end / dt)

# Derivative function
def derivatives(x, y, z):
    dx = A * (y - x)
    dy = -x * z + B * x - y
    dz = x * y - C * z
    return np.array([dx, dy, dz], dtype=np.float64)

# Soft limiter to prevent instability
def soft_limit(val, threshold=1e3):
    return val if abs(val) < threshold else threshold * np.tanh(val / threshold)

# Apply soft_limit to all elements
def limit_array(arr):
    return np.array([soft_limit(val) for val in arr], dtype=np.float64)

# Single step functions
def euler_step(x, y, z):
    dx, dy, dz = limit_array(derivatives(x, y, z))
    return x + dt * dx, y + dt * dy, z + dt * dz

def midpoint_step(x, y, z):
    dx1, dy1, dz1 = limit_array(derivatives(x, y, z))
    xm, ym, zm = x + dt/2 * dx1, y + dt/2 * dy1, z + dt/2 * dz1
    dx2, dy2, dz2 = limit_array(derivatives(xm, ym, zm))
    return x + dt * dx2, y + dt * dy2, z + dt * dz2

def rk4_step(x, y, z):
    k1 = limit_array(derivatives(x, y, z))
    k2 = limit_array(derivatives(x + dt/2 * k1[0], y + dt/2 * k1[1], z + dt/2 * k1[2]))
    k3 = limit_array(derivatives(x + dt/2 * k2[0], y + dt/2 * k2[1], z + dt/2 * k2[2]))
    k4 = limit_array(derivatives(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2]))
    delta = (k1 + 2*k2 + 2*k3 + k4) / 6
    return x + dt * delta[0], y + dt * delta[1], z + dt * delta[2]

# General solver
def solve(method_func, x0=1, y0=1, z0=1):
    x, y, z = [np.float64(x0)], [np.float64(y0)], [np.float64(z0)]
    for step in range(steps):
        xn, yn, zn = method_func(x[-1], y[-1], z[-1])
        if not np.all(np.isfinite([xn, yn, zn])):
            print(f"Stopped at step {step} due to instability.")
            break
        x.append(xn)
        y.append(yn)
        z.append(zn)
    return np.array(x), np.array(y), np.array(z)

# Solve with each method
x_euler, y_euler, z_euler = solve(euler_step)
x_mid, y_mid, z_mid = solve(midpoint_step)
x_rk4, y_rk4, z_rk4 = solve(rk4_step)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_euler, z_euler, label='Euler Method', alpha=0.7)
plt.plot(x_mid, z_mid, label='Midpoint Method', alpha=0.7)
plt.plot(x_rk4, z_rk4, label='RK4 Method', alpha=0.7)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Plot of z vs x with Stability Enhancements')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
