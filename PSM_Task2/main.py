import numpy as np
import matplotlib.pyplot as plt

v0 = 500  # initial velocity (m/s)
angle = 45  # launch angle in degrees
vx0 = v0 * np.cos(np.radians(angle))
vy0 = v0 * np.sin(np.radians(angle))

g = 9.81  # gravity
m = 2.0   # mass
k = 0.2   # drag coefficient
dt = v0 / 1000  # time step
t_max = 5  # total simulation time

# Euler's Method
def euler_method():
    t = 0
    x, y = 0, 0
    vx, vy = vx0, vy0

    x_vals, y_vals = [x], [y]

    while y >= 0:
        ax = -k * vx / m
        ay = -g - (k * vy / m)

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        x_vals.append(x)
        y_vals.append(y)
        t += dt
        print(y)

    return x_vals, y_vals

# Midpoint Method
def midpoint_method():
    t = 0
    x, y = 0, 0
    vx, vy = vx0, vy0

    x_vals, y_vals = [x], [y]

    while y >= 0:
        ax1 = -k * vx / m
        ay1 = -g - (k * vy / m)

        vx_half = vx + ax1 * (dt / 2)
        vy_half = vy + ay1 * (dt / 2)

        ax2 = -k * vx_half / m
        ay2 = -g - (k * vy_half / m)

        vx += ax2 * dt
        vy += ay2 * dt
        x += vx * dt
        y += vy * dt

        x_vals.append(x)
        y_vals.append(y)
        t += dt

    return x_vals, y_vals

x_euler, y_euler = euler_method()
x_midpoint, y_midpoint = midpoint_method()

plt.figure(figsize=(10, 5))
plt.plot(x_euler, y_euler, label="Euler's Method", linestyle="--", marker="o", markersize=3)
plt.plot(x_midpoint, y_midpoint, label="Midpoint Method", linestyle="-", marker="s", markersize=3)
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.title("Projectile Motion with Air Resistance")
plt.legend()
plt.grid()
plt.show()