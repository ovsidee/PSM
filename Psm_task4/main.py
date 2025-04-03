import matplotlib.pyplot as plt
import numpy as np

# Constants
g = 9.81
alpha = np.radians(30)
m = 1.0  #mass
r = 0.1  #radius
h = 2.0  #height
s0 = 0.0  #osition
v0 = 0.0  #velocity
theta0 = 0.0  #angular position
omega0 = 0.0  #angular velocity
dt = 0.01  #time step
t_max = 2.0
num_steps = int(t_max / dt)

# Moments of inertia
I_solid = (2 / 5) * m * (r ** 2)
I_hollow = (2 / 3) * m * (r ** 2)



def acceleration(I):
    return g * np.sin(alpha) / (1 + (I / (m * r ** 2)))


#Midpoint method
def simulate_rolling(I):
    a = acceleration(I)

    # Initialize arrays
    t_vals = np.linspace(0, t_max, num_steps)
    s_vals = np.zeros(num_steps)
    theta_vals = np.zeros(num_steps)
    x_vals = np.zeros(num_steps)
    y_vals = np.zeros(num_steps)
    PE_vals = np.zeros(num_steps)
    KE_trans_vals = np.zeros(num_steps)
    KE_rot_vals = np.zeros(num_steps)
    TE_vals = np.zeros(num_steps)

    s = s0
    v = v0
    theta = theta0
    omega = omega0

    for i in range(num_steps):
        # Midpoint method update
        v_mid = v + 0.5 * dt * a
        omega_mid = omega + 0.5 * dt * (a / r)
        s += dt * v_mid
        theta += dt * omega_mid
        v += dt * a
        omega += dt * (a / r)


        x = s * np.cos(alpha)
        y = h - s * np.sin(alpha)

        # Store values
        s_vals[i] = s
        theta_vals[i] = theta
        x_vals[i] = x
        y_vals[i] = y
        PE_vals[i] = m * g * y
        KE_trans_vals[i] = 0.5 * m * v ** 2
        KE_rot_vals[i] = 0.5 * I * omega ** 2
        TE_vals[i] = PE_vals[i] + KE_trans_vals[i] + KE_rot_vals[i]

    return t_vals, x_vals, y_vals, theta_vals, PE_vals, KE_trans_vals, KE_rot_vals, TE_vals


#Run simulations
t_solid, x_solid, y_solid, theta_solid, PE_solid, KE_trans_solid, KE_rot_solid, TE_solid = simulate_rolling(I_solid)
t_hollow, x_hollow, y_hollow, theta_hollow, PE_hollow, KE_trans_hollow, KE_rot_hollow, TE_hollow = simulate_rolling(
    I_hollow)

#trajectory
plt.figure(figsize=(6, 4))
plt.plot(x_solid, y_solid, label="Solid Sphere")
plt.plot(x_hollow, y_hollow, label="Hollow Sphere", linestyle="dashed")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory of Center of Mass")
plt.legend()
plt.grid()
plt.show()

#mass position and rotation
plt.figure(figsize=(6, 4))
plt.plot(t_solid, x_solid, label="Solid Sphere - Position")
plt.plot(t_hollow, x_hollow, label="Hollow Sphere - Position", linestyle="dashed")
plt.plot(t_solid, theta_solid, label="Solid Sphere - Rotation")
plt.plot(t_hollow, theta_hollow, label="Hollow Sphere - Rotation", linestyle="dashed")
plt.xlabel("Time (s)")
plt.ylabel("Position (m) / Rotation (rad)")
plt.title("Center of Mass Position & Rotation")
plt.legend()
plt.grid()
plt.show()

#energy distribution
plt.figure(figsize=(6, 4))
plt.plot(t_solid, PE_solid, label="Solid - Potential Energy")
plt.plot(t_solid, KE_trans_solid, label="Solid - Kinetic Energy (Trans)")
plt.plot(t_solid, KE_rot_solid, label="Solid - Kinetic Energy (Rot)")
plt.plot(t_solid, TE_solid, label="Solid - Total Energy", linestyle="dashed")

plt.plot(t_hollow, PE_hollow, label="Hollow - Potential Energy", linestyle="dotted")
plt.plot(t_hollow, KE_trans_hollow, label="Hollow - Kinetic Energy (Trans)", linestyle="dotted")
plt.plot(t_hollow, KE_rot_hollow, label="Hollow - Kinetic Energy (Rot)", linestyle="dotted")
plt.plot(t_hollow, TE_hollow, label="Hollow - Total Energy", linestyle="dashed")

plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Energy Distribution Over Time")
plt.legend(loc="lower left", fontsize=6)
plt.grid()
plt.show()
