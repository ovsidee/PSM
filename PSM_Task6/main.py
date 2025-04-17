import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
L = np.pi
n_segments = 10
c = 1.0
dx = L / n_segments
dt = 0.01
T_max = 10.0
steps = int(T_max / dt)

# Space discretization
x = np.linspace(0, L, n_segments + 1)
u = np.zeros_like(x)
v = np.zeros_like(x)
u[1:-1] = np.sin(x[1:-1])  # Initial condition

# Energy tracking
E_kin, E_pot, E_total, t_vals = [], [], [], []

# Snapshots
snapshot_secs = [0, 2, 3, 4, 5, 6, 7, 8]
snapshot_steps = {int(s / dt): s for s in snapshot_secs}
snapshots = []

# Time evolution loop
for t_step in range(steps):
    # Acceleration from wave equation
    a = np.zeros_like(u)
    a[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

    # Velocity at half step
    v_half = v + 0.5 * dt * a
    u += dt * v_half

    # Boundary conditions
    u[0] = u[-1] = 0

    # Recompute acceleration and update full velocity
    a[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    v = v_half + 0.5 * dt * a

    # Energy calculations
    KE = 0.5 * np.sum(v**2) * dx
    grad_u = (u[1:] - u[:-1]) / dx
    PE = 0.5 * np.sum(grad_u**2) * dx
    total_E = KE + PE

    E_kin.append(KE)
    E_pot.append(PE)
    E_total.append(total_E)
    t_vals.append(t_step * dt)

    # Store snapshots
    if t_step in snapshot_steps:
        snapshots.append(u.copy())

# Plotting
fig, (ax_energy, ax_wave) = plt.subplots(2, 1, figsize=(10, 10))

# Energy plot
ax_energy.plot(t_vals, E_kin, label='Kinetic Energy')
ax_energy.plot(t_vals, E_pot, label='Potential Energy')
ax_energy.plot(t_vals, E_total, label='Total Energy')
ax_energy.set_title("Dynamics of Energy Components")
ax_energy.set_xlabel("Time (s)")
ax_energy.set_ylabel("Energy")
ax_energy.legend()
ax_energy.grid(True)

# Wave snapshots
for i, profile in enumerate(snapshots):
    ax_wave.plot(x, profile, label=f't = {snapshot_secs[i]}s')
ax_wave.set_title("Wave Profile Evolution")
ax_wave.set_xlabel("Position (x)")
ax_wave.set_ylabel("Amplitude")
ax_wave.legend()
ax_wave.grid(True)

plt.tight_layout()
plt.show()
