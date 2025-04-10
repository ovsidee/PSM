import numpy as np
import matplotlib.pyplot as plt

G = 6.6743e-11                  # Gravitational constant [Nm^2/kg^2]
M_sun = 1.989e30                # Mass of the Sun [kg]
M_earth = 5.972e24              # Mass of the Earth [kg]
M_moon = 7.347e22               # Mass of the Moon [kg]
R_earth_sun = 1.5e11            # Distance Earth-Sun [m]
R_earth_moon = 384400e3         # Distance Earth-Moon [m]
T_earth = 365.25 * 24 * 3600    # Orbital period of Earth [s]
T_moon = 27.32 * 24 * 3600      # Orbital period of Moon [s]

omega_earth = 2 * np.pi / T_earth # Angular velocity Earth
omega_moon = 2 * np.pi / T_moon # Angular velocity Moon

dt = 100 # Time step [s]
total_time = T_earth # Simulate one Earth year
steps = int(total_time / dt)

x_earth = np.zeros(steps)
y_earth = np.zeros(steps)
x_moon = np.zeros(steps)
y_moon = np.zeros(steps)

x_earth[0] = R_earth_sun
y_earth[0] = 0
x_moon[0] = x_earth[0] + R_earth_moon
y_moon[0] = 0

vx_earth, vy_earth = 0, omega_earth * R_earth_sun
vx_moon = vx_earth
vy_moon = vy_earth + omega_moon * R_earth_moon

def gravity_accel(x, y, m_source, x_src, y_src):
    dx = x - x_src
    dy = y - y_src
    r = np.hypot(dx, dy)
    if r < 1e-5:
        return 0.0, 0.0
    a = -G * m_source / r**3
    return a * dx, a * dy

for i in range(steps - 1):
    # Midpoint velocities for Earth
    ax1_e, ay1_e = gravity_accel(x_earth[i], y_earth[i], M_sun, 0, 0)
    vx_mid_e = vx_earth + 0.5 * dt * ax1_e
    vy_mid_e = vy_earth + 0.5 * dt * ay1_e

    # Earth's position at midpoint
    x_mid_e = x_earth[i] + 0.5 * dt * vx_mid_e
    y_mid_e = y_earth[i] + 0.5 * dt * vy_mid_e

    ax2_e, ay2_e = gravity_accel(x_mid_e, y_mid_e, M_sun, 0, 0)
    # Update Earth's velocity using midpoint acceleration
    vx_earth += dt * ax2_e
    vy_earth += dt * ay2_e
    # Update Earth's position using new velocity
    x_earth[i + 1] = x_earth[i] + dt * vx_earth
    y_earth[i + 1] = y_earth[i] + dt * vy_earth

    # Midpoint velocities for Moon
    ax1_sun, ay1_sun = gravity_accel(x_moon[i], y_moon[i], M_sun, 0, 0)
    ax1_earth, ay1_earth = gravity_accel(x_moon[i], y_moon[i], M_earth, x_earth[i], y_earth[i])

    # Estimate Moon's velocity at midpoint
    vx_mid_m = vx_moon + 0.5 * dt * (ax1_sun + ax1_earth)
    vy_mid_m = vy_moon + 0.5 * dt * (ay1_sun + ay1_earth)

    # Estimate Moon's position at midpoint
    x_mid_m = x_moon[i] + 0.5 * dt * vx_mid_m
    y_mid_m = y_moon[i] + 0.5 * dt * vy_mid_m

    ax2_sun, ay2_sun = gravity_accel(x_mid_m, y_mid_m, M_sun, 0, 0)
    ax2_earth, ay2_earth = gravity_accel(x_mid_m, y_mid_m, M_earth, x_earth[i], y_earth[i])

    vx_moon += dt * (ax2_sun + ax2_earth)
    vy_moon += dt * (ay2_sun + ay2_earth)

    x_moon[i + 1] = x_moon[i] + dt * vx_moon
    y_moon[i + 1] = y_moon[i] + dt * vy_moon

plt.figure(figsize=(10, 10))
plt.plot(0, 0, 'yo', markersize=10, label='Sun')
plt.plot(x_earth, y_earth, 'b-', label='Earth Orbit')
plt.plot(x_moon, y_moon, 'r-', label='Moon Trajectory')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectory of the Moon Relative to the Sun')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()