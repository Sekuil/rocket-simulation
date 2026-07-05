import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_e = 5.972168e24  # Mass of Earth (kg)
R_e = 6371000  # Radius of Earth (m), Global average of the radius

# Rocket parameters, based on Starship, two-stage super heavy-lift launch vehicle with no payload.
drag_coefficient = 0.4  # Drag coefficient, approximated since it is usually determined experimentally
# The drag coefficient depends on the velocity of the rocket, and fluctuates a lot around mach 1.
area = np.pi * (9/2)**2  # Cross-sectional area of the rocket (m^2), diameter 9 m
empty_mass1 = 275000  # Rocket mass of first stage without fuel (kg)
empty_mass2 = 100000  # Rocket mass of second stage without fuel (kg)
thrust1 = 7590 * 1000 * 9.80665
thrust2 = 1500 * 1000 * 9.80665
fuel_rate1 = 20481.92771  # Mass flow rate (kg/s)
fuel_rate2 = 3723.986857  # (kg/s)
burn_time1 = 166  # (s)
burn_time2 = 402.8  # (s)
# thrust1 = 7607000  # Thrust of stage 1 engine (N) at sea level, at vacuum 8227000
# thrust2 = 981000  # Thrust of stage 2 engine (N) in vacuum
# Thrust equation F = dm/dt * V_e * A_e*(p_e-p_0)

# Atmospheric parameters
rho_0 = 1.225  # Sea level air density (kg/m^3)
h_scale = 8500  # Scale height of the atmosphere (m)

# Escape velocity and low earth orbit velocity
v_escape = np.sqrt(2 * G * M_e / R_e)  # 11,186 km/s
v_leo = 7800  # Low earth orbit: 7,8 km/s or 28,000 km/h

# Time step
dt = 0.01  # Integration time step (s)


def escape_velocity(altitude):
    return np.sqrt(2*G*M_e / (R_e + altitude))


def gravity(mass, altitude):
    r = R_e + altitude
    return G * M_e * mass / r**2


def drag(v_r, v_theta, altitude):
    rho = rho_0 * np.exp(-altitude / h_scale)
    return 0.5 * drag_coefficient * rho * area * (v_r**2 + v_theta**2) * np.sign(v_r)


def thrust(fuel_rate, t=None, ramp_time=10):
    if t is not None and t < ramp_time:
        return thrust1 * (t/ramp_time)  # Linearly increase thrust over 'ramp_time'
    elif fuel_rate == fuel_rate1:
        return thrust1
    elif fuel_rate == fuel_rate2:
        return thrust2
    else:
        return 0


def rk4_step(func, y, t, dt, *args):
    k1 = func(t, y, *args)
    k2 = func(t + dt / 2, y + dt * k1 / 2, *args)
    k3 = func(t + dt / 2, y + dt * k2 / 2, *args)
    k4 = func(t + dt, y + dt * k3, *args)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def equations(t, state, stage, fuel_rate, stage_mass):
    angle, altitude, v_r, v_theta, mass = state
    if mass <= stage_mass:  # No fuel left
        fuel_rate = 0
    w = v_theta / (R_e + altitude)

    # Forces acting on the rocket
    thrust_force = thrust(fuel_rate, t)
    g_force = gravity(mass, altitude)
    drag_force = drag(v_r, v_theta, altitude)

    # Acceleration components
    if v_r**2 + v_theta**2 == 0 or altitude < 10000:
        a_r = (thrust_force - g_force - drag_force) / mass
        a_theta = 0
    elif altitude >= 10000 and v_theta < 0.2*v_r:
        a_r = (thrust_force * (v_r / np.sqrt(v_r ** 2 + v_theta ** 2)) - g_force - drag_force * (
                v_r / np.sqrt(v_r ** 2 + v_theta ** 2))) / mass
        a_theta = 1 * a_r
        #print(a_theta, v_theta)
    else:
        a_r = (thrust_force * (v_r / np.sqrt(v_r**2 + v_theta**2)) - g_force - drag_force * (
                v_r / np.sqrt(v_r**2 + v_theta**2))) / mass
        a_theta = (thrust_force * (v_theta / np.sqrt(v_r**2 + v_theta**2)) - drag_force * (
                v_theta / np.sqrt(v_r**2 + v_theta**2))) / mass
        #print(a_theta, v_theta)
    """
    cruise = 1
    if cruise == 1 and mass <= empty_mass2:
        a_r = - g_force/mass + (v_theta**2/(R_e + altitude))
        # a_r = (v_theta**2/(R_e + altitude)) - G * M_e/(R_e + altitude)**2
        a_theta = 0
        print(angle, altitude, v_r)"""

    return np.array([w, v_r, a_r, a_theta, -fuel_rate])


def simulate_rocket(fuel_stage1, fuel_stage2):
    # Initialize mass
    mass_stage1 = empty_mass1 + fuel_stage1 + empty_mass2 + fuel_stage2
    mass_stage2 = empty_mass2 + fuel_stage2

    # Initial conditions
    state = np.array([0, 0, 0, 0, mass_stage1])  # angle, altitude, v_r, v_theta, mass
    t = 0
    trajectory = []
    notif_leo = 0

    # Stage 1
    while state[4] - empty_mass1 > mass_stage2:  # Burn until stage 1 fuel is exhausted
        trajectory.append((t, *state))
        state = rk4_step(equations, state, t, dt, 2, fuel_rate1, mass_stage2)
        t += dt

    # MECO and stage separation delay
    state[4] = mass_stage2  # Update mass after stage separation
    for _ in range(int(10/dt)):  # Simulate 10-second delay
        trajectory.append((t, *state))
        state = rk4_step(equations, state, t, dt, 2, 0, empty_mass2)
        t += dt

    # Stage 2
    while state[4] > empty_mass2:
        if state[3] >= v_leo and notif_leo == 0:  # Enter low earth orbit (LEO)
            state_leo = (t, *state)
            notif_leo = 1
        trajectory.append((t, *state))
        state = rk4_step(equations, state, t, dt, 2, fuel_rate2, empty_mass2)
        t += dt

    # Cruise
    cruise = 0
    while state[0] < np.pi and cruise == 1:
        trajectory.append((t, *state))
        state = rk4_step(equations, state, t, dt, 2, 0, empty_mass2)
        t += dt

    # Check if rocket achieved the escape velocity for any state
    states = np.array(trajectory)
    #print(state[2], state[3], state[4])

    if state[3] >= escape_velocity(state[1]):
        return states, True
    else:
        return states, False

#    for j in range(len(states[:, 0])):
#       if states[j, 4] >= escape_velocity(states[j, 2]):
#           return states, True
#
#   return states, False


# Plot of trajectory with earth in center
sim1, holder = simulate_rocket(3400000, 1500000)
plt.figure(figsize=(8, 8))
earth = plt.Circle((0, 0), R_e, color='blue', alpha=0.5, label='Earth')
plt.gca().add_artist(earth)
leo = plt.Circle((0, 0), R_e + 2000000, color='green', linestyle='dashed', fill=False, label='Low Earth Orbit')
plt.gca().add_artist(leo)
vleo = plt.Circle((0, 0), R_e + 400000, color='red', linestyle='dashed', fill=False, label='Very Low Earth Orbit')
plt.gca().add_artist(vleo)
plt.plot((sim1[:, 2] + R_e) * np.sin(sim1[:, 1]), (sim1[:, 2] + R_e) * np.cos(sim1[:, 1]),
         color='orange', label='Rocket Trajectory')
plt.axis('equal')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend()
plt.grid()
#plt.show()

# Plot drag vs. altitude
drag_lst = []
for i in range(len(sim1[:, 0])):
    drag_lst.append(drag(sim1[i, 3], sim1[i, 4], sim1[i, 2]))
plt.figure(figsize=(10, 6))
plt.plot(sim1[:, 2], drag_lst)
plt.xlabel("Altitude (m)")
plt.ylabel("Drag (N)")
plt.grid()
#plt.show()

# Plot Altitude vs. time
plt.figure(figsize=(10, 6))
plt.plot(sim1[:, 0], sim1[:, 2])
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid()
#plt.show()

# Plot velocity vs. time
plt.figure(figsize=(10, 6))
plt.plot(sim1[:, 0], sim1[:, 3], label='Radial velocity')
plt.plot(sim1[:, 0], sim1[:, 4], label='Transverse velocity')
plt.plot(sim1[:, 0], np.sqrt(sim1[:, 3]**2 + sim1[:, 4]**2), label='Total velocity')
#v_escape_lst = np.linspace(v_escape, v_escape, len(sim1[:, 0]))
plt.plot(sim1[:, 0], escape_velocity(sim1[:, 2]), 'r--', label='Escape Velocity')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid()
plt.show()

"""---------------- FUEL ANALYSIS----------------"""
# Analyze different fuel combinations 3400000, 1500000
fuel_values1 = np.linspace(2920000, 3100000, 12)
fuel_values1 = fuel_values1.astype(int)
fuel_values2 = np.linspace(1440000, 1500000, 12)
fuel_values2 = fuel_values2.astype(int)
escape_results = []

for fuel1 in fuel_values1:
    for fuel2 in fuel_values2:
        print(fuel1)
        sim, escaped = simulate_rocket(fuel1, fuel2)
        escape_results.append((fuel1, fuel2, escaped))
# print(escape_results)
# Plot fuel combinations
fuel1_vals, fuel2_vals, results = zip(*escape_results)
plt.figure(figsize=(10, 6))
for f1, f2, res in escape_results:
    plt.scatter(f1, f2, c='green' if res else 'red', marker='o' if res else 'x',
                label='Successful' if res else 'Unsuccessful')
plt.xlabel("Stage 1 Fuel (kg)")
plt.ylabel("Stage 2 Fuel (kg)")
plt.grid()
plt.show()
