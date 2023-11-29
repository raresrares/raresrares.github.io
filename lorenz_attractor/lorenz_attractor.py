import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Lorenz system parameters
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

step = 50  # Increase step size to skip more frames and make the animation faster

# Lorenz system equations
def lorenz_system(current_state, t):
    x, y, z = current_state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial state and time points
initial_state = [1.0, 1.0, 1.0]
time_points = np.linspace(0, 50, 10000)

# Solve differential equations
solution = odeint(lorenz_system, initial_state, time_points)

# Extract solutions
x, y, z = solution.T

# Set up figure for animation
fig, ax = plt.subplots()
ax.axis('off')  # Turn off the axes

# Set the face and edge color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Set up the line, make it gold-colored and very thin
line, = ax.plot([], [], lw=0.5, color='#FFD700')  # Gold color in hex

# Set the axis limits if you want to control the scale, or you can remove this if you prefer auto-scaling
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(z), max(z))

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation function
def update(frame):
    line.set_data(x[:frame * step], z[:frame * step])
    return line,

# Number of frames reduced by the step size
num_frames = len(time_points) // step

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=10, blit=True)

# Save animation with increased fps for faster playback
ani.save('lorenz_attractor_xz_plane_faster.gif', writer='pillow', fps=120)

plt.close()