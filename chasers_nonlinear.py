"""
This file simulates the dynamics for the nonlinear system where the acceleration of the sheep
follows an inverse square law. Due to the nonlinear dynamics of the system, we linearize the 
system around any given state and compute the control using this linearization.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm

# Global progress bar
t_f = 30
progress_bar = tqdm(total=t_f, desc="Solving ODE")

# Define a bunch of hyperparameters
# Number of sheep
N = 2
# Number of Dogs
M = 10
# This is introduced in to the denominators for numerical stability
eps = 0.1
# This is the exponent on our distance function in the denominator
# 0 corresponds to the linear case, I have found that 3 is often stable
# This corresponds to an inverse square law 
expo_r = 3
expo_p = 3
# This corresponds to how far away runners see pursuers
width = 3
# This is how fast our runner runs away from our pursuer
d=1

# Note that each state is ordered (x_pos, y_pos, x_vel, y_vel) and each of these are stacked
# in a single long array.
state_dim = 4*(M + N)
control_dim = 2*M

# Define all of our cost matrices. Note that only the A in LQR is nonlinear, so 
# we can define all of the explicitely
# The cost on our control
R = np.eye(control_dim)/10
# The cost on each individual state
Q = np.eye(state_dim)
for i in range(N):
    # Only penalize sheep distance from origin
    Q[4*i:4*i+4, 4*i:4*i+4] = np.diag([1000,1000, 0, 0])
for j in range(M):
    # Mostly penaliez dog velocity, penalizing position is more stable.
    idx = 4*N + 4*j
    Q[idx:idx+4, idx:idx+4] = np.diag([1,1, 200, 200])
Q /= 1000

# Control matrix (how our control affects the state)
B = np.zeros((state_dim, control_dim))
for j in range(M):
    # We control the acceleration of our dogs
    B[4*N + 4*j + 2, 2*j] = 1
    B[4*N + 4*j + 3, 2*j + 1] = 1

# The next section explicitely linearizes the system around a given state
# These derivatives will be used in the A matrix.
# Define symbolic variables
x0, y0, v0x, v0y = sp.symbols('rx ry rvx rvy')
x1, y1, v1x, v1y, e1, w = sp.symbols('c1x c1y c1vx c1vy e1 w')
d1 = (sp.sqrt((x1 - x0)**2 + (y1 - y0)**2)/w)**(e1) + eps
vx_i_n = (x0 - x1) / d1
vy_i_n = (y0 - y1) / d1
# Differentiate first with respect to the runner
dvx_drx = sp.diff(vx_i_n, x0)
f_dvx_drx = sp.lambdify((x0, y0, x1, y1, e1, w), dvx_drx, "numpy")
dvx_dry = sp.diff(vx_i_n, y0)
f_dvx_dry = sp.lambdify((x0, y0, x1, y1, e1, w), dvx_dry, "numpy")
dvy_drx = sp.diff(vy_i_n, x0)
f_dvy_drx = sp.lambdify((x0, y0, x1, y1, e1, w), dvy_drx, "numpy")
dvy_dry = sp.diff(vy_i_n, y0)
f_dvy_dry = sp.lambdify((x0, y0, x1, y1, e1, w), dvy_dry, "numpy")
# Next with respect to the pursuer
dvx_dpx = sp.diff(vx_i_n, x1)
f_dvx_dpx = sp.lambdify((x0, y0, x1, y1, e1, w), dvx_dpx, "numpy")
dvx_dpy = sp.diff(vx_i_n, y1)
f_dvx_dpy = sp.lambdify((x0, y0, x1, y1, e1, w), dvx_dpy, "numpy")
dvy_dpx = sp.diff(vy_i_n, x1)
f_dvy_dpx = sp.lambdify((x0, y0, x1, y1, e1, w), dvy_dpx, "numpy")
dvy_dpy = sp.diff(vy_i_n, y1)
f_dvy_dpy = sp.lambdify((x0, y0, x1, y1, e1, w), dvy_dpy, "numpy")

# This is computes the linearized A matrix at each step
# The idea is that the nonlinear equation is a sum of equations that all look like (a_x-b_x)/norm(a_x-b_x)^lambda
def linearize_dynamics(x):
    """This is computes the linearized A matrix at each step
        The idea is that the nonlinear equation is a sum of equations that all look like (a_x-b_x)/norm(a_x-b_x)^lambda
        so I just made it faster

    Args:
        x (np.array): state at specific time

    Returns:
        np.array: Linearized A matrix
    """
    # Compute the linearization of dynamics corresponding to A matrix at time t
    A_r = np.zeros((4*N,(N+M)*4))
    # First set up the derivatives with respect to velocity (of the sheep)
    for i in range(N):
        A_r[i*4,i*4+2] = 1
        A_r[i*4+1,i*4+3] = 1
    # Next we iterate through calculating the derivatives each with respect to individual states
    for i in range(N):
        for j in range(M):
            A_r[i*4+2,4*i] += f_dvx_drx(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            A_r[i*4+2,4*i+1] += f_dvx_dry(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            A_r[i*4+2,(N+j)*4] = f_dvx_dpx(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            A_r[i*4+2,(N+j)*4+1] = f_dvx_dpy(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            
            A_r[i*4+3,4*i] += f_dvy_drx(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            A_r[i*4+3,4*i+1] += f_dvy_dry(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            A_r[i*4+3,(N+j)*4] = f_dvy_dpx(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
            A_r[i*4+3,(N+j)*4+1] = f_dvy_dpy(x[4*i],x[4*i+1],x[4*(N+j)],x[4*(N+j)+1], expo_r, width)/d
    A_p = np.zeros((4*M,4*(M+N)))
    # these correspond to the acceleration moving the position (of the dogs)
    for j in range(M):
        A_p[j*4,(N+j)*4+2] = 1
        A_p[j*4+1,(N+j)*4+3] = 1
    A = np.block([[A_r],[A_p]])
    return A

def compute_control(x, Q, R):
    """This computes the control, How should we update each of the chasers

    Args:
        x (np.array): Entire State
        Q (np.array): Quadratic cost of x
        R (np.array): Quadratic cost of u

    Returns:
        _type_: _description_
    """
    A = linearize_dynamics(x)
    try:
        # Solve the ARE
        P = solve_continuous_are(A, B, Q, R)
        # Get the gain matrix
        K = np.linalg.inv(R) @ B.T @ P
        # Use that to get the control
        u = -K @ x
    except np.linalg.LinAlgError:
        print("Division by zero")
        u = np.zeros(2*M)  # Just in case we do divide by zero
    return u

def dynamics(t, x):
    """
    We explicitly calculate the dynamics of the nonlinear system
    """
    progress_bar.update(t - progress_bar.n)  # Update the progress bar
    dxdt = np.zeros_like(x)
    u = compute_control(x, Q, R)
    
    # Runners' dynamics
    for i in range(N):
        pos = x[4*i:4*i+2]
        vel = x[4*i+2:4*i+4]
        acc = np.zeros(2)
        
        for j in range(M):
            p_pos = x[4*N+4*j:4*N+4*j+2]
            dist = (np.linalg.norm(pos - p_pos)/width)**expo_r + eps
            acc += (pos - p_pos) / dist
        
        dxdt[4*i:4*i+2] = vel
        dxdt[4*i+2:4*i+4] = acc/d
    
    # Pursuers' dynamics
    for j in range(M):
        idx = 4*N + 4*j
        
        dxdt[idx:idx+2] = x[idx+2:idx+4]  # Position update
        dxdt[idx+2] = u[2*j] + acc[0]/d          # x acceleration
        dxdt[idx+3] = u[2*j+1] + acc[1]/d        # y acceleration
    
    return dxdt

# Initialize the states randomly
x_0 = np.random.normal(size=(state_dim))
# Simulate with solve_ivp
t_eval = np.linspace(0, t_f, 500) 
sol = solve_ivp(dynamics, [0,t_f], x_0, method='RK45', t_eval=t_eval, max_step=t_eval[1]-t_eval[0])

# Create the figure
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=f"Runner {i+1}", color="r")[0] for i in range(N)]
scatters = [ax.scatter([], [], s=50, c="r") for _ in range(N)]
lines += [ax.plot([], [], label=f"Pursuer {i+1-N}", color="g")[0] for i in range(N,N+M)]
scatters += [ax.scatter([], [], s=50, c="g") for _ in range(N,N+M)]

# Animation update function
plt.xlim(np.min(sol.y),np.max(sol.y))
plt.ylim(np.min(sol.y),np.max(sol.y))
def update(frame):
    for i in range(N+M):
        x_data = sol.y[i*4, :frame+1]
        y_data = sol.y[i*4+1, :frame+1]
        lines[i].set_data(x_data, y_data)
        scatters[i].set_offsets([x_data[-1], y_data[-1]])
    return lines + scatters

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(sol.t), blit=True)
progress_bar.close()
# Save the animation (optional)
ani.save("trajectory.mp4", writer="ffmpeg", fps=60)

plt.legend()
plt.show()
