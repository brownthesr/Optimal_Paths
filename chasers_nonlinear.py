import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

import matplotlib.pyplot as plt

import sympy as sp

# Define symbolic variables
eps = .001
expo = 1
x0, y0, v0x, v0y = sp.symbols('rx ry rvx rvy')
x1, y1, v1x, v1y = sp.symbols('c1x c1y c1vx c1vy')
x2, y2, v2x, v2y = sp.symbols('c2x c2y c2vx c2vy')
x3, y3, v3x, v3y = sp.symbols('c3x c3y c3vx c3vy')
d1 = sp.sqrt((x1 - x0)**2 + (y1 - y0)**2)**(expo) + eps
d2 = sp.sqrt((x2 - x0)**2 + (y2 - y0)**2)**(expo) + eps
d3 = sp.sqrt((x3 - x0)**2 + (y3 - y0)**2)**(expo) + eps
vx_i = (x0 - x1) / d1 + (x0 - x2) / d2 + (x0 - x3) / d3
vy_i = (y0 - y1) / d1 + (y0 - y2) / d2 + (y0 - y3) / d3

# Compute derivatives symbolically
# First component
dvx_dx0_ = sp.diff(vx_i, x0)
f_dvx_dx0 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dx0_, "numpy")
dvx_dy0_ = sp.diff(vx_i, y0)
f_dvx_dy0 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dy0_, "numpy")
dvx_dx1_ = sp.diff(vx_i, x1)
f_dvx_dx1 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dx1_, "numpy")
dvx_dy1_ = sp.diff(vx_i, y1)
f_dvx_dy1 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dy1_, "numpy")
dvx_dx2_ = sp.diff(vx_i, x2)
f_dvx_dx2 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dx2_, "numpy")
dvx_dy2_ = sp.diff(vx_i, y2)
f_dvx_dy2 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dy2_, "numpy")
dvx_dx3_ = sp.diff(vx_i, x3)
f_dvx_dx3 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvx_dx3_, "numpy")
dvx_dy3_ = sp.diff(vx_i, y3)
f_dvx_dy3 = sp.lambdify( (x0, y0, x1, y1, x2, y2, x3, y3), dvx_dy3_, "numpy")

# Second component
dvy_dx0_ = sp.diff(vy_i, x0)
f_dvy_dx0 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dx0_, "numpy")
dvy_dy0_ = sp.diff(vy_i, y0)
f_dvy_dy0 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dy0_, "numpy")
dvy_dx1_ = sp.diff(vy_i, x1)
f_dvy_dx1 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dx1_, "numpy")
dvy_dy1_ = sp.diff(vy_i, y1)
f_dvy_dy1 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dy1_, "numpy")
dvy_dx2_ = sp.diff(vy_i, x2)
f_dvy_dx2 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dx2_, "numpy")
dvy_dy2_ = sp.diff(vy_i, y2)
f_dvy_dy2 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dy2_, "numpy")
dvy_dx3_ = sp.diff(vy_i, x3)
f_dvy_dx3 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dx3_, "numpy")
dvy_dy3_ = sp.diff(vy_i, y3)
f_dvy_dy3 = sp.lambdify((x0, y0, x1, y1, x2, y2, x3, y3), dvy_dy3_, "numpy")

# Create the Q matrix
Q = np.eye(16)
# This encourages our herders to try and move the runner toward the origin
Q[0,0] = 1
Q[1,1] = 1
# This encourages our herders to make our runner stop movving
Q[2,2] = 1000
Q[3,3] = 1000
# EAch of these encourage our herders to move toward the origin and or stop moving
Q[0+4,0+4] = 1
Q[1+4,1+4] = 1
Q[2+4,2+4] = 1000
Q[3+4,3+4] = 1000
Q[0+8,0+8] = 1
Q[1+8,1+8] = 1
Q[2+8,2+8] = 1000
Q[3+8,3+8] = 1000
Q[0+12,0+12] =1
Q[1+12,1+12] =1
Q[2+12,2+12] = 1000
Q[3+12,3+12] = 1000
Q /= 1000
R = np.eye(6)*3
def dvx_dx0(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dx0(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dy0(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dy0(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dx0(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dx0(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dy0(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dy0(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dx1(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dx1(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dy1(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dy1(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dx2(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dx2(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dy2(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dy2(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dx3(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dx3(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvx_dy3(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvx_dy3(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dx1(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dx1(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dy1(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dy1(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dx2(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dx2(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dy2(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dy2(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dx3(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dx3(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)
def dvy_dy3(x):
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    return f_dvy_dy3(rx,ry,c1x,c1y,c2x,c2y,c3x,c3y)


d=1
def linearize_dynamics(x):
    # Compute Jacobians A(t) and B(t) at current state x
    # (Your specific implementation here)
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    A_a = np.zeros((2,16))
    A_a[0,2] = 1
    A_a[1,3] = 1
    A_b = -np.array([
        [dvx_dx0(x), dvx_dy0(x),0,0,
         dvx_dx1(x), dvx_dy1(x),0,0,
         dvx_dx2(x), dvx_dy2(x),0,0,
          dvx_dx3(x), dvx_dy3(x),0,0],
        [dvy_dx0(x), dvy_dy0(x),0,0,
         dvy_dx1(x), dvy_dy1(x),0,0,
         dvy_dx2(x), dvy_dy2(x),0,0,
          dvy_dx3(x), dvy_dy3(x),0,0]
    ])*d
    A_c = np.zeros((12,16))
    A_c[0,6] = 1
    A_c[1,7] = 1
    A_c[4,10] = 1
    A_c[5,11] = 1
    A_c[8,14] = 1
    A_c[9,15] = 1
    A = np.block([[A_a],[A_b],[A_c]])
    # plt.show()
    B = np.zeros((16, 6))
    # Pursuer 1 acceleration affects its velocity derivatives (rows 6-7)
    B[6, 0] = 1  # dv1x/dt = a1x
    B[7, 1] = 1  # dv1y/dt = a1y
    # Pursuer 2 acceleration (rows 10-11)
    B[10, 2] = 1  # dv2x/dt = a2x
    B[11, 3] = 1  # dv2y/dt = a2y
    # Pursuer 3 acceleration (rows 14-15)
    B[14, 4] = 1  # dv3x/dt = a3x
    B[15, 5] = 1  # dv3y/dt = a3y
    return A, B

def compute_control(x, Q, R):
    A, B = linearize_dynamics( x)
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        u = -K @ x
    except np.linalg.LinAlgError:
        print("here")
        u = np.zeros(6)  # Fallback (e.g., zero control)
    return u

def dynamics(t, x):
    u1x, u1y, u2x, u2y, u3x, u3y = compute_control(x, Q, R)
    rx, ry, rvx, rvy, \
        c1x, c1y, c1vx, c1vy, \
        c2x, c2y, c2vx, c2vy, \
        c3x, c3y, c3vx, c3vy = x
    r0 = np.array([rx,ry])
    p1 = np.array([c1x,c1y])
    p2 = np.array([c2x,c2y])
    p3 = np.array([c3x,c3y])
    rv = ((r0-p1)/(np.linalg.norm(r0-p1)**(expo)+eps)+(r0-p2)/(np.linalg.norm(r0-p2)**(expo)+eps)+(r0-p3)/(np.linalg.norm(r0-p3)**(expo)+eps))/d
    return np.array([rvx,rvy,rv[0],rv[1],
        c1vx,c1vy, u1x,u1y,
        c2vx,c2vy, u2x,u2y,
        c3vx,c3vy, u3x,u3y
    ])

x_0 = np.array([
    1,1,0,0,
    1,-5,0,0,
    1.5,-5.1,0,0,
    -1.25,4,0,0
])
# x_0 = np.random.normal(size=(16))
# Simulate with solve_ivp
sol = solve_ivp(dynamics, [0,20], x_0, method='RK45')
# Create the figure
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)  # Adjust based on expected trajectory range
ax.set_ylim(-10, 10)
lines = [ax.plot([], [], label=f"Runner", color="r")[0] for i in range(1)]
scatters = [ax.scatter([], [], s=50, c="r") for _ in range(1)]
lines += [ax.plot([], [], label=f"Pursuer {i+1}", color="g")[0] for i in range(1,4)]
scatters += [ax.scatter([], [], s=50, c="g") for _ in range(1,4)]

# Animation update function
plt.xlim(np.min(sol.y),np.max(sol.y))
plt.ylim(np.min(sol.y),np.max(sol.y))
def update(frame):
    for i in range(4):
        x_data = sol.y[i*4, :frame+1]
        y_data = sol.y[i*4+1, :frame+1]
        lines[i].set_data(x_data, y_data)
        scatters[i].set_offsets([x_data[-1], y_data[-1]])
    return lines + scatters

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(sol.t), blit=True)

# Save the animation (optional)
ani.save("trajectory.mp4", writer="ffmpeg", fps=30)

plt.legend()
plt.show()