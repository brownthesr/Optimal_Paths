import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import sympy as sp

num_runners = 1
num_chasers = 1
total_agents = num_runners + num_chasers

Q = np.eye(total_agents*4)
for i in range(0,4*num_runners, 4):
    Q[i+0,i+0] = 1
    Q[i+1,i+1] = 1
    Q[i+2,i+2] = 1000
    Q[i+3,i+3] = 1000
for i in range(4*num_runners, 4*num_runners+4*num_chasers, 4):
    Q[0+i,0+i] = 1
    Q[1+i,1+i] = 1
    Q[2+i,2+i] = 200
    Q[3+i,3+i] = 200
Q /= 1000
R = np.eye(num_chasers*2)*3
d=100
# Create our A
A_a = np.zeros((1,4*total_agents))
for i in range(num_runners):
    b = np.zeros((4, 4*total_agents))
    b[0,4*i+2] = 1
    b[1,4*i+3] = 1
    b[2,4*i] = num_chasers/d
    b[3,4*i+1] = num_chasers/d
    for j in range(num_runners, total_agents):
        b[2,4*j] = -1/d
        b[3,4*j+1] = -1/d
    A_a = np.concatenate([A_a, b], axis=0)
A_a = A_a[1:]
A_b = np.zeros((1,4*total_agents))
for j in range(num_runners, total_agents):
    b = np.zeros((4,4*total_agents))
    b[0,4*j+2] = 1
    b[1,4*j+3] = 1
    A_b = np.concatenate([A_b,b], axis=0)
A_b = A_b[1:]
A = np.concatenate([A_a, A_b], axis=0)
print(A)
# Create our B
B = np.zeros((4*total_agents, 2*num_chasers))
for j in range(num_runners, total_agents):
    B[4*j+2, 2*(j-num_runners)] = 1
    B[4*j+3, 2*(j-num_runners)+1] = 1
print(B)
# B = np.array([
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [1, 0, 0, 0, 0, 0],
# [0, 1, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 1, 0, 0, 0],
# [0, 0, 0, 1, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 1, 0],
# [0, 0, 0, 0, 0, 1]
# ])
print(A.shape)
P = solve_continuous_are(A,B,Q,R)
def dynamics(t, x):
    return (A-B@np.linalg.inv(R)@B.T@P)@x

x0 = np.random.normal(size=(total_agents*4))
x0[:1] -= 3
# x0 = np.array([
#     1,1,0,0,
#     1,-5,0,0,
#     1.5,-5.1,0,0,
#     1.25,-4,0,0
# ])
# Simulate with solve_ivp
sol = solve_ivp(dynamics, [0, 100], x0, method='RK45')
# Create the figure
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)  # Adjust based on expected trajectory range
ax.set_ylim(-10, 10)
lines = [ax.plot([], [], label=f"Runner {i+1}", color="r")[0] for i in range(num_runners)]
scatters = [ax.scatter([], [], s=50, c="r") for _ in range(num_runners)]
lines += [ax.plot([], [], label=f"Pursuer {i+1}", color="g")[0] for i in range(num_runners,num_chasers+num_runners)]
scatters += [ax.scatter([], [], s=50, c="g") for _ in range(num_runners,total_agents)]

# Animation update function
plt.xlim(np.min(sol.y),np.max(sol.y))
plt.ylim(np.min(sol.y),np.max(sol.y))
def update(frame):
    for i in range(total_agents):
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