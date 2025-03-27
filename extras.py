
x0, y0, v0x, v0y = sp.symbols('rx ry rvx rvy')
x1, y1, v1x, v1y = sp.symbols('c1x c1y c1vx c1vy')
x2, y2, v2x, v2y = sp.symbols('c2x c2y c2vx c2vy')
x3, y3, v3x, v3y = sp.symbols('c3x c3y c3vx c3vy')
d1 = sp.sqrt((x1 - x0)**2 + (y1 - y0)**2)**(expo) + eps
d2 = sp.sqrt((x2 - x0)**2 + (y2 - y0)**2)**(expo) + eps
d3 = sp.sqrt((x3 - x0)**2 + (y3 - y0)**2)**(expo) + eps
# We want to linearize this part of the nonlinear dynamics
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

