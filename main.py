import numpy as np
import matplotlib.pyplot as plt
import math

def der_ex01(t, x):
    ret = [0,0]
    ret[0] = x[1]
    ret[1] = -(10.0 + math.cos(t))*x[0]-(5.0 + math.cos(t))*x[1]

    return ret

def der_ex02(t, x):
    ret = [0,0]
    ret[0] = x[1]
    ret[1] = -(1.0)*x[0]-(2.0 + math.exp(t))*x[1]

    return ret

def der_ex03(t, x):
    ret = [0,0]
    ret[0] = x[1]
    ret[1] = -(1)*x[0]-(2.1 + math.cos(t))*x[1]

    return ret

def move_point(x, dx, ss):
    ret = list(x)
    for i in range(len(ret)):
        ret[i] += dx[i] * ss

    return ret

def doRungeKuttaStep(t, dt, x, fun_der):
    dx1 = fun_der(t, x)
    x2 = move_point(x, dx1, dt / 2.0)
    dx2 = fun_der(t + dt / 2.0, x2)
    x3 = move_point(x, dx2, dt / 2.0)
    dx3 = fun_der(t + dt / 2.0, x3)
    x4 = move_point(x, dx3, dt)
    dx4 = fun_der(t + dt, x4)

    ret = list(x)
    for i in range(len(x)):
        ret[i] += dt * (dx1[i] + 2.0 * dx2[i] + 2.0 * dx3[i] + dx4[i]) / 6.0

    return ret

for function in [der_ex01, der_ex02, der_ex03]:
    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection='3d')

    N = 10000
    dt = 8.0 / N
    M = 20
    dX = 1.0
    for i in range(M):
        for j in range(M):
            t = [0]
            x = [[-dX + i * 2.0 * dX / (M-1.0), -dX + j * 2.0 * dX / (M-1.0)]]

            for k in range(N):
                nx = doRungeKuttaStep(t[-1], dt, x[-1], function)

                t.append(dt * (k+1.0))
                x.append(nx)

            x = np.array(x)
            ax.plot3D(t, x[:,0], x[:,1], 'gray', alpha = 0.2)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticks([0, 2, 4, 6, 8])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(12)
    ax.set_xlabel(r"$t$", fontsize = 18)
    ax.set_ylabel(r"$x$", fontsize = 18)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$\dot{x}$", rotation=0, fontsize = 18)
    ax.tick_params(axis='x', which='major', pad=0)
    ax.tick_params(axis='y', which='major', pad=0)
    ax.tick_params(axis='z', which='major', pad=6)
    plt.savefig(str(function.__name__) + '.png', bbox_inches='tight')

    plt.show()
