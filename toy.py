from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy.optimize import minimize, Bounds, minimize_scalar

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import seaborn as sns
import sys

################################################################################
#
# Define the Optimization Problem
#
################################################################################
LOWER = 0.000005

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]])

    def forward(self, x, compute_grad=False):
        x1 = x[0]
        x2 = x[1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f = torch.tensor([f1, f2])
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            return f, g
        else:
            return f

    def batch_forward(self, x):
        x1 = x[:,0]
        x2 = x[:,1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f  = torch.cat([f1.view(-1, 1), f2.view(-1,1)], -1)
        return f

################################################################################
#
# Plot Utils
#
################################################################################

def plotme(F, all_traj=None, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    colormaps = {
        "sgd": "tab:blue",
        "pcgrad": "tab:orange",
        "mgd": "tab:cyan",
        "cagrad": "tab:red",
    }

    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    c = plt.contour(X, Y, Ys[:,0].view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.title("L1(x)")

    plt.subplot(132)
    c = plt.contour(X, Y, Ys[:,1].view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.title("L2(x)")

    plt.subplot(133)
    c = plt.contour(X, Y, Ys.mean(1).view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.legend()
    plt.title("0.5*(L1(x)+L2(x))")

    plt.tight_layout()
    plt.savefig(f"toy_ct.png")

def plot3d(F, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.viridis)
    print(Ys.mean(1).min(), Ys.mean(1).max())

    ax.set_zticks([-16, -8, 0, 8])
    ax.set_zlim(-20, 10)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.view_init(25)
    plt.tight_layout()
    plt.savefig(f"3d-obj.png", dpi=1000)

def plot_contour(F, task=1, traj=None, xl=11, plotbar=False, name="tmp"): 
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    cmap = cm.get_cmap('viridis')

    yy = -8.3552
    if task == 0:
        Yv = Ys.mean(1)
        plt.plot(-8.5, 7.5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot(-8.5, -5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot( 9, 9, marker='o', markersize=10, zorder=5, color='k')
        plt.plot([-7, 7], [yy, yy], linewidth=8.0, zorder=0, color='gray')
        plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k')
    elif task == 1:
        Yv = Ys[:,0]
        plt.plot(7, yy, marker='*', markersize=15, zorder=5, color='k')
    else:
        Yv = Ys[:,1]
        plt.plot(-7, yy, marker='*', markersize=15, zorder=5, color='k')

    c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.viridis, linewidths=4.0)

    if traj is not None:
        for tt in traj:
            l = tt.shape[0]
            color_list = np.zeros((l,3))
            color_list[:,0] = 1.
            color_list[:,1] = np.linspace(0, 1, l)
            #color_list[:,2] = 1-np.linspace(0, 1, l)
            ax.scatter(tt[:,0], tt[:,1], color=color_list, s=6, zorder=10)

    if plotbar:
        cbar = fig.colorbar(c, ticks=[-15, -10, -5, 0, 5])
        cbar.ax.tick_params(labelsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()

def smooth(x, n=20):
    l = len(x)
    y = []
    for i in range(l):
        ii = max(0, i-n)
        jj = min(i+n, l-1)
        v = np.array(x[ii:jj]).astype(np.float64)
        if i < 3:
            y.append(x[i])
        else:
            y.append(v.mean())
    return y

def plot_loss(trajs, name="tmp"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormaps = {
        "sgd": "tab:blue",
        "pcgrad": "tab:orange",
        "mgd": "tab:purple",
        "cagrad": "tab:red",
    }
    maps = {
        "sgd" : "Adam",
        "pcgrad" : "PCGrad",
        "mgd" : "MGDA",
        "cagrad" : "RGD (ours)",
    }
    for method in ["sgd", "mgd", "pcgrad", "cagrad"]:
        traj = trajs[method][::100]
        Ys = F.batch_forward(traj)
        x = np.arange(traj.shape[0])
        #y = torch.cummin(Ys.mean(1), 0)[0]
        y = Ys.mean(1)

        ax.plot(x, smooth(list(y)),
                color=colormaps[method],
                linestyle='-',
                label=maps[method], linewidth=4.)

    plt.xticks([0, 200, 400, 600, 800, 1000],
               ["0", "20K", "40K", "60K", "80K", "100K"],
               fontsize=15)

    plt.yticks(fontsize=15)
    ax.grid()
    plt.legend(fontsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()

################################################################################
#
# Multi-Objective Optimization Solver
#
################################################################################

def mean_grad(grads):
    return grads.mean(1)

def pcgrad(grads):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        return ((1-g12/g11)*g1+(1-g12/g22)*g2)/2
    else:
        return (g1+g2)/2

def mgd(grads):
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    if g12 < min(g11, g22):
        x = (g22-g12) / (g11+g22-2*g12 + 1e-8)
    elif g11 < g22:
        x = 1
    else:
        x = 0

    g_mgd = x * g1 + (1-x) * g2 # mgd gradient g_mgd
    return g_mgd

def cagrad(grads, c=0.5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)


### Define the problem ###
F = Toy()

maps = {
    "sgd": mean_grad,
    "cagrad": cagrad,
    "mgd": mgd,
    "pcgrad": pcgrad,
}

### Start experiments ###

def run_all():
    all_traj = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([-8.5, -5.]),
        torch.Tensor([9.,   9.]),
    ]

    for i, init in enumerate(inits):
        for m in tqdm(["sgd", "mgd", "pcgrad", "cagrad"]):
            all_traj[m] = None
            traj = []
            solver = maps[m]
            x = init.clone()
            x.requires_grad = True

            n_iter = 100000
            opt = torch.optim.Adam([x], lr=0.001)

            for it in range(n_iter):
                traj.append(x.detach().numpy().copy())

                f, grads = F(x, True)
                if m== "cagrad":
                    g = solver(grads, c=0.5)
                else:
                    g = solver(grads)
                opt.zero_grad()
                x.grad = g
                opt.step()

            all_traj[m] = torch.tensor(traj)
        torch.save(all_traj, f"toy{i}.pt")


def plot_results():
    plot3d(F)
    plot_contour(F, 1, name="toy_task_1")
    plot_contour(F, 2, name="toy_task_2")
    t1 = torch.load(f"toy0.pt")
    t2 = torch.load(f"toy1.pt")
    t3 = torch.load(f"toy2.pt")

    length = t1["sgd"].shape[0]

    for method in ["sgd", "mgd", "pcgrad", "cagrad"]:
        ranges = list(range(10, length, 1000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0, # task == 0 meeas plot for both tasks
                         traj=[t1[method][:t],t2[method][:t],t3[method][:t]],
                         plotbar=(method == "cagrad"),
                         name=f"./imgs/toy_{method}_{t}")


if __name__ == "__main__":
    run_all()
