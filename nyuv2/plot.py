import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

methods = [
        "cross-equal",
        "mtan-equal",
        "mgd-equal",
        "pcgrad-equal",
        "graddrop-equal",
        "cagrad-equal-4e-1",
        ]

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "tab:green", "tab:cyan", "tab:blue", "tab:red"]
stats = ["semantic loss", "mean iou", "pix acc", "depth loss",
         "abs err", "rel err", "normal loss", "mean",
         "median", "<11.25", "<22.5", "<30"]

delta_stats = ["mean iou", "pix acc", "abs err", "rel err", "mean", "median", "<11.25", "<22.5", "<30"]

stats_idx_map = [4,5,6,8,9,10,12,13,14,15,16,17]

time_idx = 34

seeds = [0,1,2]

logs = {}
min_epoch = 100000


for m in methods:
    logs[m] = {"train":[None for _ in range(3)], "test":[None for _ in range(3)]}

    for seed in seeds:
        logs[m]["train"][seed] = {}
        logs[m]["test"][seed] = {}

    for stat in stats:
        for seed in seeds:
            logs[m]["train"][seed][stat] = []
            logs[m]["test"][seed][stat] = []

    for seed in seeds:
        logs[m]["train"][seed]["time"] = []

    for seed in seeds:
        fname = f"logs/{m}-sd{seed}.log"
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Epoch"):
                    ws = line.split(" ")
                    for i, stat in enumerate(stats):
                        logs[m]["train"][seed][stat].append(float(ws[stats_idx_map[i]]))
                        logs[m]["test"][seed][stat].append(float(ws[stats_idx_map[i]+15]))
                    logs[m]["train"][seed]["time"].append(float(ws[time_idx]))
            min_epoch = min(min(min_epoch, len(logs[m]["train"][seed]["semantic loss"])), len(logs[m]["test"][seed]["semantic loss"]))

test_stats = {}
train_stats = {}
learning_time = {}

print(" "*25 + " | ".join([f"{s:5s}" for s in stats]))

for mi, mode in enumerate(["train", "test"]):
    if mi == 1:
        print(mode)
    for mmi, m in enumerate(methods):
        if m not in test_stats:
            test_stats[m] = {}
            train_stats[m] = {}

        string = f"{m:30s} "
        for stat in stats:
            x = []
            for seed in seeds:
                x.append(np.array(logs[m][mode][seed][stat][min_epoch-10:min_epoch]).mean())
            x = np.array(x)
            if mode == "test":
                test_stats[m][stat] = x.copy()
            else:
                train_stats[m][stat] = x.copy()
            mu = x.mean()
            std = x.std() / np.sqrt(3)
            string += f" | {mu:5.4f}"
        if mode == "test":
            print(string)

for m in methods:
    learning_time[m] = np.array([np.array(logs[m]["train"][sd]["time"]).mean() for sd in seeds])


### print delta M

base = np.array([0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915])
sign =np.array([1,1,0,0,0,0,1,1,1])
kk = np.ones(9) * -1

def delta_fn(a):
    return (kk**sign * (a - base) / base).mean() * 100. # *100 for percentage

deltas = {}
for method in methods:
    tmp = np.zeros(9)
    for i,stat in enumerate(delta_stats):
        tmp[i] = test_stats[method][stat].mean()
    deltas[method] = delta_fn(tmp)
    print(f"{method:30s} delta: {deltas[method]:4.3f}")



target_method = "cagrad-equal-4e-1"
other_methods = [
        "cross-equal",
        "mtan-equal",
        "mgd-equal",
        "pcgrad-equal",
        "pcgrad-equal-2e-4",
        "graddrop-equal",
]

from scipy import stats as st
print(" "*10 + " ".join(other_methods))
for stat in delta_stats:
    string = f"{stat:10s}"
    for method in other_methods:
        t,p = st.ttest_ind(test_stats[method][stat], test_stats[target_method][stat], equal_var = False)
        string += f" {p:5.4f}"
    print(string)


### plot test losses + learning time

def barplot(ys, methods, name, loss, title):
    # ys [#methods, 3], 3 means 3 seeds

    sns.set(style="whitegrid")
    plt.figure(figsize=(2,4))

    n = ys.shape[0]
    bw = 0.1
    x = np.arange(n) * (bw) + 1 # x_pos

    yy = ys.mean(1)
    y_min = yy.min()
    y_max = yy.max()

    if "depth" in loss:
        y_lim_min = y_min - (y_max-y_min) * 1.5
        y_lim_max = y_max + (y_max-y_min) * 1.5
    elif "time" in loss:
        y_lim_min = 0
        y_lim_max = y_max * 1.2
    else:
        y_lim_min = y_min - (y_max-y_min)
        y_lim_max = y_max + (y_max-y_min)/2
     
    # Create barplot
    for a, b, c, cl in zip(x, ys, methods, colors[:n]):
        plt.bar(a, b.mean(), yerr=b.std()/np.sqrt(n), 
                width=bw, color=cl, edgecolor='black',
                capsize=5, alpha=1., linewidth=1.,
                error_kw={"elinewidth":1, "capsize":3})
    plt.box(False)
    plt.title(title, fontsize=15)
    plt.ylim(y_lim_min, y_lim_max)
    plt.yticks(fontsize=10)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=400)
    plt.close()

plot_items = ["semantic loss", "depth loss", "normal loss", "time"]
names = ["Semantic", "Depth", "Normal", "Time"]

plot_methods = [
        "mtan-equal",
        "pcgrad-equal",
        "graddrop-equal",
        "cagrad-equal-4e-1",
]

Y = []
for i, item in enumerate(plot_items):
    if item == "time":
        ys = np.stack([learning_time[mm] for mm in plot_methods])
    else:
        ys = np.stack([test_stats[mm][item] for mm in plot_methods])
        Y.append(ys)
    barplot(ys, plot_methods, f"imgs/nyuv2_{item}", item, names[i])

Y = np.stack(Y).mean(0)
barplot(Y, plot_methods, f"imgs/nyuv2_average_loss", "average loss", "Average")

### plot the legend

matplotlib.rc_file_defaults()

methods = ["MTAN", "PCGrad", "GradDrop", "CAGrad (ours)"]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(len(methods))]
labels = methods
legend = plt.legend(handles, labels, framealpha=0.0, frameon=True, ncol=len(methods))
plt.axis('off')


def export_legend(legend, filename="./imgs/legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=400, bbox_inches=bbox)

export_legend(legend)
