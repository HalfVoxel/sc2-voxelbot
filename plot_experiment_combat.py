#!/usr/local/bin/python3
import numpy as np
import sys
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import matplotlib.cm
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 14
colormap = 3
defaultCM = matplotlib.cm.tab20

if colormap == 0:
    cm1 = matplotlib.cm.tab20b
    cm2 = matplotlib.cm.tab20c

    seenUnits = {
        "Archon": cm1(0+0),
        "Stalker": cm1(0+1),
        "Zealot": cm2(0+0),

        "SiegeTankSieged": cm1(12+0),
        "Marine": cm1(12+1),
        "Banshee": cm2(4+0),
        "Marauder": cm2(4+1),

        "Roach": cm1(16+0),
        "Hydralisk": cm1(16+1),
        "Zergling": cm1(12+0),
    }
elif colormap == 1:
    cm = matplotlib.cm.tab20

    seenUnits = {
        "Archon": cm(0),
        "Stalker": cm(1),

        "SiegeTankSieged": cm(2),
        "Marine": cm(3),

        "Roach": cm(4),
        "Hydralisk": cm(5),
        "Banshee": cm(6),
        "Marauder": cm(7),

        "Zealot": cm(8),
        "Zergling": cm(9),
    }
elif colormap == 2:
    seenUnits = {
        "Archon": "#116bb8",
        "Zealot": "#0e51cb",
        "Stalker": "#0e29cb",

        "Marine": "#c32626",
        "SiegeTankSieged": "#cf4b18",
        "Banshee": "#ce710e",
        "Marauder": "#ceb10e",

        "Roach": "#59119c",
        "Hydralisk": "#9b16c2",
        "Zergling": "#d30dc7",
    }
else:
    seenUnits = {
        "Archon": "#3481c3",
        "Zealot": "#326bd3",
        "Stalker": "#3249d3",

        "Marine": "#cc4646",
        "SiegeTankSieged": "#d6663a",
        "Banshee": "#d58632",
        "Marauder": "#d5bd32",

        "Roach": "#7234ab",
        "Hydralisk": "#aa39cb",
        "Zergling": "#d540cc",
    }


def getColor(name):
    if name not in seenUnits:
        seenUnits[name] = defaultCM(len(seenUnits))
    return seenUnits[name]


def parse(filename):
    owners = open(filename).read().strip().split('-------------')
    results = []
    for owner in owners[0:2]:
        lines = [l.strip().split('\t') for l in owner.strip().split('\n')]
        headers = lines[0][1:]
        lines = lines[1:]
        data = np.array([[float(x) for x in line] for line in lines])
        firstChange = np.argmax(data[:, 1:] != data[0, 1:], axis=0)
        order = firstChange.argsort()
        results.append((headers, data, order))

    return results



def plot(filename, axes, title, tmax, show_xlabel):
    assert len(axes) == 2
    owners = open(filename).read().strip().split('-------------')
    plot_owner(owners[0], axes[0], title, tmax, show_xlabel)
    plot_owner(owners[1], axes[1], title, tmax, show_xlabel)


def plot_owner(data_tuple, ax, title, tmax, show_xlabel):
    headers, data, sortOrder = data_tuple
    colors = [getColor(h) for h in headers]

    data = np.concatenate([data, [data[-1,:]]], axis=0)
    data[-1,0] = tmax

    times = data[:,0]
    data = data[:,1:]

    data = data[:, sortOrder]
    colors = np.array(colors)[sortOrder]
    headers = np.array(headers)[sortOrder]
    print(sortOrder)

    cnt = ax.stackplot(times, np.transpose(data), labels=headers, colors=colors, linewidth=0.3)
    for c in cnt:
        c.set_edgecolor("face")
    ax.set_title(title)
    ax.set_xlim([0, tmax])
    # ax.set_ylabel("Total Health [hp]")

    if show_xlabel:
        ax.set_xlabel("Time [s]")
    # ax.legend()


for si in range(6):

    tmax = 40
    prefix = "experiment_results/combat/test"

    indices = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        [(1, 0), (1, 2)],
        [(5, 0), (5, 3)],
        [(0, 0), (0, 4)],
        [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)],
    ]

    fig, axs = plt.subplots(len(indices[si]), 4, figsize=(9, 5), sharex=True, sharey='row')

    for row, i in enumerate(indices[si]):
        real1, real2 = parse(prefix + str(i[0]) + "_real.csv")
        sim1, sim2 = parse(prefix + str(i[0]) + f"_sim_{i[1]}.csv")
        sim1 = (sim1[0], sim1[1], real1[2])
        sim2 = (sim2[0], sim2[1], real2[2])

        plot_owner(real1, axs[row, 0], "Ground Truth" if row == 0 else "", tmax, False)
        plot_owner(real2, axs[row, 2], "Ground Truth" if row == 0 else "", tmax, False)
        plot_owner(sim1, axs[row, 1], "Simulated" if row == 0 else "", tmax, False)
        plot_owner(sim2, axs[row, 3], "Simulated" if row == 0 else "", tmax, False)
        # plot_owner(owners[1], axes[1], title, tmax, show_xlabel)

        # plot(, axs[i,[0, 2]], "Ground Truth" if i == 0 else "", tmax, show_xlabel=False)
        # plot(, axs[i,[1, 3]], "Simulated" if i == 0 else "", tmax, show_xlabel=False)

    # fig.text(0.5, 0.04, 'common X', ha='center')
    plt.subplots_adjust(left=0.1, bottom=0.22, right=0.95, top=0.88, wspace=0.10, hspace=0.24)
    fig.text(0.01, 0.5, 'Total Health [hp]', va='center', rotation='vertical')
    xmid = (fig.subplotpars.left + fig.subplotpars.right) * 0.5
    leftQuarter = (fig.subplotpars.left * 0.75 + fig.subplotpars.right * 0.25)
    rightQuarter = (fig.subplotpars.left * 0.25 + fig.subplotpars.right * 0.75)

    fig.text(xmid, 0.125, 'Time [s]', ha='center')
    fig.text(leftQuarter - 0.01, 0.96, 'Team 1', ha='center')
    fig.text(rightQuarter + 0.01, 0.96, 'Team 2', ha='center')

    xs = np.linspace(fig.subplotpars.left, fig.subplotpars.right, len(seenUnits), endpoint=True)
    for (x, (unit, color)) in zip(xs, seenUnits.items()):
        if unit == "SiegeTankSieged":
            unit = "Tank"
        fig.patches.append(matplotlib.patches.Rectangle((x - 0.025, 0.05), 0.05, 0.05, facecolor=color, zorder=10, edgecolor="#000000", transform=fig.transFigure, figure=fig))
        fig.texts.append(matplotlib.text.Text(x=x, y=0.01, text=unit, ha='center', transform=fig.transFigure, figure=fig))


    # plt.tight_layout(h_pad=0.0)
    

    pdf = PdfPages(f"/Users/arong/cloud/Skolarbeten/ML-2/thesis/draft/graphics/generated/combat_{si}.pdf")
    pdf.savefig(fig)
    pdf.close()
    # plt.savefig('../final_images/speed' + ("_annotated" if annotate else "") + '.png', transparent=True)


plt.show()