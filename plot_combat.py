#!/usr/local/bin/python3
import numpy as np
import sys
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 14


def plot(filename, axes, title, tmax):
    assert len(axes) == 2
    owners = open(filename).read().strip().split('-------------')
    plot_owner(owners[0], axes[0], title + " Team 1", tmax)
    plot_owner(owners[1], axes[1], title + " Team 2", tmax)


def plot_owner(owner, ax, title, tmax):
    lines = [l.strip().split('\t') for l in owner.strip().split('\n')]
    headers = lines[0][1:]
    lines = lines[1:]

    data = np.array([[float(x) for x in line] for line in lines])

    times = data[:,0]
    data = data[:,1:]
    ax.stackplot(times, np.transpose(data), labels=headers)
    ax.set_title(title)
    ax.set_xlim([0, tmax])
    ax.set_ylabel("Total Health+Shield [hp]")
    ax.set_xlabel("Time [s]")
    ax.legend()


fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey='col')

tmax = 40
prefix = "recording"

if len(sys.argv) > 1:
    prefix = sys.argv[1]

plot("recording.csv", axs[0,:], "Ground Truth", tmax)
plot(prefix + "2.csv", axs[1,:], "Simulated", tmax)
# plot(prefix + "4.csv", axs[2,:], "Simulated (bad micro)", tmax)
plot(prefix + "4.csv", axs[2,:], "Simulated (restarted)", tmax)
plt.show()
