import numpy as np
import sys
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import matplotlib.cm
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import pickle

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 14

savePath = "experiment_results/"


def save(label, times, scores, steps):
    with open(savePath + f"buildorder_{label}.pickle", "wb") as f:
        pickle.dump((times, scores, steps), f)


def visualize(times, scores, steps):
    print(len(times))
    print([len(t) for t in times])
    times, scores = zip(*[(t, s) for (t,s) in zip(times, scores) if t[-1] != 0])
    print(len(times))

    times = np.array(times)
    scores = np.array(scores)
    steps = np.array(steps)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    times /= 0.0001 + times[:, -1].reshape(times.shape[0], 1)
    # times = 1 / (0.001 + times)

    scores /= scores[:, -1].reshape(scores.shape[0], 1)
    # invscores = 1 / (0.001 + scores)

    plt.sca(axs[0])
    for i in range(times.shape[0]):
        plt.scatter(steps, times[i], color="#377eb8", alpha=0.1)

    meanCol = "#000000"
    Tmean = np.mean(times, axis=0)
    Tdev = np.std(times, axis=0)

    # plt.plot(steps, Tmean, color=meanCol)
    plt.errorbar(steps, Tmean, yerr=Tdev, color=meanCol, fmt="--", capthick=1, capsize=2)
    plt.xscale('log', basex=2)
    plt.ylim(0, 3.5)
    plt.xticks(steps[1:])

    Smean = np.mean(scores, axis=0)
    Sdev = np.std(scores, axis=0)

    # SmeanInv = 1 / (0.001 + Smean)
    # SdevInv = 1 / (0.001 + Sdev)

    plt.sca(axs[1])
    for i in range(times.shape[0]):
        plt.scatter(steps, scores[i], color="#377eb8", alpha=0.1)

    # plt.plot(steps, Smean, color=meanCol)
    plt.errorbar(steps, Smean, yerr=Sdev, color=meanCol, fmt='--', capthick=1, capsize=2)
    plt.xscale('log', basex=2)
    plt.xticks(steps[1:])
    plt.ylim(0, 3.5)
    # plt.yscale('log')

    plt.show()


if __name__ == "__main__":
    with open(savePath + f"buildorder_iterations.pickle", "rb") as f:
        times, scores, steps = pickle.load(f)
    visualize(times, scores, steps)

    with open(savePath + f"buildorder_genePoolSize.pickle", "rb") as f:
        times, scores, steps = pickle.load(f)
    visualize(times, scores, steps)

    with open(savePath + f"buildorder_varianceBias.pickle", "rb") as f:
        times, scores, steps = pickle.load(f)
    visualize(times, scores, steps)
