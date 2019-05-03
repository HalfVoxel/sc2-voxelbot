import numpy as np
import sys
import math
import os
import pickle

savePath = "experiment_results/buildorder_sim/"


def save(buildOrder, chronoBoosted, expectedTimings, realTimings):
    index = 0
    while True:
        path = savePath + f"buildorder_{index}.pickle"
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump((buildOrder, chronoBoosted, expectedTimings, realTimings), f)
            break
        
        index += 1


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
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.cm
    from matplotlib.ticker import MultipleLocator
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.patches import Rectangle

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['CMU Serif']
    rcParams['font.size'] = 14

    for correctTimings in [False, True]:
        allRealT = []
        allSimT = []
        for path in os.listdir(savePath):
            if path.endswith(".pickle"):
                with open(savePath + path, "rb") as f:
                    buildOrder, chronoBoosted, expectedTimings, realTimings = pickle.load(f)
                    if len(expectedTimings) > 0:
                        print(np.array(expectedTimings, dtype=np.float32)[-10:] / np.array(realTimings, dtype=np.float32)[-10:])
                        mx = ((1 - (np.array(expectedTimings, dtype=np.float32)[-10:] / np.array(realTimings, dtype=np.float32)[-10:]))**2).mean()
                        print(mx)
                        if mx > 0.01 or True:
                            allRealT.append(realTimings)
                            allSimT.append(expectedTimings)
                            # plt.plot(realTimings, expectedTimings, label=path)

        
        allRealT = np.concatenate(allRealT)
        allSimT = np.concatenate(allSimT)
        if correctTimings:
            allSimT /= 0.9662882837688399
        mx = max(allRealT.max(), allSimT.max())

        fit = np.polyfit(allRealT, allSimT, 1)
        fit_fn = np.poly1d(fit) 
        xs = np.linspace(0, mx, 2)

        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(xs, fit_fn(xs), "--", color="#000000")
        plt.scatter(allRealT, allSimT, color="#e41a1c", s=6, edgecolors="none", alpha=0.3)
        
        # plt.axis('scaled')
        plt.plot([0, mx], [0, mx], c="#000000", alpha=0.5)
        # plt.ylim(bottom=-1, top=mx)
        # plt.xlim(left=-1, right=mx)
        # plt.xlabel("Game Time [s]")
        plt.ylabel("Sim. Time [s]")

        plt.subplot(2, 1, 2)

        plt.plot(xs, fit_fn(xs) - xs, "--", color="#000000")
        plt.scatter(allRealT, allSimT - allRealT, color="#e41a1c", edgecolors="none", s=6, alpha=0.3)
        plt.plot([0, mx], [0, 0], c="#000000", alpha=0.5)
        # plt.ylim(bottom=-1, top=mx)
        # plt.xlim(left=-1, right=mx)
        plt.yticks([0, -50, 50])
        plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
        plt.xlabel("Game Time [s]")
        plt.ylabel("Relative Sim. Time [s]")
        print(fit_fn(mx)/mx)

        plt.subplots_adjust(bottom=0.13, top=0.99)
        
        print(len(allSimT))
        pdf = PdfPages(f"/Users/arong/cloud/Skolarbeten/ML-2/thesis/draft/graphics/generated/buildorder_sim_{'adjusted' if correctTimings else 'original'}.pdf")
        pdf.savefig(fig)
        pdf.close()
        

        # plt.legend()
    # plt.show()

