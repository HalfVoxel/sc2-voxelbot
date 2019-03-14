import os
import math
import numpy as np
from mappings import UnitLookup, terranUnits, zergUnits, protossUnits
import random
from collections import namedtuple
import mappings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Turn on non-blocking plotting mode
# plt.ion()

previous_units = []
unit_offsets = {}
last_time = 0
frame = 0
last_unit_type_counts = [{}, {}]

Unit = namedtuple("Unit", "x, y, owner, tag, unit_type, health, max_health")

def lerp(a, b, t):
    return a + (b - a) * t


def animate_unit1(unit1, unit2, time):
    unit1 = Unit(*unit1) if unit1 is not None else None
    unit2 = Unit(*unit2) if unit2 is not None else None

    unit = unit1 if unit1 is not None else unit2


    if unit.tag not in unit_offsets:
        spread = 2
        unit_offsets[unit.tag] = (random.uniform(-spread, spread), random.uniform(-spread, spread))
    
    offset = unit_offsets[unit.tag]

    x = unit.x
    y = unit.y
    from_health = unit.health
    to_health = unit2.health if unit2 is not None else 0
    health_fraction = lerp(from_health, to_health, time) / unit.max_health

    if unit1 is not None and unit2 is not None:
        # Interpolate
        x = lerp(unit1.x, unit2.x, time)
        y = lerp(unit1.y, unit2.y, time)
    
    color = np.array([55,126,184]) if unit.owner == 1 else np.array([228,26,28])
    color = color * health_fraction

    area = 3 * math.sqrt(unit.max_health)

    if unit2 is None:
        area *= (1 - time)**2

    x += offset[0]
    y += offset[1]

    return x, y, color, area
    
def visualize(units, state_time, health_fraction1, health_fraction2):
    global previous_units, last_time, frame, last_unit_type_counts
    frame += 1
    all_units = [(u, None) for u in previous_units]

    for u in units:
        tag = u[3]
        found = 0
        for i in range(len(all_units)):
            if all_units[i][0] is not None and tag == all_units[i][0][3]:
                all_units[i] = (all_units[i][0], u)
                found += 1
        
        if found > 1:
            raise Exception("Duplicate units")
        
        if found == 0:
            all_units.append((None, u))
    
    previous_units = units

    unit_type_counts = [{ k: 0 for k in counts.keys() } for counts in last_unit_type_counts]
    for u in units:
        unit = Unit(*u)
        tp = int(unit.unit_type)
        if tp not in unit_type_counts[int(unit.owner) - 1]:
            unit_type_counts[int(unit.owner) - 1][tp] = 0
        unit_type_counts[int(unit.owner) - 1][tp] += 1

    last_unit_type_counts = unit_type_counts

    ts = np.linspace(0, 1, 1 * round(state_time - last_time))

    # f, axs = plt.subplots(1, 1, num=1)
    # plt.sca(axs)
    # plt.scatter([], [], c=[])
    # plt.show()

    lookup = mappings.UnitLookup(mappings.terranUnits + mappings.zergUnits + mappings.protossUnits)


    for t in ts:
        actual_time = lerp(last_time, state_time, t)
        print(f"Time: {actual_time}")
        f, axs = plt.subplots(2, 2, num=1, clear=True, figsize=(8, 8))
        
        data = [animate_unit1(u1, u2, t) for (u1, u2) in all_units]
        xs = np.array([v[0] for v in data])
        ys = np.array([v[1] for v in data])
        colors = np.array([v[2] for v in data]) / 255
        alphas = np.array([v[3] for v in data])

        plt.sca(axs[0, 0])
        plt.title(f"{frame} {round(actual_time)}")
        plt.xlim(0, 120)
        plt.ylim(0, 120)
        plt.scatter(xs, ys, c=colors, alpha=0.5, s=alphas)

        plt.sca(axs[0, 1])
        plt.ylim(0, 1)
        plt.bar([1, 2], [health_fraction1, health_fraction2])

        for playerIndex in [0, 1]:
            plt.sca(axs[1, playerIndex])
            tps = [(k, v) for (k, v) in unit_type_counts[playerIndex].items()]
            vals = np.array([x[1] for x in tps])
            indices = np.array([x[0] for x in tps])
            names = [lookup[x].name.replace("PROTOSS_", "").title() for x in indices]
            plt.bar(list(range(len(names))), vals, tick_label=names, color=np.array([55,126,184])/255 if playerIndex == 0 else np.array([228,26,28])/255)
            plt.xticks(rotation=90)
        plt.pause(0.0001)
    
    last_time = state_time


def visualize_bar(stats):
    labels = [
        "ArmyAttackClosestEnemy",
        "IdleArmyAttackClosestEnemy",
        "IdleNonArmyAttackClosestEnemy",
        "NonArmyAttackClosestEnemy",
        "ArmyConsolidate",
        "IdleArmyConsolidate",
        "ArmyMoveC1",
        "ArmyMoveC2",
        "None",
        "ArmyMoveBase",
        "NonArmyMoveBase",
        "ArmyAttackBase",
        "ArmySuicide",
        "?",
        "?",
    ]
    data = np.array(stats)
    percent = data.transpose()
    percent = percent / percent.sum(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.stackplot(np.arange(len(stats)), percent)
    plt.legend(labels)
    plt.show()
