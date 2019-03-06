import sys
import random
import os
import gzip
import pickle
import json
import time
sys.path.append("build/bin")
import botlib_bindings as bl


def load(filepath):
    t1 = time.time()
    with gzip.open(filepath, 'rb') as f:
        s = pickle.load(f)

    t2 = time.time()
    js = json.dumps(s)
    t3 = time.time()

    print("Loading", (t2 - t1) * 1000, (t3 - t2) * 1000)
    return js


# replays_dir = "training_data/replays/s3"
# paths = os.listdir(replays_dir)
# random.shuffle(paths)
# for i, replay in enumerate(paths):
#     print(f"\r {i}/{len(paths)}")
#     dest = "training_data/replays/b1/" + replay.replace(".pickle", ".bin")
#     if not os.path.exists(dest):
#         bl.load(load(replays_dir + "/" + replay), dest)

data = bl.load(load("training_data/replays/s3/f0dbcf0f117aeaac1f1d953b532c5dbbfe694bab55ec4cbd961fa7624b17fe69.pickle"), "blah.temp")

# print(data.observations[0].selfStates)

with gzip.open("training_data/replays/s3/f0dbcf0f117aeaac1f1d953b532c5dbbfe694bab55ec4cbd961fa7624b17fe69.pickle", 'rb') as f:
    s = pickle.load(f)


for k in range(5):
    t1 = time.time()
    for i in range(10):
        cum = 0
        for state in s["observations"][0]["rawUnits"][0]["units"]:
            cum += state["unit_type"]

        print(cum)

    t2 = time.time()




    arr = data.observations[0].rawUnits[0].units
    for i in range(10):
        cum = 0
        for state in arr:
            # print(state.mineralsPerSecond)
            cum += state.unit_type
            # state.unit_type_id = 0

        print(cum)
    t3 = time.time()

    print((t2 - t1), (t3 - t2))

# print(data.observations[0].selfStates[0].mineralsPerSecond)
print(type(data))
