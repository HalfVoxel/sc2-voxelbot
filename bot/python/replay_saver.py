import json
import pickle
import random
import gzip
import torch
import os


def save(jsonData, filepath):
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(json.loads(jsonData), f, protocol=pickle.HIGHEST_PROTOCOL)


def saveTensors(binaryData, filepath):
    data = [(torch.tensor(x) if isinstance(x, list) else x) for x in binaryData]
    torch.save(data, filepath)


def isReplayAlreadySaved(filepath):
    return os.path.exists(filepath)


def replaySavePath(replayPath, outputDir):
    return os.path.join(outputDir, replayPath.split("/")[-1].replace(".SC2Replay", ".pickle"))


def findReplays(file):
    files = open(file).read().strip().split('\n')
    random.shuffle(files)
    return files


def getPort():
    try:
        with open(".sc2port") as f:
            lastPort = int(f.read())
    except:
        lastPort = 10000

    lastPort += 2
    if lastPort >= 11000:
        lastPort = 10000

    with open(".sc2port", "w") as f:
        f.write(str(lastPort))

    return lastPort
