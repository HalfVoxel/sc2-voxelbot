import json
import pickle
import gzip

def save(jsonData, filepath):
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(json.loads(jsonData), f, protocol=pickle.HIGHEST_PROTOCOL)
