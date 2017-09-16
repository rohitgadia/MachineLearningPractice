import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random

def k_nearest_neighbors(data,new_feature,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(new_feature))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = { 2:[], 4:[] }
test_set = { 2:[], 4:[] }
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in test_data:
    test_set[i[-1]].append(i[:-1])
for i in train_data:
    train_set[i[-1]].append(i[:-1])

counter = 0
total_votes = 0

for groups in test_set:
    for features in test_set[groups]:
        group = k_nearest_neighbors(train_set, features, k=5)
        if group==groups:
            counter+=1
        total_votes+=1

accuracy = (counter/total_votes)*100

print(accuracy)
