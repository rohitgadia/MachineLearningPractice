import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
from collections import Counter

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

dataset = {'k':[[1,2],[2,3],[2,1]],'r':[[4,5],[5,7],[7,6]]}

new_feature = [1,3]

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
vote = k_nearest_neighbors(dataset,new_feature)
plt.scatter(new_feature[0],new_feature[1],s=100,color=vote)
plt.show()
