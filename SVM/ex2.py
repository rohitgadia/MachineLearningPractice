import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors  = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fix.add_subplot(1,1,1)

    def fit(self,data):
        self.data = data
        opt_dict = {}

        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]

        all_data = []

        for yi in data:
            for featureset in data[yi]:
                for features in featureset:
                    all_data.append(features)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value*0.1,self.max_feature_value*0.01,self.max_feature_value*0.001]

        self.b_range_multiple = 5

        self.b_multiple = 5

        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                pass

    def predict(self,features):
        #sign(w.x+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        return classification

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),

             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}