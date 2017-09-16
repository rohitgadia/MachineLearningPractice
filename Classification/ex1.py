from sklearn import preprocessing, cross_validation, neighbors
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = df.drop(['class'], 1)
y = df['class']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train,y_train)
#
# with open('cancer.pickle','wb') as f:
#     pickle.dump(clf,f)

pickle_in = open('cancer.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test,y_test)

print(accuracy)

new_measures = np.array([[4,2,1,1,1,3,2,1,2],[7,4,5,4,2,3,2,1,5]])
new_measures = new_meas;ures.reshape(2,-1);

classification = clf.predict(new_measures)

print(classification)
