import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

strikers = pd.read_csv('stats3.csv', names=['goals', 'fin', 'comp', 'brav', 'xg'])


X = np.array(strikers)
y = np.array(np.zeros(len(strikers)))

names =['fin', 'comp', 'brav']

for i in range(0,len(strikers)):
    if (X[i][0]>(1.2*X[i][4])):
        y[i]=1
    else:
        y[i]=0
    
  
    
X = np.delete(X,0,1)
X = np.delete(X,3,1)



strikers_tree = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=1, min_impurity_decrease=0.001) 
strikers_tree.fit(X,y)

plt.figure(figsize=(12,12))
tree.plot_tree(strikers_tree,fontsize=12, feature_names=names, class_names=['0','1'])
plt.show()

'''
import graphviz 
dot_data = tree.export_graphviz(strikers_tree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("strikers_tree") 
'''