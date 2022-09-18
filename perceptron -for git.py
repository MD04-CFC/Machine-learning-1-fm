import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import re
#%matplotlib inline


stats = pd.read_csv(r"stats3.csv",
                   header = None, 
                   names = ['goals', 'finishing', 
                            'composure', 'bravery', 'xg']) 


y=np.array(np.zeros(len(stats)))

stats.head()
 


class Perceptron:
    
    def __init__(self, eta=0.01, epochs=50,is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
        
    def predict(self, x):
        
        total_stimulation = np.dot(x, self.w)       
        y_pred = 1 if total_stimulation > 200 else -1
        return y_pred
        
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        self.w = np.random.rand(X.shape[1])
        
        for e in range(self.epochs):
 
            number_of_errors = 0 
            
            for x, y_target in zip(X,y):
            
                y_pred = self.predict(x)
                delta_w = self.eta * (y_target - y_pred) * x
                self.w += delta_w
                
                number_of_errors += 1 if y_target != y_pred else 0
                
            self.list_of_errors.append(number_of_errors)
        
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, number of errors {}".format(
                        e, self.w, number_of_errors))
              
                
                
X=np.array(stats)


for i in range(0,len(stats)):
    if (X[i][0]>(1.2 * X[i][4])):
        y[i]=1
    else:
        y[i]=-1
       


X=np.delete(X,0,1)
X=np.delete(X,3,1)



 
perceptron = Perceptron(eta=0.4, epochs=1000, is_verbose=False)            
perceptron.fit(X, y)
print(perceptron.w)


file2 = open(r"stats_5.txt", "w")
data_collected = np.loadtxt(r"stats_4.txt",delimiter=',', dtype=int)
print(data_collected)


for i in range(0,len(data_collected)):
    print(perceptron.predict(data_collected[i]))
    if (perceptron.predict(data_collected[i])==1):
        file2.writelines('1')
        file2.writelines('\n')
    else:   
        file2.writelines('-1')
        file2.writelines('\n')
        
        
                 
file2.close()        
     










