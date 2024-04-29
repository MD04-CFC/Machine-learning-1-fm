import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import re
#%matplotlib inline

import time

 
def bin(w1,w2,w3,gracz):
    if((w1*gracz[0]+w2*gracz[2]+w3*gracz[2])>200):
        print("1")
    else:
        print("-1")
    

    

class Perceptron:
    
    def __init__(self, eta=0.004, epochs=1000,is_verbose = False):
        
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
              
                
                


def fun(rozmiar,wsp,eta2,epochs2):
    start_time = time.time()
    stats = pd.read_csv(r"stats3.csv",
                       header = None, 
                       names = ['goals', 'finishing', 
                                'composure', 'bravery', 'xg']) 

    with open(r"stats3.csv", 'r') as d:
        a=len(d.readlines())
    y=np.array(np.zeros(a))

    stats.head()
    X=np.array(stats)
    
    for i in range(0,a):
        
        if (X[i][0]>(wsp * X[i][4])):
            y[i]=1
        else:
            y[i]=-1
           


    X=np.delete(X,0,1)
    X=np.delete(X,3,1)


    w1=0
    w2=0
    w3=0

    
    
    for i in range(0,rozmiar):
        perceptron = Perceptron(eta=eta2, epochs=epochs2, is_verbose=False)            
        perceptron.fit(X, y)
        #print(perceptron.w)
        
        w1 += perceptron.w[0]
        w2 += perceptron.w[1]
        w3 += perceptron.w[2]


    srednie_w1 = w1/rozmiar   
    srednie_w2 = w2/rozmiar   
    srednie_w3 = w3/rozmiar    

    print("\n")
    print("\n")
    print(srednie_w1)

    print(srednie_w2)

    print(srednie_w3)
    print("\n")
    
    data_collected = np.loadtxt(r"stats_4.txt",delimiter=',', dtype=int)
    
    for i in range(0,6):
        print(data_collected[i])
        bin(srednie_w1, srednie_w2, srednie_w3, data_collected[i])
        print("\n")
        
      
    print("--- %s seconds ---" % (time.time() - start_time))
    d.close()      


  
fun(1000,1.2,0.001,1000)
fun(1000,1.2,0.004,1000)
fun(1000,1.2,0.01,1000) # perfect
fun(1000,1.2,0.02,1000)







