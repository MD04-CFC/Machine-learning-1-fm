import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


stats = pd.read_csv(r"stats2.csv",
                    header = None, 
                    names = ['goals','fin', 'cmp', 
                    'bravery', 'xg','team'])
stats.head()

""" another program- predict goals code.py
A = stats.iloc[:, 1:-1]
b = stats.loc[:, 'goals']
A.head()
b.head()
stats=stats+[10,16,14,9,4.01,'Fio]
lr = LinearRegression()
lr.fit(A,b)
lr.score(A,b)
player_1 = [16, 14, 9, 4.01]
player_2 = [12, 14, 12, 2.50]
players = [player_1, player_2]
goals_predict=lr.predict(players)
#goals_predict_Reischl = lr.predict(player_1)
#goals_predict_Nkunku = lr.predict(player_2)
print(goals_predict)
"""


stats.shape
stats.shape[0]
stats.shape[1]


x_min, x_max = stats['xg'].min(), stats['xg'].max()
y_min, y_max = stats['goals'].min(), stats['goals'].max()
 
colors = {'Fio':'purple', 'Int':'navy', 
          'Mil':'red','Zeb':'black',
          'Laz':'aqua','Ata':'blue',
          'Sam':'snow','Cag':'crimson',
          'Udi':'gray','Sas':'green',
          'Sal':'brown','Tor':'peru'}
 

pd.plotting.scatter_matrix(stats, figsize=(18, 18), 
                           color = stats['team'].apply(lambda x: colors[x]));
plt.show()
 

import seaborn as sns
sns.set()
sns.pairplot(stats, hue="team", hue_order=None, palette=colors)




