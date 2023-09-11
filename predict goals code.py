import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression

stats = pd.read_csv(r"stats.csv",
                    header = None, 
                    names = ['goals','fin', 'cmp', 
                    'bravery', 'xg','team'])
stats.head()


X = stats.iloc[:, 1:-1]
y = stats.loc[:, 'goals']
X.head()
y.head()


lr = LinearRegression()
lr.fit(X,y)
lr.score(X,y)

 
player_1 = [16, 14, 9, 4.01]
player_2 = [12, 14, 12, 2.50]
players = [player_1, player_2]

#goals_predict_Reischl = lr.predict(player_1)
#goals_predict_Nkunku = lr.predict(player_2)

goals_predict = lr.predict(players)
print(goals_predict)
