import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

strikers = pd.read_csv('stats3.csv', names=['goals', 'fin', 'comp', 'brav', 'xg'])

x = strikers[['fin', 'comp', 'brav']].copy()
km = KMeans(n_clusters = 6)

    
strikers['cluster'] = km.fit_predict(x) 
strikers['cluster'] = strikers['cluster'].astype('category')



fig = px.scatter_3d(strikers,
                    x='brav',
                    y='comp',
                    z='fin',
                    color='cluster')

fig.show(renderer='browser')
      