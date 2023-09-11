import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


%matplotlib inline

stats = pd.read_csv(r"stats3.csv",
                    header = None, 
                    names = ['goals','fin', 'cmp', 
                    'bravery', 'xg'])
stats.head()


g_xg = stats.iloc[:,[4,0]]
#g_f = stats.iloc[:,[1,0]]

points = g_xg.to_numpy()


simplices = Delaunay(points).simplices

plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')
plt.xlabel(r"xg")
plt.ylabel(r"goals")

plt.show()



hull = ConvexHull(points)
hull_points = hull.simplices

plt.scatter(points[:,0], points[:,1])
for simplex in hull_points:
  plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.xlabel(r"xg")
plt.ylabel(r"goals")
plt.show()

