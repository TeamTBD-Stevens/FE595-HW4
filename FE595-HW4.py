# import packages, set directory

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV

# Part I

bos = load_boston()
print(bos.keys())
print(bos.DESCR)

df_bos = pd.DataFrame(bos.data, columns=bos.feature_names)
df_bos.head()
# The MEDV target variable, as is listed in the DESCR command, is missing

df_bos['MEDV'] = bos.target
df_bos.head()
# The MEDV target variable now appears in the dataframe

df_features = df_bos.drop("MEDV",1)
df_target = df_bos["MEDV"]

# to figure out how much impact each factor has on the target
# variable, we need to apply a few different feature
# selection methods


# create a linear regression model
model = LinearRegression()
model.fit(df_features, df_target)

# find the slopes and intercept
model.intercept_
model.coef_

# test the model to see if it works
model.predict([[-1,3,5,7,8,9,2,4,3,2,4,7,2]])

# find the correlation coefficient of the model
model.score(df_features, df_target)

# filter method using pearson correlation
pearson = abs(df_bos.corr()['MEDV'])
print(pearson.sort_values(ascending=False))

# it should be noted that some of these features may be
# highly correlated with each other.



# For part II, we will use the Iris dataset
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.keys())
print(iris.DESCR)

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris.head()

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_iris)
kmeans.labels_

results = []
for _ in range(1, 10):
    kmeans = KMeans(n_clusters=_, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit(df_iris)
    results.append(kmeans.inertia_)

plt.plot(range(1, 10), results, color='lightsteelblue')
plt.show()

# based on the elbow heuristic, we can confirm that 3 is
# the correct number of clusters to use for the iris
# dataset.