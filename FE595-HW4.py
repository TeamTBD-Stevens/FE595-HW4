# import packages, set directory

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

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

# store the coefficients next to corresponding feature
# names
coef_df = pd.DataFrame()
coef_df['coefficient'] = np.abs(model.coef_)
coef_df['feature'] = bos.feature_names
print(coef_df.sort_values(by='coefficient', ascending=False))



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
