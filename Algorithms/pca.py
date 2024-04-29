# import all libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import the breast _cancer dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data.keys()

# Check the output classes
print(data['target_names'])

# Check the input attributes
print(data['feature_names'])

# construct a dataframe using pandas
df1 = pd.DataFrame(data['data'], columns=data['feature_names'])

# Scale data before applying PCA
scaling = StandardScaler()

# Use fit and transform method
scaling.fit(df1)
Scaled_data = scaling.transform(df1)

# Set the n_components=3
principal = PCA(n_components=3)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print("\nDimensions of data after PCA:", x.shape)
print()
# Check the values of eigen vectors
# produced by principal components
print(principal.components_)


# check how much variance is explained by each principal component
print("\nVariance explained by each principle component:", principal.explained_variance_ratio_)
# array([0.44272026, 0.18971182, 0.09393163])
