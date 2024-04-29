import numpy as np


def pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Perform eigendecomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort the eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    components = sorted_eigenvectors[:, :n_components]

    # Project the data onto the selected components
    projected_data = np.dot(X_centered, components)

    return projected_data


if __name__ == "__main__":
    # Example dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # Number of components to keep
    n_components = 2

    # Apply PCA
    projected_data = pca(X, n_components)

    # Print the projected data
    print("Projected Data:")
    print(projected_data)
