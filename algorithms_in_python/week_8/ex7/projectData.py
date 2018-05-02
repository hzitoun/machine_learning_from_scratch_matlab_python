def projectData(X, U, K):
    """PROJECTDATA Computes the reduced data representation when projecting only
     on to the top k eigenvectors
       Z = projectData(X, U, K) computes the projection of
       the normalized inputs X into the reduced dimensional space spanned by
       the first K columns of U. It returns the projected examples in Z.
    """
    Ureduce = U[:, 0: K]  # take the first k directions
    Z = X.dot(Ureduce)  # compute the projected data points
    return Z
