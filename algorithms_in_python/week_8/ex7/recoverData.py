def recoverData(Z, U, K):
    """RECOVERDATA Recovers an approximation of the original data when using the
        projected data
       X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
       original data that has been reduced to K dimensions. It returns the
       approximate reconstruction in X_rec.
    """
    Ureduce = U[:, 0:K]  # take the first k directions
    X_rec = Z.dot(Ureduce.T)  # go back to our original number of features
    return X_rec
