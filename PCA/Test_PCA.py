import Libraries as lib
import PCA

"This code's implementation is based on the mathematics derivation in https://www.coursera.org/learn/pca-machine-learning"

if __name__ =='__main__':

    X_small_dim = lib.np.random.randint(low=100,high=100000,size=(100,20))
    X_high_dim = lib.np.random.randint(low=100,high=100000,size=(20,100))

    # Test Data with low dimension than the number of samples
    X_small_dim = (X_small_dim - lib.np.mean(X_small_dim,axis=0))/(lib.np.std(X_small_dim,axis=0))
    pca_obj = PCA.PCA(Samples_Number=100,Sample_Dimensions_Number=20,Principal_Dimensions_Number=5)
    X_telda,_= pca_obj.Compute_PCA(X_small_dim)
    pca = lib.SKPCA(n_components=5, svd_solver='full')
    sklearn_reconst = pca.inverse_transform(pca.fit_transform(X_small_dim))
    lib.np.testing.assert_almost_equal(X_telda, sklearn_reconst)
    print("Success")

    # Test Data with high dimension than the number of samples
    X_high_dim = (X_high_dim - lib.np.mean(X_high_dim, axis=0)) / (lib.np.std(X_high_dim, axis=0))
    pca_obj = PCA.PCA(Samples_Number=20, Sample_Dimensions_Number=100, Principal_Dimensions_Number=5)
    X_telda_high, _ = pca_obj.Compute_PCA_High_Dim(X_high_dim)
    pca = lib.SKPCA(n_components=5, svd_solver='full')
    sklearn_reconst_high = pca.inverse_transform(pca.fit_transform(X_high_dim))
    lib.np.testing.assert_almost_equal(X_telda_high, sklearn_reconst_high)
    print("Success")