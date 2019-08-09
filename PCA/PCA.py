import  Libraries as lib


class PCA:
    def __init__(self, Samples_Number, Sample_Dimensions_Number, Principal_Dimensions_Number):

        " Samples_Number is the size of data-set X.T =[x1,x2,...,xn] where n = Samples_Number\
            xi belongs to R-(D*1) for i = 1..n such that D = Sample_Dimensions_Number (xi.T belongs to R-(1*D) )\
            X =(X.T).T belongs to R-(N*D) <X is the expected input> and X.T belongs to R-(D*N)\
            xi = sum(ci*bi for i= 1...m ) + sum(ci*bi_ for  i= m+1...n) where bi_ is the  orthogonal complement of bi \
            m = Principal_Dimensions_Number, S is the covariance matrix of X, proj_mat is the projection matrix that \
            project xi orthogonally onto the principal subspace \
            S = (1/n)*X.T@X, proj_mat = B@B.T, B contains the (Principal_Dimensions_Number = m) eigenvectors \
            corresponding to the m largest eigenvalues of S, @ denote the matrix multiplication operator"

        self.__S = lib.np.zeros((Sample_Dimensions_Number,Sample_Dimensions_Number))
        self.__B = lib.np.zeros((Sample_Dimensions_Number,Principal_Dimensions_Number))
        self.__Proj_mat = lib.np.zeros((Principal_Dimensions_Number,Principal_Dimensions_Number))
        self.__n = Samples_Number
        self.__m = Principal_Dimensions_Number
        return
    def __Compute_Covariance_X(self,X):

        "self.__S which equal to the covariance matrix in case of \
        low dimension data and equal to (1/N)*(X.T @ X) in case of high dimensional data"

        self.__S = (1/self.__n) * (X.T @ X)

        return
    def __Compute_Eigen_Values_Vectors_S(self):

        "Calcute the eigenvalues and the eigenvector of self.__S which equal to the covariance matrix in case of \
        low dimension data and equal to (1/N)*(X.T @ X) in case of high dimensional data"

        eig_values, eig_vectors = lib.np.linalg.eig(self.__S)

        return eig_values, eig_vectors

    def __Construc_B(self, eig_values, eig_vectors):

        "B contains the first m eigenvectors corresponding the largest m eigenvalue "

        sorted_indexs = lib.np.argsort(eig_values)
        sorted_indexs = sorted_indexs[::-1]
        eig_vectors = eig_vectors[:,sorted_indexs]
        self.__B = eig_vectors[:,:self.__m]

        return

    def __Recover_Eigen_Vectors_Covariance(self,X):

        "This function recover the eigenvectors of the covariance matrix of the dataset \
        by matrix multiplication between the eigenvectors of (1/n)X@X.T and X.T\
        the normalization of B guarantees that each vector has length 1 it is necessary when dealing with\
        high dimensional data even the assumption of orthonormal basis be true "

        self.__B = X.T @ self.__B
        norm = lib.np.linalg.norm(self.__B, axis=0)
        for i in range(self.__m):
            self.__B[:, i] = self.__B[:, i] * (1 / norm[i])

        return

    def __Construct_Projection_Matrix(self):

        "This function compute and return the projection matrix that orthogonally project xi into the principal subspace "

        self.__Proj_mat = self.__B @ self.__B.T

        return

    def __Compute_Approximated_Optimal_Coordinate(self,X):

        "This function compute and return the coordinates according to the principal subspace of each vector in X.T\
         B belongs to R-(D*M), X.T belongs to R-(D*N) then B.T @ X.T belongs to R-(M*N), the result must be the same\
         organization of the input X, we assume that X=(X.T).T belongs to R-(N*D) then the output is code.T"

        Code = self.__B.T @ X.T

        return Code.T
    def __Compute_approximated_Vectors_In_Principal_Subspace(self,X):

        "We assume that X belongs to R-(N*D) so X.T belongs to R-(D*N), B belongs to R-(D*M) "

        X_telda = self.__B @ self.__B.T @ X.T

        return X_telda.T

    def Compute_PCA(self,X):

        "We assume that X belongs to R-(N*D) with mean = 0 and std = 1 this function compute and return PCA"

        self.__Compute_Covariance_X(X)
        eig_values,eig_vectors = self.__Compute_Eigen_Values_Vectors_S()
        self.__Construc_B(eig_values,eig_vectors)
        self.__Construct_Projection_Matrix()
        Code_X = self.__Compute_Approximated_Optimal_Coordinate(X)
        X_telda = self.__Compute_approximated_Vectors_In_Principal_Subspace(X)

        return X_telda, Code_X

    def Compute_PCA_High_Dim(self,X):

        "This function is computationally efficient for high dimensional data with the assumption\
        that the number of samples less than the number of dimension of X "

        self.__Compute_Covariance_X(X.T)
        eig_values, eig_vectors = self.__Compute_Eigen_Values_Vectors_S()
        self.__Construc_B(eig_values, eig_vectors)
        self.__Recover_Eigen_Vectors_Covariance(X)
        self.__Construct_Projection_Matrix()
        Code_X = self.__Compute_Approximated_Optimal_Coordinate(X)
        X_telda = self.__Compute_approximated_Vectors_In_Principal_Subspace(X)

        return X_telda, Code_X