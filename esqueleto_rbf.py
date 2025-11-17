import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy.spatial.distance import pdist, squareform


class RBFNN(BaseEstimator):
    def __init__(
        self,
        classification: bool = False,
        ratio_rbf: float = 0.1,
        l2: bool = False,
        eta: float = 0.01,
        logisticcv: bool = False,
        random_state: int = 0,
    ) -> None:
        """
        Constructor of the class

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: float
            True if we want to use L2 regularization for logistic regression
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        logisticcv: bool
            True if we want to use LogisticRegressionCV
        random_state: int
            Seed for the random number generator
        """

        self.classification = classification
        self.ratio_rbf = ratio_rbf
        self.l2 = l2
        self.eta = eta
        self.logisticcv = logisticcv
        self.random_state = random_state
        self.is_fitted = False

    def fit(self, X: np.array, y: np.array):
        """
        Fits the model to the input data

        Parameters
        ----------
        X: array, shape (n_patterns,n_inputs)
            Matrix with the inputs for the patterns to fit
        y: array, shape (n_patterns,n_outputs)
            Matrix with the outputs for the patterns to fit

        Returns
        -------
        self: object
            Returns an instance of self
        """

        np.random.seed(self.random_state)
        self.num_rbf = int(self.ratio_rbf * y.shape[0])
        print(f"Number of RBFs used: {self.num_rbf}")
        # 1. Init centroids
        # TODO: Call the appropriate function

        # 2. clustering
        # TODO: Call the appropriate function

        # 3. Calculate radii
        # TODO: Call the appropriate function

        # 4. R matrix
        # TODO: Call the appropriate function

        # 5. Calculate betas
        if self.classification:
            # TODO: Call the appropriate function
        else:
            # TODO: Call the appropriate function

        self.is_fitted = True

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the output of the model for a given input

        Parameters
        ----------
        X: array, shape (n_patterns,n_inputs)
            Matrix with the inputs for the patterns to predict

        Returns
        -------
        predictions: array, shape (n_patterns,n_outputs)
            Predictions for the patterns in the input matrix
        """

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        # 2. clustering
        # TODO: Call the appropriate function

        # 4. radii for test set
        # TODO: Call the appropriate function

        if self.classification:
            # TODO: Call the appropriate function
        else:
            # TODO: Call the appropriate function 

        return predictions

    def score(self, X: np.array, y: np.array):
        """
        Returns the score of the model for a given input and output

        Parameters
        ----------
        X: array, shape (n_patterns,n_inputs)
            Matrix with the inputs for the patterns to predict
        y: array, shape (n_patterns,n_outputs)
            Matrix with the outputs for the patterns to predict

        Returns
        -------
        score: float
            Score of the model for the given input and output. It can be
            accuracy or mean squared error depending on the classification
            parameter
        """
        if self.classification:
            # TODO: Return the appropriate values

        else:
            # TODO: Return the appropriate values

    def _init_centroids_classification(
        self, X_train: np.array, y_train: np.array
    ) -> np.array:
        """
        Initialize the centroids for the case of classification

        This method selects num_rbf patterns in a stratified manner.


        Parameters
        ----------
        X_train: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        y_train: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        centroids: array, shape (num_rbf)
            Array with the centroids selected
        """

        # TODO: Complete the code of the function
        
        return centroids

    def _clustering(self, X_train: np.array, y_train: np.array) -> tuple[KMeans]:
        """
        Apply the clustering process

        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification().
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        X_train: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        y_train: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        """
        # TODO: Complete the code of the function
        
        return kmeans

    def _calculate_radii(self) -> np.array:
        """
        Obtain the value of the radii after clustering

        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
        """
        # TODO: Complete the code of the function
        
        return radii

    def _calculate_r_matrix(self, distances: np.array) -> np.array:
        """
        Obtain the R matrix

        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias

        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center

        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        """
        # TODO: Complete the code of the function

        return r_matrix

    def _invert_matrix_regression(
        self, r_matrix: np.array, y_train: np.array
    ) -> np.array:
        """
        Invert the matrix for regression case

        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression

        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        y_train: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value
            of the bias
        """
        # TODO: Complete the code of the function
        
        return coefficients

    def _logreg_classification(self) -> LogisticRegression | LogisticRegressionCV:
        """
        Perform logistic regression training for the classification case

        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)

        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression or LogisticRegressionCV
            Scikit-learn logistic regression model already trained
        """
        # TODO: Complete the code of the function

        return logreg
