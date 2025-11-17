import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.spatial.distance import pdist, squareform, cdist


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
        l2: bool
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
        
        # apply kmeans clustering to establish rbf centers
        self.kmeans = self._clustering(X, y)

        # calculate radii using heuristic formula
        self.radii = self._calculate_radii()

        # calculate distances from each training pattern to each rbf center
        distances = cdist(X, self.kmeans.cluster_centers_, metric='euclidean')

        # calculate r matrix (rbf activations + bias column)
        self.r_matrix = self._calculate_r_matrix(distances)

        # store y_train for classification
        self.y_train = y

        # calculate output layer weights
        if self.classification:
            # use logistic regression for classification
            self.logreg = self._logreg_classification()
        else:
            # use moore-penrose pseudo-inverse for regression
            self.coefficients = self._invert_matrix_regression(self.r_matrix, y)

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

        # 1. calculate distances from test patterns to RBF centers
        distances = cdist(X, self.kmeans.cluster_centers_, metric='euclidean')

        # 2. calculate R matrix for test set (RBF activations + bias column)
        r_matrix_test = self._calculate_r_matrix(distances)

        # 3. make predictions based on problem type
        if self.classification:
            # for classification: use logistic regression
            predictions = self.logreg.predict(r_matrix_test)
        else:
            # for regression: multiply coefficients by R matrix
            # coefficients shape: (n_outputs, num_rbf+1)
            # r_matrix_test shape: (n_patterns, num_rbf+1)
            # predictions shape: (n_patterns, n_outputs)
            predictions = r_matrix_test @ self.coefficients.T

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
        # get predictions
        predictions = self.predict(X)
        
        if self.classification:
            # for classification: return accuracy score
            # handle one-hot encoded y if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y.flatten()
            
            return accuracy_score(y_true, predictions)
        else:
            # for regression: return negative mean squared error
            # (sklearn convention: higher is better, so we negate MSE)
            return -mean_squared_error(y, predictions)

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
        centroids: array, shape (num_rbf, n_inputs)
            Array with the centroids selected
        """
        # convert y_train to class labels if it's one-hot encoded
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_labels = np.argmax(y_train, axis=1)
        else:
            y_labels = y_train.flatten()
        
        # get unique classes
        unique_classes = np.unique(y_labels)
        
        # calculate how many centroids per class (proportional to class distribution)
        centroids_per_class = []
        remaining = self.num_rbf
        
        for i, class_label in enumerate(unique_classes):
            class_count = np.sum(y_labels == class_label)
            if i == len(unique_classes) - 1:
                # last class gets remaining centroids
                n_centroids = remaining
            else:
                # proportional allocation
                proportion = class_count / len(y_labels)
                n_centroids = max(1, int(self.num_rbf * proportion))
                remaining -= n_centroids
            centroids_per_class.append(n_centroids)
        
        # select patterns from each class
        selected_indices = []
        for class_label, n_centroids in zip(unique_classes, centroids_per_class):
            # get indices of patterns belonging to this class
            class_indices = np.where(y_labels == class_label)[0]
            
            # randomly select n_centroids patterns from this class
            selected_class_indices = np.random.choice(
                class_indices, 
                size=min(n_centroids, len(class_indices)), 
                replace=False
            )
            selected_indices.extend(selected_class_indices)
        
        # if we need more centroids (due to rounding), randomly select from all
        if len(selected_indices) < self.num_rbf:
            remaining_needed = self.num_rbf - len(selected_indices)
            all_indices = np.arange(len(X_train))
            remaining_indices = np.setdiff1d(all_indices, selected_indices)
            if len(remaining_indices) > 0:
                additional = np.random.choice(
                    remaining_indices,
                    size=min(remaining_needed, len(remaining_indices)),
                    replace=False
                )
                selected_indices.extend(additional)
        
        # return selected patterns as centroids
        centroids = X_train[selected_indices[:self.num_rbf]]
        
        return centroids

    def _clustering(self, X_train: np.array, y_train: np.array) -> KMeans:
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
        if self.classification:
            # use stratified initialization for classification
            init_centroids = self._init_centroids_classification(X_train, y_train)
            kmeans = KMeans(
                n_clusters=self.num_rbf,
                init=init_centroids,
                n_init=1,
                max_iter=500,
                random_state=self.random_state
            )
        else:
            # use random initialization for regression
            kmeans = KMeans(
                n_clusters=self.num_rbf,
                init='random',
                n_init=1,
                max_iter=500,
                random_state=self.random_state
            )
        
        # fit kmeans to the training data
        kmeans.fit(X_train)
        
        return kmeans

    def _calculate_radii(self) -> np.array:
        """
        Obtain the value of the radii after clustering

        This method is used to heuristically obtain the radii of the RBFs
        based on the centers

        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
        """
        # get the cluster centers
        centers = self.kmeans.cluster_centers_
        
        # calculate pairwise distances between all centers
        dist_matrix = squareform(pdist(centers))
        
        # for each center j, calculate the average distance to all other centers
        # formula: σⱼ = ½ × (1/(n₁-1)) × Σᵢ≠ⱼ ||cⱼ - cᵢ||
        n_rbf = len(centers)
        radii = np.zeros(n_rbf)
        
        for j in range(n_rbf):
            # sum distances from center j to all other centers
            sum_distances = np.sum(dist_matrix[j, :]) - dist_matrix[j, j]
            # calculate radius as half of average distance
            if n_rbf > 1:
                radii[j] = 0.5 * (1.0 / (n_rbf - 1)) * sum_distances
            else:
                # edge case: only one rbf, use default radius
                radii[j] = 1.0
        
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
        # square the distances
        squared_distances = distances ** 2
        
        # calculate 2σ² for each RBF 
        two_sigma_squared = 2 * (self.radii ** 2)
        
        # avoid division by zero (if radius is 0, set to small value)
        two_sigma_squared = np.where(two_sigma_squared == 0, 1e-10, two_sigma_squared)
        
        # calculate Gaussian activations: exp(-d²/(2σ²))
        rbf_activations = np.exp(-squared_distances / two_sigma_squared)
        
        # add bias column (ones) at the end
        n_patterns = rbf_activations.shape[0]
        bias_column = np.ones((n_patterns, 1))
        r_matrix = np.hstack([rbf_activations, bias_column])
        
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
        # calculate moore-penrose pseudo-inverse
        r_pseudo_inv = np.linalg.pinv(r_matrix)
        
        # multiply pseudo-inverse by targets to get coefficients
        beta_transpose = r_pseudo_inv @ y_train
        
        # transpose to get coefficients in correct shape
        coefficients = beta_transpose.T
        
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
        # determine regularization type
        penalty = 'l2' if self.l2 else 'l1'
        
        # handle y_train format (1D or 2D, one-hot encoded or not)
        if len(self.y_train.shape) == 1:
            y_train_flat = self.y_train
        else:
            # convert one-hot encoded y to class labels
            y_train_flat = np.argmax(self.y_train, axis=1) if self.y_train.shape[1] > 1 else self.y_train.flatten()
        
        if self.logisticcv:
            # use logisticregressioncv with cross-validation
            Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
            
            logreg = LogisticRegressionCV(
                Cs=Cs,
                penalty=penalty,
                solver='saga',
                max_iter=10,
                cv=3,
                random_state=self.random_state,
                multi_class='multinomial' if len(np.unique(y_train_flat)) > 2 else 'auto'
            )
        else:
            # use regular logisticregression
            C = 1.0 / self.eta if self.eta != 0 else 1e10
            
            logreg = LogisticRegression(
                C=C,
                penalty=penalty,
                solver='saga',
                max_iter=10,
                random_state=self.random_state,
                multi_class='multinomial' if len(np.unique(y_train_flat)) > 2 else 'auto'
            )
        
        # train logistic regression on the r matrix
        logreg.fit(self.r_matrix, y_train_flat)
        
        return logreg