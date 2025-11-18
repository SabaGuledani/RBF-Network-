import os
import pickle
import numpy as np
import pandas as pd
import click
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate

from esqueleto_rbf import RBFNN


@click.command()
@click.option(
    "--dataset_filename",
    "-d",
    default=None,
    required=True,
    help="Name of the file with training data.",
    type=str,
)
# TODO: Capture the necessary parameters
@click.option(
    "--model_filename",
    "-m",
    default="",
    show_default=True,
    help="Directory name to save the models (or name of the file to load the model, if "
    "the prediction mode is active).",
    type=str,
)  # KAGGLE
@click.option(
    "--pred",
    "-p",
    default=None,
    show_default=True,
    help="Specifies the seed used to predict the output of the dataset_filename.",
    type=int,
)  # KAGGLE
@click.option(
    "--standarize",
    "-s",
    is_flag=True,
    default=False,
    help="Standarize input variables.",
)
@click.option(
    "--classification",
    "-c",
    is_flag=True,
    default=False,
    help="Use classification instead of regression.",
)
@click.option(
    "--ratio_rbf",
    "-r",
    default=0.1,
    show_default=True,
    help="Ratio of RBFs with respect to the total number of patterns.",
    type=float,
)
@click.option(
    "--l2",
    "-l",
    is_flag=True,
    default=False,
    help="Use L2 regularization for logistic regression.",
)
@click.option(
    "--eta",
    "-e",
    default=0.01,
    show_default=True,
    help="Value of the regularization factor for logistic regression.",
    type=float,
)
@click.option(
    "--fairness",
    "-f",
    is_flag=True,
    default=False,
    help="Calculate fairness metrics.",
)
@click.option(
    "--logisticcv",
    "-v",
    is_flag=True,
    default=False,
    help="Use LogisticRegressionCV.",
)
@click.option(
    "--seeds",
    "-n",
    default=5,
    show_default=True,
    help="Number of seeds to use.",
    type=int,
)
@click.option(
    "--cm_out_folder",
    "-cm",
    default=None,
    help="Name of the folder to save confusion matrices.",
    type=str,
)
def main(
    dataset_filename: str,
    standarize: bool,
    classification: bool,
    ratio_rbf: float,
    l2: bool,
    eta: float,
    fairness: bool,
    logisticcv: bool,
    seeds: int,
    model_filename: str,
    pred: int,
    cm_out_folder: str,
):
    """
    Run several executions of RBFNN training and testing.

    RBF neural network based on hybrid supervised/unsupervised training. Every run uses
    a different seed for the random number generator. The results of the training and
    testing are stored in a pandas DataFrame.

    Parameters
    ----------
    dataset_filename: str
        Name of the data file
    standarize: bool
        True if we want to standarize input variables (and output ones if
          it is regression)
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
    fairness: bool
        False. If set to true, it will calculate fairness metrics on the prediction
    logisticcv: bool
        True if we want to use LogisticRegressionCV
    seeds: int
        Number of seeds to use
    model_filename: str
        Name of the directory where the models will be written. Note that it will create
        a directory with the name of the dataset file and the seed number
    pred: int
        If used, it will predict the output of the dataset_filename using the model
        stored in model_filename with the seed indicated in this parameter
    cm_out_folder: str
        Name of the folder to save confusion matrices (optional)
    """

    # check that when logisticcv is set to True, eta is not included
    if logisticcv and eta != 0.01:
        raise ValueError("You cannot use eta when logisticcv is set to True.")

    # error validation checks
    # 1. check that dataset file exists
    if not os.path.exists(dataset_filename):
        raise ValueError(f"The dataset file {dataset_filename} does not exist.")

    # 2. check that ratio_rbf is valid (between 0 and 1)
    if ratio_rbf <= 0 or ratio_rbf > 1:
        raise ValueError(f"ratio_rbf must be between 0 and 1, got {ratio_rbf}.")

    # 3. check that eta is positive
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}.")

    # 4. check that seeds is positive
    if seeds <= 0:
        raise ValueError(f"seeds must be positive, got {seeds}.")

    # 5. check that fairness is only used with Triage dataset
    dataset_name = dataset_filename.split("/")[-1].split(".")[0]
    if fairness and dataset_name != "triage":
        raise ValueError(
            f"You can only calculate fairness metrics when using the triage dataset. "
            f"Current dataset: {dataset_name}."
        )

    # 6. check that fairness requires classification
    if fairness and not classification:
        raise ValueError("Fairness metrics can only be calculated for classification problems.")

    # 7. check that if pred is used, model_filename must be provided
    if pred is not None and not model_filename:
        raise ValueError("You have not specified the model directory (-m).")

    # 8. check that if pred is used, model file must exist
    if pred is not None and model_filename:
        model_file_path = f"{model_filename}/{dataset_name}/{pred}.p"
        if not os.path.exists(model_file_path):
            raise ValueError(
                f"The model file {model_file_path} does not exist.\n"
                f"You can create it by firstly using the parameter (n={pred}) and "
                f"removing the flag -p (for pred) to train the model."
            )

    # 9. check that if pred is used, Kaggle dataset file must exist
    if pred is not None:
        kaggle_dataset_path = dataset_filename.replace(".csv", "_kaggle.csv")
        if not os.path.exists(kaggle_dataset_path):
            raise ValueError(
                f"The Kaggle dataset file {kaggle_dataset_path} does not exist. "
                f"It should be named as {os.path.basename(kaggle_dataset_path)}."
            )

    # 10. check that logisticcv is only used for classification
    if logisticcv and not classification:
        raise ValueError("LogisticRegressionCV can only be used for classification problems.")

    results = []

    seeds_list = range(seeds)
    dataset_name = dataset_filename.split("/")[-1].split(".")[0]

    if pred is not None:
        seeds_list = [pred]

    for random_state in seeds_list:
        print(f"Running on {dataset_name} - seed: {random_state}.")
        np.random.seed(random_state)

        # read the data
        data = read_data(
            dataset_filename=dataset_filename,
            standarize=standarize,
            random_state=random_state,
            classification=classification,
            fairness=fairness,
            prediction_mode=(pred is not None)
        )

        if not fairness and pred is None:
            X_train, y_train, X_test, y_test = data
        elif pred is not None:
            X_train, y_train, X_test, y_test, X_test_kaggle = data
        elif fairness:
            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_train_disc,
                X_test_disc,
            ) = data

        if pred is None:  # Train the model
            # create rbfnn object
            rbf = RBFNN(
                classification=classification,
                ratio_rbf=ratio_rbf,
                l2=l2,
                eta=eta,
                logisticcv=logisticcv,
                random_state=random_state
            )
            
            # train the model
            rbf.fit(X_train, y_train)

            if model_filename:
                dir_name = f"{model_filename}/{dataset_name}/{random_state}.p"
                save(rbf, dir_name)
                print(f"Model saved in {dir_name}")

        else:  # Load the model from file
            dir_name = f"{model_filename}/{dataset_name}/{random_state}.p"
            rbf = load(dir_name, random_state)

        # predict the output using the trained model
        preds_train = rbf.predict(X_train)
        preds_test = rbf.predict(X_test)

        # handle prediction shapes for metrics
        # ensure y is in correct format for metrics
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train_flat = y_train.flatten()
        else:
            y_train_flat = y_train
            
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test_flat = y_test.flatten()
        else:
            y_test_flat = y_test
        
        # ensure predictions are in correct format
        if len(preds_train.shape) > 1:
            if preds_train.shape[1] == 1:
                preds_train_flat = preds_train.flatten()
            else:
                # multi-class: use argmax if needed
                preds_train_flat = preds_train.flatten() if classification else preds_train
        else:
            preds_train_flat = preds_train
            
        if len(preds_test.shape) > 1:
            if preds_test.shape[1] == 1:
                preds_test_flat = preds_test.flatten()
            else:
                # multi-class: use argmax if needed
                preds_test_flat = preds_test.flatten() if classification else preds_test
        else:
            preds_test_flat = preds_test

        if pred is not None:
            preds_kaggle = rbf.predict(X_test_kaggle)
            # flatten kaggle predictions if needed
            if len(preds_kaggle.shape) > 1 and preds_kaggle.shape[1] == 1:
                preds_kaggle = preds_kaggle.flatten()
            elif len(preds_kaggle.shape) > 1:
                preds_kaggle = preds_kaggle.flatten()
            
            dir_name = f"{model_filename}/{dataset_name}/predictions_{pred}.csv"
            # include index in the first column from 1 to length preds_kaggle
            preds_kaggle_with_index = np.column_stack(
                (np.arange(1, len(preds_kaggle)+1), preds_kaggle)
            )
            np.savetxt(
                dir_name,
                preds_kaggle_with_index,
                delimiter=",",
                header="ID,survived",
                comments="",
                fmt="%d",
            )
            print(f"Predictions saved in {dir_name}.")

        train_results_per_seed = {
            "seed": random_state,
            "partition": "Train",
            "MSE": mean_squared_error(y_train_flat, preds_train_flat),
        }
        test_results_per_seed = {
            "seed": random_state,
            "partition": "Test",
            "MSE": mean_squared_error(y_test_flat, preds_test_flat),
        }

        if classification:
            train_results_per_seed["CCR"] = accuracy_score(y_train_flat, preds_train_flat) * 100
            test_results_per_seed["CCR"] = accuracy_score(y_test_flat, preds_test_flat) * 100

        # fairness evaluation
        if fairness:
            metrics = {
                "false negative rate": false_negative_rate,
                "false positive rate": false_positive_rate,
            }

            train_mf = MetricFrame(
                metrics=metrics,
                y_true=y_train_flat,
                y_pred=preds_train_flat,
                sensitive_features=X_train_disc,
            )
            test_mf = MetricFrame(
                metrics=metrics,
                y_true=y_test_flat,
                y_pred=preds_test_flat,
                sensitive_features=X_test_disc,
            )

            train_results_per_seed["FN0"] = (
                train_mf.by_group.loc["Men"]["false negative rate"] * 100
            )
            train_results_per_seed["FN1"] = (
                train_mf.by_group.loc["Women"]["false negative rate"] * 100
            )
            test_results_per_seed["FN0"] = (
                test_mf.by_group.loc["Men"]["false negative rate"] * 100
            )
            test_results_per_seed["FN1"] = (
                test_mf.by_group.loc["Women"]["false negative rate"] * 100
            )

            train_results_per_seed["FP0"] = (
                train_mf.by_group.loc["Men"]["false positive rate"] * 100
            )
            train_results_per_seed["FP1"] = (
                train_mf.by_group.loc["Women"]["false positive rate"] * 100
            )
            test_results_per_seed["FP0"] = (
                test_mf.by_group.loc["Men"]["false positive rate"] * 100
            )
            test_results_per_seed["FP1"] = (
                test_mf.by_group.loc["Women"]["false positive rate"] * 100
            )

        results.append(train_results_per_seed)
        results.append(test_results_per_seed)

    results = pd.DataFrame(results)
    if pred is None:
        metrics = results.columns[2:]

        mean_std = []

        # TODO: Calculate the mean and standard deviation of the metrics and add them to the
        #  results DataFrame

    results.set_index(["seed", "partition"], inplace=True)

    print("******************")
    print("Summary of results")
    print("******************")

    print(results)


def save(
    model: RBFNN,
    dir_name: str,
) -> None:
    """
    Save the model to a file

    Parameters
    ----------
    model: RBFNN
        Model to be saved
    dir_name: str
        Name of the file where the model will be saved
    """

    dir = os.path.dirname(dir_name)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    with open(dir_name, "wb") as f:
        pickle.dump(model, f)


def load(dir_name: str, random_state: int) -> RBFNN:
    """
    Load the model from the file

    Parameters
    ----------
    dir_name: str
        Name of the model file
    random_state: int
        Seed for the random number generator

    Returns
    -------
    model: RBFNN
        Model loaded from the file
    """

    if not os.path.exists(dir_name):
        raise ValueError(
            f"The model file {dir_name} does not exist.\n"
            f"You can create it by firstly using the parameter (n = {random_state}) and"
            f" removing the flag P (for pred) to train the model."
        )
    with open(dir_name, "rb") as f:
        self = pickle.load(f)
    return self


def read_data(
    dataset_filename: str,
    standarize: bool,
    random_state: int,
    classification: bool = False,
    fairness: bool = False,
    prediction_mode: bool = False,
) -> (
    tuple[np.array, np.array, np.array, np.array]
    | tuple[np.array, np.array, np.array, np.array, np.array, np.array]
    | tuple[np.array, np.array, np.array, np.array, np.array]
):
    """
    Read the input data

    It receives the name of the dataset file and returns the corresponding matrices.

    Parameters
    ----------
    dataset_filename: str
        Name of the dataset file
    standarize: bool
        True if we want to standarize input variables (and output ones if
          it is regression)
    random_state: int
        Seed for the random number generator
    classification: bool
        True if it is a classification problem
    fairness: bool
        True if we want to calculate fairness metrics. The discriminative attribute
        is assumed to be the last column of the input data.
    prediction_mode: bool
        True if we are in prediction mode. This is to load the data for Kaggle.

    Returns
    -------
    X_train: array, shape (n_train_patterns,n_inputs)
        Matrix containing the inputs for the training patterns
    y_train: array, shape (n_train_patterns,n_outputs)
        Matrix containing the outputs for the training patterns
    X_test: array, shape (n_test_patterns,n_inputs)
        Matrix containing the inputs for the test patterns
    y_test: array, shape (n_test_patterns,n_outputs)
        Matrix containing the outputs for the test patterns
    X_test_kaggle: array, shape (n_test_patterns_kaggle,n_inputs)
        Matrix containing the inputs for the test patterns (only if prediction_mode is
        set to True)
    X_train_disc: array, shape (n_train_patterns,)
        Array containing the discriminative attribute for the training patterns (only
        if fairness is set to True)
    X_test_disc: array, shape (n_test_patterns,)
        Array containing the discriminative attribute for the testing patterns (only
        if fairness is set to True)
    """
    # read csv file using pandas (no headers)
    data = pd.read_csv(dataset_filename, header=None)
    data = data.values  # convert to numpy array
    
    # split into X (all columns except last) and y (last column)
    X_all = data[:, :-1]  # all input features
    y = data[:, -1]  # output
    
    # handle fairness mode: gender is in the last column of input variables
    if fairness:
        # gender is the last column of input features
        X_gender = X_all[:, -1].copy()
        X = X_all[:, :-1]  # remove gender from X for training
    else:
        X = X_all
    
    # split into train and test sets
    # test_size=0.25, shuffle=True, stratified for classification
    stratify_param = y if classification else None
    
    # for fairness mode, include gender in the split to get matching indices
    if fairness:
        # split X and X_gender together to ensure matching indices
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test = train_test_split(
            X, y, X_gender,
            test_size=0.25,
            shuffle=True,
            stratify=stratify_param,
            random_state=random_state
        )
        # convert to "Men"/"Women" labels (0 = Men, 1 = Women)
        X_train_disc = np.where(X_gender_train == 1, "Women", "Men")
        X_test_disc = np.where(X_gender_test == 1, "Women", "Men")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.25,
            shuffle=True,
            stratify=stratify_param,
            random_state=random_state
        )
    
    # standardize if requested
    scaler_X = None
    scaler_y = None
    if standarize:
        # for regression: standardize both X and y
        # for classification: standardize only X
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        if not classification:
            # standardize y for regression
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # convert to numpy arrays and ensure proper shapes
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # ensure y is 2D if needed (for multi-output)
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    
    # handle prediction mode (kaggle dataset)
    if prediction_mode:
        kaggle_dataset_path = dataset_filename.replace(".csv", "_kaggle.csv")
        data_kaggle = pd.read_csv(kaggle_dataset_path, header=None)
        data_kaggle = data_kaggle.values
        
        # kaggle dataset has same structure as training X (all input features)
        X_test_kaggle = data_kaggle
        
        # if fairness mode, remove gender column from kaggle data
        if fairness:
            X_test_kaggle = X_test_kaggle[:, :-1]  # remove last column (gender)
        
        # standardize kaggle data if needed
        if standarize and scaler_X is not None:
            X_test_kaggle = scaler_X.transform(X_test_kaggle)
        
        X_test_kaggle = np.array(X_test_kaggle)
    
    # return appropriate tuple based on mode
    if fairness:
        return X_train, y_train, X_test, y_test, X_train_disc, X_test_disc
    
    if prediction_mode:
        return X_train, y_train, X_test, y_test, X_test_kaggle
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
