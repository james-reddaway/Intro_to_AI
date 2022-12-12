# import my_funcs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def read_csv_dataset(file_path, scale, hold_out_size):
    '''
        Use the specified csv filepath to extract and sort data to data and target.
    Code used from: (import csv) https://code-examples.net/en/q/a83433
                    (data scaling) https://datagy.io/pandas-normalize-column/
                    (One-Hot encoding) https://www.statology.org/one-hot-encoding-in-python/

    :param file_path: csv filepath
    :param scale: normalise the data when True
    :param hold_out_size: define the split of data to be reserved as a hold out set
    :return: training data, training data labels, hold out data, hold out data targets
    '''

    target_label = 'Rented Bike Count'
    # read the csv file
    raw_data = pd.read_csv(file_path, header=0, encoding='unicode_escape')

    # drop dates from data
    raw_data = raw_data.drop(['Date'], axis=1)

    # one hot encode the string data
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(raw_data[['Seasons', 'Holiday', 'Functioning_Day']]).toarray())
    raw_data = raw_data.join(encoder_df)
    raw_data.drop(['Seasons', 'Holiday', 'Functioning_Day'], axis=1, inplace=True)

    # scale the columns in the pandas frame
    if scale:
        scaler = MinMaxScaler()
        scaler.fit(raw_data)
        scaled = scaler.fit_transform(raw_data)
        raw_data = pd.DataFrame(scaled, columns=raw_data.columns)

    # create a hold out set to test with later
    # set for constant/random samples in set
    random.seed(1)
    # generate random indexes for hold out set
    hold_out_index = random.sample(range(len(raw_data)), round(hold_out_size * len(raw_data)))
    # extract indexes
    hold_out_set = raw_data.iloc[hold_out_index]
    # remove copied indexes from original set
    raw_data.drop(hold_out_index, axis=0, inplace=True)

    # drop target label from the set for data and hold out set
    data = raw_data.drop([target_label], axis=1)
    hold_out_data = hold_out_set.drop([target_label], axis=1)

    # split into to targets and data
    targets = raw_data[target_label]
    hold_out_targets = hold_out_set[target_label]

    # convert to numpy arrays
    targets = targets.to_numpy()
    data = data.to_numpy()
    hold_out_data = hold_out_data.to_numpy()
    hold_out_targets = hold_out_targets.to_numpy()

    return data, targets, hold_out_data, hold_out_targets


def decision_tree_baseline(X_train, y_train):
    '''
    Generates the decision tree model to be used as the baseline

    :param X_train: training data
    :param y_train: training data targets
    :return: baseline model
    '''

    dt_model = tree.DecisionTreeRegressor()
    baseline_model = dt_model.fit(X_train, y_train)

    return baseline_model


def knn(X_train, y_train):
    '''
    Train the KNN regressor model on the specified data. The applied parameters are generated from the GA optimiser.

    :param X_train: training data
    :param y_train: training data targets
    :return: KNN regressor model
    '''

    knn_model = KNeighborsRegressor(n_neighbors=7, weights='distance', algorithm='brute', leaf_size=23, p=1)
    knn_model.fit(X_train, y_train)

    return knn_model


def mlp(X_train, y_train):
    '''
    Train the KNN regressor model on the specified data. The applied parameters are generated from the GA optimiser.

    :param X_train: training data
    :param y_train: training data targets
    :return: MLP regressor model
    '''

    mlp_model = MLPRegressor(hidden_layer_sizes=(20, 20, 20, 20, 20), activation='relu', solver='lbfgs',
                             alpha=0.00013743216419409218, batch_size=405, max_iter=5000, tol=0.00013433035342599393,
                             warm_start=True)
    mlp_model.fit(X_train, y_train)

    return mlp_model


def param_test(X_train, X_test, y_train, y_test, iterations, avg_runs):
    '''
    Used to evaluate the impact of varying parameters and generate data for plots eg the impact of varying k for the
    K-NN model

    :param X_train: training data
    :param X_test: test data
    :param y_train: training data targets
    :param y_test: testing data targets
    :param iterations: iterations to apply for the test parameter
    :param avg_runs: number of runs used to average the data
    '''

    train_data = []
    test_data = []
    iter = range(5, iterations,2)

    # iterate for each tested i
    for i in iter:
        train_temp = []
        test_temp = []
        # iterate to average samples
        for averaging in range(avg_runs):
            # build models and return R^2 error for iter
            # knn_model = KNeighborsRegressor(n_neighbors=i, weights='distance', algorithm='brute', leaf_size=23, p=1)
            # model = knn_model.fit(X_train, y_train)

            mlp_model= MLPRegressor(hidden_layer_sizes=(i, i, i, i, i), activation='relu', solver='lbfgs',
                                    alpha=0.00013743216419409218, batch_size=405, max_iter=5000,
                                    tol=0.00013433035342599393, warm_start=True)
            model = mlp_model.fit(X_train, y_train)

            preds_trn = model.predict(X_train)
            preds_test = model.predict(X_test)

            r2_train_score = r2_score(preds_trn, y_train)
            r2_test_score = r2_score(preds_test, y_test)

            print(r2_train_score)

            train_temp.append(r2_train_score)
            test_temp.append(r2_test_score)
        train_data.append(train_temp)
        test_data.append(test_temp)

    # convert list to nump arrays
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    # mean of test data
    train_data = np.mean(train_data, axis=1)
    test_data = np.mean(test_data, axis=1)

    # plot average for test and train
    plt.plot(iter, train_data, color='k', label='Train')
    plt.plot(iter, test_data, color='b', label='Test')

    print(train_data)
    print(test_data)

    # plot line at best test error
    plt.axvline(x=np.argmax(test_data)+1, color='r', label='Max test score')
    print('Best test error:  ')
    print(max(test_data))

    # format plot
    plt.ylabel('R^2 score')
    plt.xlabel('K')
    plt.legend()
    plt.show()


def knn_genetic_optimisation(X_train, X_test, y_train, y_test):
    '''
    Runs a genetic optimiser to search for parameters for the K-NN regressor
    Code used form: https://towardsdatascience.com/tune-your-scikit-learn-model-using-evolutionary-algorithms-30538248ac16

    :param X_train: training data
    :param X_test: test data
    :param y_train: train data targets
    :param y_test: test data targets
    :return: history of generations
    '''

    # the parameters and ranges/ options to be used in the search
    param_grid = {'n_neighbors': Integer(1, 50),
                  'weights': Categorical(['uniform', 'distance']),
                  'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
                  'leaf_size': Integer(5, 100),
                  'p': Integer(1, 2),
                  }
    # set the model to use in the search
    clf = KNeighborsRegressor()

    # set search parameters
    evolved_estimator = GASearchCV(estimator=clf,
                                   scoring='r2',
                                   population_size=10,
                                   generations=5,
                                   tournament_size=3,
                                   elitism=True,
                                   crossover_probability=0.8,
                                   mutation_probability=0.2,
                                   param_grid=param_grid,
                                   criteria='max',
                                   algorithm='eaMuPlusLambda',
                                   n_jobs=-1,
                                   verbose=True,
                                   keep_top_k=4)
    # run the search
    evolved_estimator.fit(X_train, y_train)

    # make knn_pred the evolved estimator
    y_predicy_ga_test = evolved_estimator.predict(X_test)
    y_predicy_ga_trn = evolved_estimator.predict(X_train)

    # evaluate the generations
    score_test = r2_score(y_test, y_predicy_ga_test)
    score_trn = r2_score(y_train, y_predicy_ga_trn)

    # plot fitness history
    plot_fitness_evolution(evolved_estimator)
    # plt.show()

    # print
    print('Testing score:  ')
    print(score_test)
    print('Training score:  ')
    print(score_trn)
    print(evolved_estimator.best_params_)

    return evolved_estimator.history


def mlp_genetic_optimisation(X_train, X_test, y_train, y_test):
    '''
    Runs a genetic optimiser to search for parameters for the MLP regressor
    Code used form: https://towardsdatascience.com/tune-your-scikit-learn-model-using-evolutionary-algorithms-30538248ac16

    :param X_train: training data
    :param X_test: test data
    :param y_train: train data targets
    :param y_test: test data targets
    :return: history of generations
    '''

    param_grid = {'hidden_layer_sizes': Categorical([(10, 10), (10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10, 10),
                                                     (20, 20), (20, 20, 20), (20, 20, 20, 20), (20, 20, 20, 20, 20),
                                                     (30, 30), (30, 30, 30), (30, 30, 30, 30), (30, 30, 30, 30, 30)]),
                  'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
                  'solver': Categorical(['lbfgs', 'sgd', 'adam']),
                  'alpha': Continuous(0.0001, 2, distribution='log-uniform'),
                  'batch_size': Integer(1, 500),
                  'max_iter': Integer(4999, 5000),
                  'tol': Continuous(0.0001, 2, distribution='log-uniform'),
                  'warm_start': Categorical([True])
                  }

    clf = MLPRegressor()
    evolved_estimator = GASearchCV(estimator=clf,
                                   scoring='r2',
                                   population_size=5,
                                   generations=20,  # 10
                                   tournament_size=3,
                                   elitism=True,
                                   crossover_probability=0.5,
                                   mutation_probability=0.5,
                                   param_grid=param_grid,
                                   criteria='max',
                                   n_jobs=-1,
                                   verbose=True,
                                   keep_top_k=4
                                   )

    evolved_estimator.fit(X_train, y_train)
    y_predicy_ga = evolved_estimator.predict(X_test)
    score = r2_score(y_test, y_predicy_ga)

    plot_fitness_evolution(evolved_estimator)
    # plt.show()

    print(score)
    print(evolved_estimator.best_params_)

    return evolved_estimator.history


def ga_averaging(X_train, X_test, y_train, y_test, avg_runs, alg):
    '''
    Used to set multiple runs of the GA

    :param X_train: train data
    :param X_test: test data
    :param y_train: train data targets
    :param y_test: test data targets
    :param avg_runs: number of runs to average by
    :param alg: algorithm to evaluate
    :return: mean data
    '''

    # define ranges and list to store maximum fitness
    avg_range = range(0, avg_runs)
    max_fits = []

    # set the algorithum to use
    for i in avg_range:
        if alg == 'knn':
            ga_history = knn_genetic_optimisation(X_train, X_test, y_train, y_test)
        elif alg == 'mlp':
            ga_history = mlp_genetic_optimisation(X_train, X_test, y_train, y_test)
        temp = list(ga_history.values())
        max_fits.append(temp[1])

    # convert list to nump arrays
    max_fits = np.asarray(max_fits)

    # mean of test data
    mean_data = np.mean(max_fits, axis=0)

    return mean_data


def evaluate_model(model, test_data, test_targets, label=None):
    '''
    Function to find the R^2 score for the passed model

    :param model: model passed for evaluation
    :param test_data: data to be used for evaluation
    :param test_targets: test data targets
    :param label: verbose with print label
    :return: R^2 label, knn_pred
    '''

    # make knn_pred
    preds = model.predict(test_data)

    # find score
    R2 = round(metrics.r2_score(test_targets, preds), 5)

    # print score and label
    if label != None:
        print(str(label) + ' R^2 score:  ' + str(R2))

    return R2, preds


if __name__ == '__main__':
    '''
    main: Tests the performance of both the algorithms and the baseline over an average of runs. 
    '''
    # define number of runs to average by and the number fof models to evaluate
    average_runs = 1
    num_models = 3

    # define variables to store data
    train_avg = []
    test_avg = []
    hold_avg = []

    train_fullset = [[] for _ in range(0, num_models)]
    test_fullset = [[] for _ in range(0, num_models)]
    hold_fullset = [[] for _ in range(0, num_models)]

    # read the csv file for the dataset
    X, y, X_hold, Y_hold = read_csv_dataset(file_path='coursework_other.csv', scale=True, hold_out_size=0.1)
    Xtr, Xtest, Ytr, Ytest = train_test_split(X, y, test_size=0.2, random_state=None)

    for avg in range(0, average_runs):
        # shuffle the data split for each average run
        Xtr, Xtest, Ytr, Ytest = train_test_split(X, y, test_size=0.2, random_state=None)
        # generate k folds splits for current data shuffle
        kf = KFold(n_splits=5, random_state=None, shuffle=True)

        # define variables to store scores for each data series
        train_score = [[] for _ in range(0, num_models)]
        test_score = [[] for _ in range(0, num_models)]
        hold_score = [[] for _ in range(0, num_models)]

        # for each model
        for model in range(0, num_models):
            # for each split
            for train_index, val_index in kf.split(Xtr):
                # split the data for the current k fold split
                X_train, X_test = Xtr[train_index], Xtr[val_index]
                y_train, y_test = Ytr[train_index], Ytr[val_index]

                # select the model to evaluate on the current data shuffle
                if model == 0:
                    reg = decision_tree_baseline(X_train, y_train)
                elif model == 1:
                    reg = knn(X_train, y_train)
                else:
                    reg = mlp(X_train, y_train)

                # generate knn_pred for all the datasets
                pred_train = reg.predict(X_train)
                pred_val = reg.predict(X_test)
                pred_hold = reg.predict(X_hold)

                # evaluate knn_pred with R^2 score
                R2_train = r2_score(y_train, pred_train)
                R2_test = r2_score(y_test, pred_val)
                R2_hold = r2_score(Y_hold, pred_hold)

                # store for means
                train_score[model].append(R2_train)
                test_score[model].append(R2_test)
                hold_score[model].append(R2_hold)

                # store for standard deviation
                train_fullset[model].append(R2_train)
                test_fullset[model].append(R2_test)
                hold_fullset[model].append(R2_hold)

            # calculate the mean for all folds of each model
            train_score[model] = np.mean(train_score[model], axis=0)
            test_score[model] = np.mean(test_score[model], axis=0)
            hold_score[model] = np.mean(hold_score[model], axis=0)

        print('Completed loop: ' + str(avg + 1) + '/' + str(average_runs))

        # store data for the full run on all models
        train_avg.append(train_score)
        test_avg.append(test_score)
        hold_avg.append(hold_score)

    # compile test data for each model
    baseline_data = [train_score[0], test_score[0], hold_score[0]]
    knn_data = [train_score[1], test_score[1], hold_score[1]]
    mlp_data = [train_score[2], test_score[2], hold_score[2]]

    # compiled standard deviations
    baseline_stdev = [np.std(train_fullset[:][0]), np.std(test_fullset[:][0]), np.std(hold_fullset[:][0])]
    knn_stdev = [np.std(train_fullset[:][1]), np.std(test_fullset[:][1]), np.std(hold_fullset[:][1])]
    mlp_stdev = [np.std(train_fullset[:][2]), np.std(test_fullset[:][2]), np.std(hold_fullset[:][2])]

    # print data
    print(' ')
    print('--------------------------------')
    print(' ')
    print('Baseline means: ')
    print(baseline_data)
    print('K-NN means: ')
    print(knn_data)
    print('MLP means: ')
    print(mlp_data)
    print(' ')
    print('--------------------------------')
    print(' ')
    print('Baseline stdev: ')
    print(baseline_stdev)
    print('K-NN stdev: ')
    print(knn_stdev)
    print('MLP stdev: ')
    print(mlp_stdev)



