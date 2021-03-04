
import sys
import numpy as np
import statistics

from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from warnings import simplefilter
from lsanomaly import LSAnomaly
from sklearn.manifold import TSNE
from sklearn.exceptions import DataConversionWarning

# remove all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DataConversionWarning)

from HYBRID_METHOD._function_base import get_data, do_resampling, create_adabost_model, create_random_forest_model, \
    create_svc_model, create_knn_model

# setup path for Spyder
# import sys
# sys.path.insert(0, 'D:/pGnip/diplomovka_predikcia/HYBRID_METHOD')
# from _function_base import get_data, create_svc_model, do_resampling, create_knn_model, create_random_forest_model
# from _function_base import create_adabost_model, create_rusboost_model


# constants
ITER_COUNT = 50
CLF_OC_SVM = OneClassSVM(gamma=0.01, nu=0.35)
CLF_LSAD = LSAnomaly(sigma=3, rho=3)
K_FOLD = KFold(n_splits=5, random_state=42)
STANDARD_SCALER = StandardScaler()

LSAD_RHO_PARAMS_RANGE = [0.01, 0.1, 0.5, 1, 5, 10]
LSAD_SIGMA_PARAMS_RANGE = [0.1, 0.3, 0.5, 0.7, 1, 5, 10]

OCSVM_GAMMA_PARAMS_RANGE = [0.001, 0.01, 0.1, 1, 5, 10, 100]
OCSVM_NU_PARAMS_RANGE = [0.001, 0.01, 0.1, 0.3, 0.5, 0.9]


# generate graph using data with particular labels
def generate_graph_via_tsne(data, labels, name):
    # scale data
    standard_scaler_instance = StandardScaler()
    bankrupt_and_non_bankrupt_data = standard_scaler_instance.fit_transform(data)

    tsne = TSNE(n_components=2, init="pca", perplexity=30.0, random_state=42)
    y = tsne.fit_transform(bankrupt_and_non_bankrupt_data)
    yyy = np.ravel(labels)
    # plotting graph
    being_saved = pyplot.figure()
    pyplot.scatter(y[yyy == 1, 0], y[yyy == 1, 1], color="blue", marker="+")
    pyplot.scatter(y[yyy == 0, 0], y[yyy == 0, 1], color="red", marker="o")
    pyplot.tight_layout()
    pyplot.ylim(-100, 100)
    pyplot.xlim(-100, 100)
    being_saved.savefig('./' + name + '.eps', format='eps', dpi=1000)
    pyplot.show()


def prepare_data(ratio, random_state):
    generated_data, generated_labels = get_data(ratio, random_state)
    return generated_data, generated_labels


# sample ratio ['5:1', '39:1', '50:1']
# sampling_strategy ['ada_syn', 'smote', 'tomek', 'smote_enn', 'ros']
# base_classifier ['SVC', 'RF', 'AB, 'KNN', 'RUSB']
def soa_with_ocsvm(samples_ratio, sampling_strategy, base_classifier):
    # average iteration result
    best_iteration_average_result = []

    for iteration in range(0, ITER_COUNT):
        print('...............................................................................................')
        print('HM, OCSVM, {}, sampling strategy - {}, samples ratio - {}'
              .format(base_classifier, sampling_strategy, samples_ratio))
        print('Current iteration number: {}/{}'.format(iteration + 1, ITER_COUNT))
        print('...............................................................................................')
        # generate data
        generated_data, generated_labels = get_data(samples_ratio, iteration)

        # tmp variable to store best scores from CV
        best_fold_geometric_mean_score = []

        # our hybrid method starts from here
        for train_index, test_index in K_FOLD.split(generated_data):
            x_train, x_test = generated_data[train_index], generated_data[test_index]
            y_train, y_test = generated_labels[train_index], generated_labels[test_index]

            x_train = STANDARD_SCALER.fit_transform(x_train)
            x_test = STANDARD_SCALER.transform(x_test)

            x_train_major = []
            x_train_minor = []
            y_train_major = []
            y_train_minor = []

            # major samples (0), minor samples (1)
            for data, label in zip(x_train, y_train):
                if label == 1:
                    x_train_minor.append(data)
                    y_train_minor.append(label)
                else:
                    x_train_major.append(data)
                    y_train_major.append(label)

            tmp_geometric_mean_scores = []
            for gamma in OCSVM_GAMMA_PARAMS_RANGE:
                for nu in OCSVM_NU_PARAMS_RANGE:
                    print("OCSVM clf current params: nu - {}, gamma - {}".format(nu, gamma))
                    ocsvm_clf = OneClassSVM(gamma=gamma, nu=nu)
                    ocsvm_clf.fit(x_train_major, y_train_major)
                    y_ocsvm_predicted = ocsvm_clf.predict(x_train_minor)
                    y_ocsvm_predicted_transformed = np.ones(len(y_ocsvm_predicted), dtype=int)
                    index = 0
                    for item in y_ocsvm_predicted:
                        if item == 1:
                            y_ocsvm_predicted_transformed[index] = 0
                        index = index + 1

                    # filter correctly predicted minority samples
                    x_correctly_predicted_train_minor = []
                    y_correctly_predicted_train_minor = []
                    for i in range(len(y_ocsvm_predicted_transformed)):
                        if y_ocsvm_predicted_transformed[i] == y_train_minor[i]:
                            x_correctly_predicted_train_minor.append(x_train_minor[i])
                            y_correctly_predicted_train_minor.append(y_train_minor[i])

                    if len(x_correctly_predicted_train_minor) > 5:
                        # join correctly predicted minor samples with major samples
                        x_joined_train = x_correctly_predicted_train_minor + x_train_major
                        y_join_train = y_correctly_predicted_train_minor + y_train_major

                        # resample train samples
                        x_resampled, y_resampled = do_resampling(sampling_strategy, x_joined_train, y_join_train)
                        # use proper base classifier
                        if base_classifier == 'SVC':
                            create_svc_model(x_resampled, y_resampled, x_test, y_test, tmp_geometric_mean_scores)
                        elif base_classifier == 'RF':
                            create_random_forest_model(x_resampled, y_resampled, x_test, y_test,
                                                       tmp_geometric_mean_scores)
                        elif base_classifier == 'KNN':
                            create_knn_model(x_resampled, y_resampled, x_test, y_test, tmp_geometric_mean_scores)
                        elif base_classifier == 'AB':
                            create_adabost_model(x_resampled, y_resampled, x_test, y_test, tmp_geometric_mean_scores)
                        else:
                            raise Exception("Unexpected 'base_classifier' input value!")
                    else:
                        print("Skipping, not enough correctly predicted minority samples ")

                # calculate best fold value
                tmp_best_score = np.amax(tmp_geometric_mean_scores)
                print('...............................................................................................')
                print("Current split best score is: {}".format(tmp_best_score))
                print('...............................................................................................')
                best_fold_geometric_mean_score.append(tmp_best_score)

        # calculate best iteration result
        tmp_average_score = statistics.mean(best_fold_geometric_mean_score)
        tmp_std = statistics.stdev(best_fold_geometric_mean_score)
        print('...............................................................................................')
        print('HM, OCSVM, {}, sampling strategy - {}, samples ratio - {}'
              .format(base_classifier, sampling_strategy, samples_ratio))
        print("Best iteration score is {} with std {}".format(tmp_average_score, tmp_std))
        print('...............................................................................................')
        best_iteration_average_result.append(tmp_average_score)

    # calculate final result
    final_score = statistics.mean(best_iteration_average_result)
    final_std = statistics.stdev(best_iteration_average_result)
    print('...............................................................................................')
    print('HM, OCSVM, {}, sampling strategy - {}, samples ratio - {}'
          .format(base_classifier, sampling_strategy, samples_ratio))
    print("Final G_MEAN score is: {}, std: {}".format(final_score, final_std))
    print('...............................................................................................')


# sample ratio ['5:1', '39:1', '50:1']
# sampling_strategy ['ada_syn', 'smote', 'tomek', 'smote_enn', 'ros']
# base_classifier ['SVC', 'RF', 'AB, 'KNN', 'RUSB']
def soa_with_ocsvm_and_real_data(bankrupt, non_bankrupt, sampling_strategy, base_classifier):
    # average iteration result
    best_iteration_average_result = []

    for iteration in range(0, ITER_COUNT):
        print('...............................................................................................')
        print('HM, OCSVM, {}, sampling strategy - {}'.format(base_classifier, sampling_strategy))
        print('Current iteration number: {}/{}'.format(iteration + 1, ITER_COUNT))
        print('...............................................................................................')
        # generate data
        # removed constant value of random_state
        k_fold = KFold(n_splits=5, shuffle=True)
        import pandas as pd
        generated_data = np.concatenate((non_bankrupt, bankrupt), axis=0)
        generated_labels = np.concatenate((pd.DataFrame(np.ones((non_bankrupt.shape[0], 1))),
                                           pd.DataFrame(np.zeros((bankrupt.shape[0], 1)))))

        # tmp variable to store best scores from CV
        best_fold_geometric_mean_score = []

        # our hybrid method starts from here
        for train_index, test_index in k_fold.split(generated_data,):
            x_train, x_test = generated_data[train_index], generated_data[test_index]
            y_train, y_test = generated_labels[train_index], generated_labels[test_index]

            from sklearn.impute import SimpleImputer
            si = SimpleImputer()
            x_train = si.fit_transform(x_train)
            x_test = si.transform(x_test)

            x_train = STANDARD_SCALER.fit_transform(x_train)
            x_test = STANDARD_SCALER.transform(x_test)

            x_train_major = []
            x_train_minor = []
            y_train_major = []
            y_train_minor = []

            # major samples (0), minor samples (1)
            for data, label in zip(x_train, y_train):
                if label == 1:
                    x_train_minor.append(data)
                    y_train_minor.append(label)
                else:
                    x_train_major.append(data)
                    y_train_major.append(label)

            tmp_geometric_mean_scores = []
            for gamma in OCSVM_GAMMA_PARAMS_RANGE:
                for nu in OCSVM_NU_PARAMS_RANGE:
                    print("OCSVM clf current params: nu - {}, gamma - {}".format(nu, gamma))
                    ocsvm_clf = OneClassSVM(gamma=gamma, nu=nu)
                    ocsvm_clf.fit(x_train_major, y_train_major)
                    y_ocsvm_predicted = ocsvm_clf.predict(x_train_minor)
                    y_ocsvm_predicted_transformed = np.ones(len(y_ocsvm_predicted), dtype=int)
                    index = 0
                    for item in y_ocsvm_predicted:
                        if item == 1:
                            y_ocsvm_predicted_transformed[index] = 0
                        index = index + 1

                    # filter correctly predicted minority samples
                    x_correctly_predicted_train_minor = []
                    y_correctly_predicted_train_minor = []
                    for i in range(len(y_ocsvm_predicted_transformed)):
                        if y_ocsvm_predicted_transformed[i] == y_train_minor[i]:
                            x_correctly_predicted_train_minor.append(x_train_minor[i])
                            y_correctly_predicted_train_minor.append(y_train_minor[i])

                    if len(x_correctly_predicted_train_minor) > 5:
                        # join correctly predicted minor samples with major samples
                        x_joined_train = x_correctly_predicted_train_minor + x_train_major
                        y_join_train = y_correctly_predicted_train_minor + y_train_major

                        # resample train samples
                        x_resampled, y_resampled = do_resampling(sampling_strategy, x_joined_train, y_join_train)
                        # use proper base classifier
                        if base_classifier == 'SVC':
                            create_svc_model(x_resampled, y_resampled, x_test, y_test, tmp_geometric_mean_scores)
                        elif base_classifier == 'RF':
                            create_random_forest_model(x_resampled, y_resampled, x_test, y_test,
                                                       tmp_geometric_mean_scores)
                        elif base_classifier == 'KNN':
                            create_knn_model(x_resampled, y_resampled, x_test, y_test, tmp_geometric_mean_scores)
                        elif base_classifier == 'AB':
                            create_adabost_model(x_resampled, y_resampled, x_test, y_test, tmp_geometric_mean_scores)
                        else:
                            raise Exception("Unexpected 'base_classifier' input value!")
                    else:
                        print("Skipping, not enough correctly predicted minority samples ")

                # calculate best fold value
                tmp_best_score = np.amax(tmp_geometric_mean_scores)
                print('...............................................................................................')
                print("Current split best score is: {}".format(tmp_best_score))
                print('...............................................................................................')
                best_fold_geometric_mean_score.append(tmp_best_score)

        # calculate best iteration result
        tmp_average_score = statistics.mean(best_fold_geometric_mean_score)
        tmp_std = statistics.stdev(best_fold_geometric_mean_score)
        print('...............................................................................................')
        print('HM, OCSVM, {}, sampling strategy - {}'
              .format(base_classifier, sampling_strategy))
        print("Best iteration score is {} with std {}".format(tmp_average_score, tmp_std))
        print('...............................................................................................')
        best_iteration_average_result.append(tmp_average_score)

    # calculate final result
    final_score = statistics.mean(best_iteration_average_result)
    final_std = statistics.stdev(best_iteration_average_result)
    print('...............................................................................................')
    print('HM, OCSVM, {}, sampling strategy - {}'
          .format(base_classifier, sampling_strategy))
    print("Final G_MEAN score is: {}, std: {}".format(final_score, final_std))
    print('...............................................................................................')


# samples_ratio ['5:1', '39:1', '50:1']
# sampling_strategy ['none', 'ada_syn', 'smote', 'tomek', 'smote_enn', 'ros']
# base_classifier ['SVC', 'RF', 'AB, 'KNN', 'RUSB']
def no_soa_with_sampling(samples_ratio, sampling_strategy, base_classifier):
    # empty list to store average geometric mean result from 1 iteration
    best_iteration_average_result = []
    for iteration in range(0, ITER_COUNT):
        print('...............................................................................................')
        print('NHM, {}, sampling strategy - {}, samples ratio - {}'
              .format(base_classifier, sampling_strategy, samples_ratio))
        print('Current iteration number: {}/{}'.format(iteration + 1, ITER_COUNT))
        print('...............................................................................................')
        # generate data
        x_generated, y_generated = get_data(samples_ratio, iteration * 10)

        # empty list to store best results from each fold
        tmp_best_folds_prediction_results = []
        for train_index, test_index in K_FOLD.split(x_generated):
            x_train, x_test = x_generated[train_index], x_generated[test_index]
            y_train, y_test = y_generated[train_index], y_generated[test_index]

            x_train = STANDARD_SCALER.fit_transform(x_train)
            x_test = STANDARD_SCALER.transform(x_test)

            # empty list to store results from each prediction executed in one fold
            tmp_one_fold_prediction_results = []

            # do resampling if required
            if not sampling_strategy == 'none':
                x_train, y_train = do_resampling(sampling_strategy, x_train, y_train)

            # use proper base classifier
            if base_classifier == 'SVC':
                create_svc_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            elif base_classifier == 'RF':
                create_random_forest_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            elif base_classifier == 'KNN':
                create_knn_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            elif base_classifier == 'AB':
                create_adabost_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            else:
                raise Exception("Unexpected 'base_classifier' input value!")

            # calculate best fold value
            tmp_x = np.amax(tmp_one_fold_prediction_results)
            print('...............................................................................................')
            print("Current split best score is: {}".format(tmp_x))
            print('...............................................................................................')
            tmp_best_folds_prediction_results.append(tmp_x)

        # calculate best iteration result
        tmp_average_score = statistics.mean(tmp_best_folds_prediction_results)
        tmp_std = statistics.stdev(tmp_best_folds_prediction_results)
        print('...............................................................................................')
        print('HM, {}, sampling strategy - {}, samples ratio - {}'
              .format(base_classifier, sampling_strategy, samples_ratio))
        print("Best iteration score is {} with std {}".format(tmp_average_score, tmp_std))
        print('...............................................................................................')
        best_iteration_average_result.append(tmp_average_score)

    # calculate final result
    final_score = statistics.mean(best_iteration_average_result)
    final_std = statistics.stdev(best_iteration_average_result)
    print('...............................................................................................')
    print('NHM, {}, sampling strategy - {}, samples ratio - {}'
          .format(base_classifier, sampling_strategy, samples_ratio))
    print("Final G_MEAN score is: {}, std: {}".format(final_score, final_std))
    print('...............................................................................................')


# samples_ratio ['5:1', '39:1', '50:1']
# sampling_strategy ['none', 'ada_syn', 'smote', 'tomek', 'smote_enn', 'ros']
# base_classifier ['SVC', 'RF', 'AB, 'KNN', 'RUSB']
def no_soa_with_sampling_real_data(bankrupt, non_bankrupt, sampling_strategy, base_classifier):
    # empty list to store average geometric mean result from 1 iteration
    best_iteration_average_result = []
    for iteration in range(0, ITER_COUNT):
        print('...............................................................................................')
        print('NHM, {}, sampling strategy - {}'
              .format(base_classifier, sampling_strategy))
        print('Current iteration number: {}/{}'.format(iteration + 1, ITER_COUNT))
        print('...............................................................................................')
        # generate data
        # random_state param with constant 42 removed due to absolutely equal results in each iteration
        k_fold = KFold(n_splits=5, shuffle=True, random_state=iteration*10)
        import pandas as pd
        x_generated = np.concatenate((non_bankrupt, bankrupt), axis=0)
        y_generated = np.concatenate((pd.DataFrame(np.ones((non_bankrupt.shape[0], 1))),
                                      pd.DataFrame(np.zeros((bankrupt.shape[0], 1)))))

        # empty list to store best results from each fold
        tmp_best_folds_prediction_results = []
        for train_index, test_index in k_fold.split(x_generated):
            x_train, x_test = x_generated[train_index], x_generated[test_index]
            y_train, y_test = y_generated[train_index], y_generated[test_index]

            from sklearn.impute import SimpleImputer
            si = SimpleImputer()
            x_train = si.fit_transform(x_train)
            x_test = si.transform(x_test)

            x_train = STANDARD_SCALER.fit_transform(x_train)
            x_test = STANDARD_SCALER.transform(x_test)

            # empty list to store results from each prediction executed in one fold
            tmp_one_fold_prediction_results = []

            # do resampling if required
            if not sampling_strategy == 'none':
                x_train, y_train = do_resampling(sampling_strategy, x_train, y_train)

            # use proper base classifier
            if base_classifier == 'SVC':
                create_svc_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            elif base_classifier == 'RF':
                create_random_forest_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            elif base_classifier == 'KNN':
                create_knn_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            elif base_classifier == 'AB':
                create_adabost_model(x_train, y_train, x_test, y_test, tmp_one_fold_prediction_results)
            else:
                raise Exception("Unexpected 'base_classifier' input value!")

            # calculate best fold value
            tmp_x = np.amax(tmp_one_fold_prediction_results)
            print('...............................................................................................')
            print("Current split best score is: {}".format(tmp_x))
            print('...............................................................................................')
            tmp_best_folds_prediction_results.append(tmp_x)

        # calculate best iteration result
        tmp_average_score = statistics.mean(tmp_best_folds_prediction_results)
        tmp_std = statistics.stdev(tmp_best_folds_prediction_results)
        print('...............................................................................................')
        print('NHM, {}, sampling strategy - {},'
              .format(base_classifier, sampling_strategy))
        print("Best iteration score is {} with std {}".format(tmp_average_score, tmp_std))
        print('...............................................................................................')
        best_iteration_average_result.append(tmp_average_score)

    # calculate final result
    final_score = statistics.mean(best_iteration_average_result)
    final_std = statistics.stdev(best_iteration_average_result)
    print('...............................................................................................')
    print('NHM, {}, sampling strategy - {}'
          .format(base_classifier, sampling_strategy))
    print("Final G_MEAN score is: {}, std: {}".format(final_score, final_std))
    print('...............................................................................................')
