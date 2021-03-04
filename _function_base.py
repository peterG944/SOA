import statistics
# disable warning from Scikit-learn
from warnings import simplefilter

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import RUSBoostClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

simplefilter(action='ignore', category=FutureWarning)

# constants
N_ITERATIONS = 50
RANDOM_STATE_VALUE = 42
K_FOLD = KFold(n_splits=5, random_state=42)
STANDARD_SCALER = StandardScaler()


# generate synthetic data with the following ratio 5:1 (1250:250)
def generate_data_5_1(random_state):
    features, labels = make_classification(n_samples=1500, n_features=20, random_state=random_state,
                                           weights=[0.83333333333333334, 0.16666666666666666])
    return features, labels


# generate synthetic data with the following ratio 39:1 (1458:42) - file1
def generate_data_39_1(random_state):
    features, labels = make_classification(n_samples=1500, n_features=20, weights=[0.975, 0.025],
                                           random_state=random_state)
    return features, labels


# generate synthetic data with the following ratio 50:1 (1471:29) - file2
def generate_data_50_1(random_state):
    features, labels = make_classification(n_samples=1500, n_features=20, random_state=random_state,
                                           weights=[0.984, 0.016])
    return features, labels


# generate data with expected samples ratio
def get_data(samples_ratio, random_state):
    if samples_ratio == '5:1':
        return generate_data_5_1(random_state * 10)
    elif samples_ratio == '39:1':
        return generate_data_39_1(random_state * 10)
    elif samples_ratio == "50:1":
        return generate_data_50_1(random_state * 10)
    else:
        raise Exception("Unexpected input of samples_ratio argument!")


# compute mean_score and standard deviation
def evaluate_average_score_and_std(results):
    mean_score = statistics.mean(results)
    std = statistics.stdev(results)
    return mean_score, std


# helper method to resample data
def do_resampling(sampling_strategy, x_data, y_labels):
    if sampling_strategy == 'ada_syn':
        clf_sample = ADASYN(random_state=42, sampling_strategy='minority')
    elif sampling_strategy == 'smote':
        clf_sample = SMOTE(random_state=42)
    elif sampling_strategy == 'tomek':
        clf_sample = SMOTETomek(random_state=42)
    elif sampling_strategy == 'smote_enn':
        clf_sample = SMOTEENN(random_state=42)
    elif sampling_strategy == 'ros':
        clf_sample = RandomOverSampler(random_state=42)
    else:
        raise Exception("Unsupported sampling technique!")
    return clf_sample.fit_resample(x_data, y_labels)


# helper method to search find best result for SVC classifier
def create_svc_model(x_train_data, y_train_labels, x_test_data, y_test_labels, results):
    gamma_params = [0.001, 0.01, 0.1, 1, 10, 100]
    c_params = [0.01, 0.1, 1, 5, 10, 100]

    for c in c_params:
        for gamma in gamma_params:
            clf = SVC(C=c, gamma=gamma)
            clf.fit(x_train_data, y_train_labels)
            y_predict_labels = clf.predict(x_test_data)
            geo_score = geometric_mean_score(y_test_labels, y_predict_labels)
            print("Base clf (SVC) params: c - {}, gamma - {}, score - {}".format(c, gamma, geo_score))
            results.append(geo_score)


# helper method to search find best result for KNN classifier
def create_knn_model(x_train_data, y_train_labels, x_test_data, y_test_labels, results):
    knn_neighbours_range = [5, 7, 10]
    knn_weights_range = ['uniform', 'distance']
    knn_p_range = [1, 2]
    knn_leaf_size_range = [30, 50, 100, 150]

    for n in knn_neighbours_range:
        for weight in knn_weights_range:
            for p in knn_p_range:
                for leaf_size in knn_leaf_size_range:
                    clf_knn = KNeighborsClassifier(n_neighbors=n, weights=weight, p=p, leaf_size=leaf_size)
                    clf_knn.fit(x_train_data, y_train_labels)
                    y_predict_labels = clf_knn.predict(x_test_data)
                    geo_score = geometric_mean_score(y_test_labels, y_predict_labels)
                    print("Base clf (KNN) params: n_neighbors - {}, weights - {}, p - {}, leaf_size - {}, score - {}"
                          .format(n, weight, p, leaf_size, geo_score))
                    results.append(geo_score)


# helper method to search best results for Random Forest classifier
def create_random_forest_model(x_train_data, y_train_labels, x_test_data, y_test_labels, results):
    rf_estimators_range = [100, 200, 300, 400]
    rf_max_depth_range = [100, 200]
    rf_max_features_range = ['auto', 'log2']

    for estimators in rf_estimators_range:
        for max_depth in rf_max_depth_range:
            for max_features in rf_max_features_range:
                clf_rf = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, max_features=max_features)
                clf_rf.fit(x_train_data, y_train_labels)
                y_predict_labels = clf_rf.predict(x_test_data)
                geo_score = geometric_mean_score(y_test_labels, y_predict_labels)
                print("Base clf (RF) params: n_estimators - {}, max_depth - {}, max_features - {},"
                      " score - {}".format(estimators, max_depth, max_features, geo_score))
                results.append(geo_score)


# helper method to search best results for AdaBoost classifier
def create_adabost_model(x_train_data, y_train_labels, x_test_data, y_test_labels, results):
    ab_estimator_params_range = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for estimator_number in ab_estimator_params_range:
        adaboost_clf = AdaBoostClassifier(n_estimators=estimator_number)
        adaboost_clf.fit(x_train_data, y_train_labels)
        y_predicted = adaboost_clf.predict(x_test_data)
        geo_score = geometric_mean_score(y_test_labels, y_predicted)
        print("Base clf (AB) params: n_estimators - {}, score - {}".format(estimator_number, geo_score))
        results.append(geo_score)


# helper method to search best results for RUSBoost classifier
def create_rusboost_model(x_train_data, y_train_labels, x_test_data, y_test_labels, results):
    rb_estimator_number_range = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    rb_sampling_strategy_range = ['not minority', 'not majority']

    for estimators in rb_estimator_number_range:
        for strategy in rb_sampling_strategy_range:
            clf_rus_boost = RUSBoostClassifier(n_estimators=estimators, sampling_strategy=strategy)
            clf_rus_boost.fit(x_train_data, y_train_labels)
            y_predicted = clf_rus_boost.predict(x_test_data)
            geo_score = geometric_mean_score(y_test_labels, y_predicted)
            print("Base clf (RUSBoost) params: n_estimators - {}, sampling_strategy - {}, score - {}"
                  .format(estimators, strategy, geo_score))
            results.append(geo_score)
