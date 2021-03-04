
# this works for Pycharm
from HYBRID_METHOD._functions_hybrid_methods import no_soa_with_sampling
from HYBRID_METHOD._constants import SS_SMOTE, SS_ADASYN, SS_NONE
from HYBRID_METHOD._constants import SAMPLES_RATIO_02, SAMPLES_RATIO_03
from HYBRID_METHOD._constants import CLF_AB, CLF_RF, CLF_KNN, CLF_SVC

# this works for Spyder
# import sys
# sys.path.insert(0, 'D:/pGnip/diplomovka_predikcia/HYBRID_METHOD')
# from _functions_hybrid_methods import no_hybrid_method_with_sampling
# from _constants import SAMPLES_RATIO_01, SAMPLES_RATIO_02, SAMPLES_RATIO_03
# from _constants import SS_ADASYN, SS_SMOTE, SS_TOMEK, SS_ENN, SS_ROS, SS_NONE
# from _constants import CLF_KNN, CLF_RF, CLF_RUSBOOST, CLF_SVC, CLF_AB

# KNN classifier
no_soa_with_sampling(SAMPLES_RATIO_02, SS_NONE, CLF_KNN)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_SMOTE, CLF_KNN)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_ADASYN, CLF_KNN)

no_soa_with_sampling(SAMPLES_RATIO_03, SS_NONE, CLF_KNN)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_SMOTE, CLF_KNN)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_ADASYN, CLF_KNN)

# RANDOM FOREST classifier
no_soa_with_sampling(SAMPLES_RATIO_02, SS_NONE, CLF_RF)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_SMOTE, CLF_RF)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_ADASYN, CLF_RF)

no_soa_with_sampling(SAMPLES_RATIO_03, SS_NONE, CLF_RF)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_SMOTE, CLF_RF)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_ADASYN, CLF_RF)

# SVC classifier
no_soa_with_sampling(SAMPLES_RATIO_02, SS_NONE, CLF_SVC)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_SMOTE, CLF_SVC)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_ADASYN, CLF_SVC)

no_soa_with_sampling(SAMPLES_RATIO_03, SS_NONE, CLF_SVC)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_SMOTE, CLF_SVC)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_ADASYN, CLF_SVC)

# AdaBoost classifier
no_soa_with_sampling(SAMPLES_RATIO_02, SS_NONE, CLF_AB)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_SMOTE, CLF_AB)
no_soa_with_sampling(SAMPLES_RATIO_02, SS_ADASYN, CLF_AB)

no_soa_with_sampling(SAMPLES_RATIO_03, SS_NONE, CLF_AB)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_SMOTE, CLF_AB)
no_soa_with_sampling(SAMPLES_RATIO_03, SS_ADASYN, CLF_AB)
