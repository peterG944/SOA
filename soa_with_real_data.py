
# this works for Pycharm
from HYBRID_METHOD._constants import SS_SMOTE, SS_ADASYN, SS_NONE
from HYBRID_METHOD._functions_hybrid_methods import soa_with_ocsvm_and_real_data, \
    no_soa_with_sampling_real_data
from HYBRID_METHOD._constants import CLF_SVC, CLF_KNN, CLF_RF, CLF_AB

# this works for Spyder
# import sys
# sys.path.insert(0, 'D:/pGnip/diplomovka_predikcia/HYBRID_METHOD')
# from _functions_hybrid_methods import hybrid_method_with_ocsvm_and_real_data_input,
# no_hybrid_method_with_sampling_real_data
# from _constants import SS_ADASYN, SS_SMOTE, SS_NONE
# from _constants import CLF_SVC, CLF_KNN, CLF_AB, CLF_RF


# here, function from another package is used to load subset of bankruptcy dataset
import sys
sys.path.insert(0, 'D:/pGnip/diplomovka_predikcia')
from A_functions_data import get_updated_industry_data

ROW_NAN_LIMIT = 10
# loading manufacture data (in paper file3.csv)
bankrupt_industry_list, non_bankrupt_industry_list = get_updated_industry_data(ROW_NAN_LIMIT)


# SVC classifier
soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_SVC)
soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_SVC)

no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_SVC)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_SVC)

# KNN classifier
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_KNN)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_KNN)

soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_KNN)
soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_KNN)


#  AdaBoost classifier
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_AB)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_AB)

soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_AB)
soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_AB)


# Random Forest classifier
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_RF)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_RF)

soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_ADASYN, CLF_RF)
soa_with_ocsvm_and_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_SMOTE, CLF_RF)


# all utilized classifier without any sampling
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_NONE, CLF_SVC)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_NONE, CLF_KNN)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_NONE, CLF_AB)
no_soa_with_sampling_real_data(bankrupt_industry_list[3], non_bankrupt_industry_list[3], SS_NONE, CLF_RF)
