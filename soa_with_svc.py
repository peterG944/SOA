
# this works for Pycharm
from HYBRID_METHOD._functions_hybrid_methods import soa_with_ocsvm
from HYBRID_METHOD._constants import SS_TOMEK, SS_SMOTE, SS_ADASYN, SS_ENN, SS_ROS
from HYBRID_METHOD._constants import SAMPLES_RATIO_02, SAMPLES_RATIO_03
from HYBRID_METHOD._constants import CLF_SVC

# this works for Spyder
# import sys
# sys.path.insert(0, 'D:/pGnip/diplomovka_predikcia/HYBRID_METHOD')
# from _functions_hybrid_methods import hybrid_method_with_ocsvm
# from _constants import SAMPLES_RATIO_01, SAMPLES_RATIO_02, SAMPLES_RATIO_03
# from _constants import SS_ADASYN, SS_SMOTE, SS_TOMEK, SS_ENN, SS_ROS
# from _constants import CLF_SVC

# samples ratio 39:1
soa_with_ocsvm(SAMPLES_RATIO_02, SS_ROS, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_02, SS_ENN, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_02, SS_SMOTE, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_02, SS_ADASYN, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_02, SS_TOMEK, CLF_SVC)

# samples ratio 50:1
soa_with_ocsvm(SAMPLES_RATIO_03, SS_ROS, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_03, SS_ENN, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_03, SS_SMOTE, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_03, SS_ADASYN, CLF_SVC)
soa_with_ocsvm(SAMPLES_RATIO_03, SS_TOMEK, CLF_SVC)
