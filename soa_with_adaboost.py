
from HYBRID_METHOD._functions_hybrid_methods import soa_with_ocsvm
from HYBRID_METHOD._constants import SS_SMOTE, SS_ADASYN
from HYBRID_METHOD._constants import SAMPLES_RATIO_02, SAMPLES_RATIO_03
from HYBRID_METHOD._constants import CLF_AB

# this works for Spyder
# import sys
# sys.path.insert(0, 'D:/pGnip/diplomovka_predikcia/HYBRID_METHOD')
# from _functions_hybrid_methods import hybrid_method_with_ocsvm
# from _constants import SAMPLES_RATIO_01, SAMPLES_RATIO_02, SAMPLES_RATIO_03
# from _constants import SS_ADASYN, SS_SMOTE, SS_ROS, SS_ENN, SS_TOMEK
# from _constants import CLF_AB

# samples ratio 39:1
soa_with_ocsvm(SAMPLES_RATIO_02, SS_SMOTE, CLF_AB)
soa_with_ocsvm(SAMPLES_RATIO_02, SS_ADASYN, CLF_AB)

# samples ratio 50:1
soa_with_ocsvm(SAMPLES_RATIO_03, SS_SMOTE, CLF_AB)
soa_with_ocsvm(SAMPLES_RATIO_03, SS_ADASYN, CLF_AB)
