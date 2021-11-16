from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier #xgboost-사이킷래퍼
# import xgboost as xgb ## xgboost-파이썬래퍼
from sklearn.ensemble import GradientBoostingClassifier
# from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

def Set_KNeighborsClassifier():
    set_model = KNeighborsClassifier(n_neighbors=3)
    # print(set_model)
    return set_model

def Set_RandomForestClassifier():
    set_model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=1, warm_start=False)
    # print(set_model)
    return set_model

def Set_GaussianNB():
    set_model = GaussianNB(priors=None, var_smoothing=1e-09)
    # print(set_model)
    return set_model

def Set_AdaBoostClassifier():
    set_model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
    # print(set_model)
    return set_model

def Set_DecisionTreeClassifier():
    set_model = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=None, splitter='best')
    # print(set_model)
    return set_model

def Set_GradientBoostingClassifier():
    set_model = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None,
                           random_state=2021, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
    # print(set_model)
    return set_model

def Set_BaggingClassifier():
    set_model = BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False,
                  max_features=1.0, max_samples=1.0, n_estimators=10,
                  n_jobs=None, oob_score=False, random_state=0, verbose=0,
                  warm_start=False)
    # print(set_model)
    return set_model

def Set_SVC():
    set_model = SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    # predict_proba() 사용을 위해서는 "probability=True" 설정 필요
    # print(set_model)
    return set_model

def Set_voting(models_dict):
    voting_models = [(k,v) for k, v in models_dict.items()]
    # voting_models = [('Bagging', bag_clf), ('LightGBM', lgbm_clf), ('NaiveBayes',gnb)]
    set_model = VotingClassifier(estimators=voting_models, voting='soft')
    return set_model



# def Set_LGBMClassifier():
#     set_model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#                importance_type='split', learning_rate=0.1, max_depth=-1,
#                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
#                n_estimators=200, n_jobs=-1, num_leaves=31, objective=None,
#                random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
#     print(set_model)
#     return set_model

# def Set_XGBClassifier_sk():
#     set_model = XGBClassifier(n_estimators=200,
#                               learning_rate=0.1,
#                               base_score=0.5,
#                               early_stopping_rounds=20,
#                               verbose=True,
#                               eval_metric='merror',
#                               eval_set=[(x_test, y_test)]
#                 )
#     print(set_model)
#     return set_model