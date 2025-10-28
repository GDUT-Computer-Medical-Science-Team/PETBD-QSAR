from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold, RFE, mutual_info_classif, \
    mutual_info_regression, SelectPercentile, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression as LR
from time import time
from utils.DataLogger import DataLogger
import pandas as pd
import numpy as np

logger = DataLogger().getlog("FeatureExtraction")


class FeatureExtraction:
    """
    [Chinese text removed]features（[Chinese text removed]）[Chinese text removed]
    """

    def __init__(self, X, y, mode='regression', VT_threshold=0.02, RFE_features_to_select=50, UFE_percentile=80,
                 verbose=False):
        """
        :param X: [Chinese text removed]features[Chinese text removed]
        :param y: [Chinese text removed]
        :param mode: [Chinese text removed]，[Chinese text removed]：'regression', 'classification'
        :param VT_threshold: VarianceThreshold[Chinese text removed]
        :param RFE_features_to_select: RFE[Chinese text removed]features[Chinese text removed]
        :param UFE_percentile: UFE[Chinese text removed]
        :param verbose: [Chinese text removed]
        """
        self.X = X
        if type(y) is pd.DataFrame:
            self.y = y.squeeze().ravel()
        else:
            self.y = y.ravel()
        if mode not in ['regression', 'classification']:
            raise ValueError("Mode should be 'regression' or 'classification'")
        self.mode = mode
        self.VT_threshold = VT_threshold
        self.RFE_features_to_select = RFE_features_to_select
        self.UFE_percentile = UFE_percentile
        self.verbose = verbose

    def get_VT(self):
        # deleted all features that were either one or zero in more than 98% of samples
        selector = VarianceThreshold(self.VT_threshold)
        return selector

    def get_RFE(self):
        global RF
        if self.mode == 'regression':
            from sklearn.ensemble import RandomForestRegressor as RF
        if self.mode == 'classification':
            from sklearn.ensemble import RandomForestClassifier as RF
        # base estimator SVM
        # estimator = SVC(kernel="rbf")
        # estimator = LR(max_iter=10000, solver='liblinear', class_weight='balanced')
        estimator = RF(n_jobs=-1, verbose=False)
        selector = RFE(estimator=estimator, n_features_to_select=self.RFE_features_to_select, verbose=False)
        # selector = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(2),
        #           scoring='accuracy', n_jobs=-1)
        return selector

    def get_UFE(self):
        selector = None
        if self.mode == 'regression':
            selector = SelectPercentile(score_func=mutual_info_regression, percentile=self.UFE_percentile)
        if self.mode == 'classification':
            selector = SelectPercentile(score_func=mutual_info_classif, percentile=self.UFE_percentile)
        return selector

    def tree_based_selection(self, X, y):
        if self.mode == 'regression':
            clf = ExtraTreesRegressor()
        if self.mode == 'classification':
            clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=self.RFE_features_to_select * 2)
        X_new = model.transform(X)
        return X_new

    def feature_extraction(self, VT=True, TBE=True, UFE=True, RFE=True, returnIndex=False, index_dtype=str):
        """
        [Chinese text removed]features[Chinese text removed]features
        :param VT: [Chinese text removed]VT[Chinese text removed]
        :param TBE: [Chinese text removed]TBE[Chinese text removed]
        :param UFE: [Chinese text removed]UFE[Chinese text removed]
        :param RFE: [Chinese text removed]RFE[Chinese text removed]
        :param returnIndex: return[Chinese text removed]Complete[Chinese text removed]
        :return: Complete[Chinese text removed]features[Chinese text removed]
        :param index_dtype: return[Chinese text removed]
        """
        X = self.X
        if VT:
            X = self.get_VT().fit_transform(X, self.y)
            if self.verbose:
                logger.info(f"X shape after Variance Threshold: {X.shape}")
        if TBE:
            X = self.tree_based_selection(X, self.y)
            if self.verbose:
                logger.info(f"X shape after Tree Based Selection: {X.shape}")
        if UFE:
            X = self.get_UFE().fit_transform(X, self.y)
            if self.verbose:
                logger.info(f"X shape after Select Percentile: {X.shape}")
        if RFE:
            X = self.get_RFE().fit_transform(X, self.y)
            if self.verbose:
                logger.info(f"X shape after Recursive Feature Elimination: {X.shape}")
        if returnIndex:
            return self.get_feature_column_index(X, self.X, dtype=index_dtype)
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def get_feature_column_index(self, X, origin_X, dtype=str) -> list:
        """
        [Chinese text removed]features[Chinese text removed]
        :param X: Completefeatures[Chinese text removed]
        :param origin_X: original[Chinese text removed]
        :param dtype: return[Chinese text removed]，[Chinese text removed]str
        :return:
        """
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        if type(origin_X) is not pd.DataFrame:
            origin_X = pd.DataFrame(origin_X)

        column_header = []

        for idx, col in X.iteritems():
            for origin_idx, origin_col in origin_X.iteritems():
                if col.equals(origin_col):
                    column_header.append(idx)  # [Chinese text removed]features[Chinese text removed]
                    break

        return column_header
