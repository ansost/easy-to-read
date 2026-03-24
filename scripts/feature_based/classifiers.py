"""Provide functions for running classifiers."""

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def RF(X_train, X_test, y_train):
    clf = RandomForestClassifier(random_state=1, class_weight="balanced").fit(
        X_train, y_train
    )
    preds = clf.predict(X_test)
    return preds, clf.feature_importances_


def SVM(X_train, X_test, y_train):
    clf = SVC(random_state=1, class_weight="balanced").fit(X_train, y_train)
    preds = clf.predict(X_test)
    weights = clf.class_weight_
    return preds, weights


def MLP(X_train, X_test, y_train):
    clf = MLPClassifier(random_state=1, solver="lbfgs", max_iter=5000).fit(
        X_train, y_train
    )
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    return preds, probas


def regression(X_train, X_test, y_train):
    clf = LogisticRegression(
        random_state=0, max_iter=5000, class_weight="balanced", solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return preds, clf.coef_
