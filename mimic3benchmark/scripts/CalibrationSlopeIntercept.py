# Short description:
# -----------------
# Python implementations of calibration slope, calibration intercept, and calibration in the large for prediction models with a binary outcome.

# Reference (for explanation about the measures)
# ---------
# Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. Journal of clinical epidemiology. 2016 Jun 1;74:167-76.

# Disclaimer:
# -----------
# This code is not affiliated (nor checked by) the authors of the reference above. I did verify on multiple datasets that the outputs are the same as for the val.prob function in the 'rms' R-package using the test() function below.
import sys

from sklearn.linear_model import LogisticRegression
import numpy as np
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr


def logit_function(p):
    print(max(p))
    eps = np.finfo(np.float32).eps
    test1 = np.nextafter(0,1, dtype=np.float32)
    test2 = np.nextafter(1, 0, dtype=np.float32)
    q = np.clip(p, eps, 1-eps) # to prevent division by 0 somewhere
    # between 5 and 8, np.clip decides it will not round down anymore.
    if max(p) == 1 or min(p) == 0:
        print("Here!")
    print(max(q))
    test = 1-eps
    return np.log(q/(1-q))

def calibration_slope_intercept_inthelarge(predicted_y, true_y):
    """
     A function to calculate calibration measures for binary outcomes.
    :param predicted_y: A list/vector of predicted probabilities of the outcome
    :param true_y: A list/vector of the true (binary) outcome.
    :return: A tuple of the (calibration slope, calibration intercept, and the calibration-in-the-large)
    """
    # utils = importr('utils')
    # utils.chooseCRANmirror(ind=1)
    # # utils.install_packages('rlang')
    # # utils.install_packages('scales')
    # # utils.install_packages('ggplot2')
    # utils.install_packages('rms')
    # rms = importr('rms')
    #
    # rval = ro.r['val.prob']
    # y_pred_R = ro.FloatVector(predicted_y)
    # y_R = ro.IntVector(true_y)
    # res_R = rval(y_pred_R, y_R)
    # print('R', res_R)

    y_pred_linear_scores = logit_function(predicted_y)
    CITL = np.mean(true_y) - np.mean(predicted_y)
    calib_model = LogisticRegression(C=1e50)
    try:
        print(max(np.array(y_pred_linear_scores).reshape(-1, 1)), min(np.array(y_pred_linear_scores).reshape(-1, 1)))
        calib_model = calib_model.fit(X=np.array(y_pred_linear_scores).reshape(-1, 1), y=np.array(true_y))
    except Exception as e:
        print(e)
        print('predicted_y', predicted_y)
        print('true_y',true_y)
        print('y_pred_linear_scores',y_pred_linear_scores)
    return (calib_model.coef_[0][0], calib_model.intercept_[0], CITL)

def test():
    # We tested the above function in several dataset to correspond with the R-function val.prob from the 'rms' R-package for the Slope and Intercept measures.
    # Note that to run this function, you need additional packages.

    # from firthlogist import load_endometrial, load_sex2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    # utils.install_packages('rlang')
    # utils.install_packages('scales')
    utils.install_packages('ggplot2')
    utils.install_packages('rms')
    rms = importr('rms')

    #X, y, feature_names = load_endometrial()
    X, y, feature_names = load_sex2()

    # create a prediction model on the dummy data
    PM = LogisticRegression(penalty='l2')
    PM = PM.fit(X, y)

    # Perform predictions on the same data.
    y_pred = PM.predict_proba(X)[:,1]

    # Evaluation via RMS in R
    rval = ro.r['val.prob']
    y_pred_R = ro.FloatVector(y_pred)
    y_R = ro.IntVector(y)
    res_R = rval(y_pred_R, y_R)
    print('R', res_R)

    # Evaluation via the Python function above
    CSlope, CIntercept, CITL = calibration_slope_intercept_inthelarge(y_pred, y)
    print('Measures',CSlope, CIntercept, CITL)

if __name__ == "__main__":
    test()