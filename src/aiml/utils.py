import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def normalized_accuracy(y_true, y_pred):

    # unique_labels = np.array([0, 1])
    unique_labels = np.unique(y_true)
    assert np.all([p in unique_labels for p in np.unique(y_pred)])

    unique_labels = np.sort(unique_labels)

    accu_dict = dict()

    for l in unique_labels:
        sel = y_true == l
        correct = np.sum((y_pred == y_true)[sel])
        total = np.sum(sel)
        accu_l = correct/total
        # print('[{}] correct: {:d}, total: {:d}, accuracy: {:0.3f}'.format(l, correct, total, accu_l))
        accu_dict[l] = accu_l
    cmt = cm(y_true, y_pred)

    # print(cmt)

    cmt_norm = cmt / np.expand_dims(np.sum(cmt, axis=1), axis=1)

    # print(cmt_norm)
    # mat_pretty_print(cmt_norm)

    return cmt_norm


def confusion_matrix(y_true, y_pred, normalize=None):
    cmt = cm(y_true, y_pred, normalize=normalize)
    return cmt


def binary_classification_metrics(y_true, y_pred, y_prob=None):
    assert set(np.unique(y_true)) == {0, 1}
    cmt = confusion_matrix(y_true, y_pred, normalize=None)
    tp = cmt[1, 1]
    tn = cmt[0, 0]
    fp = cmt[0, 1]
    fn = cmt[1, 0]
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    tpr = sensitivity
    recall = sensitivity

    specificity = tn / (tn + fp)
    tnr = specificity
    selectivity = specificity

    precision = tp / (tp + fp)

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float('nan')

    f1 = 2 * precision * sensitivity / (precision + recall)
    # f2 = 2 / (1/recall + 1/precision)
    # f3 = 2 * tp / (2 * tp + fp + fn)
    cmt_norm = confusion_matrix(y_true, y_pred, normalize='true')
    return {'confusion matrix (unnormalized)': cmt, 'confusion matrix (normalized)': cmt_norm,
            'true positive': tp, 'true negative': tn,
            'false positive': fp, 'false negative': fn,
            'true positive rate': tpr,
            'true negative rate': tnr,
            'false positive rate': 1 - tnr,
            'selectivity': selectivity,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity,
            'auc': auc
            }


def mat_pretty_print(m):

    print(' |', end=' ')
    for r_idx, r in enumerate(m):
        print('{:4d}'.format(r_idx), end=' ')
    print('')
    print('-'*(2+len(m)*5))

    for r_idx, r in enumerate(m):
        print('{}|'.format(r_idx), end=' ')
        for e_idx, e in enumerate(r):
            if int(e) == e:
                format_str = '{:d}'
            else:
                format_str = '{:0.3f}'
            print(format_str.format(e), end=' ')
        print('')
    print('')


def mat_pretty_info(info):

    for i in info:
        print('{}:'.format(i), end='')
        if 'matrix' in i:
            print()
            mat_pretty_print(info[i])
        else:
            print(' {:0.3f}'.format(info[i]))


def get_luke_trop_features(trops, time_hrs):

    trops = np.log(trops)
    vels = np.diff(trops, axis=1) / np.diff(time_hrs, axis=1)

    # using these reduced measures to limit reverse causality, e.g. fewer troponins because clinician diagnosed Normal.
    avgtrop = np.nanmean(trops, axis=1)
    avgspd = np.nanmean(np.abs(vels), axis=1)
    maxtrop = np.nanmax(trops, axis=1)
    mintrop = np.nanmin(trops, axis=1)
    maxvel = np.nanmax(vels, axis=1)
    minvel = np.nanmin(vels, axis=1)
    divtrop = maxtrop / mintrop
    difftrop = maxtrop - mintrop
    diffvel = maxvel - minvel
    x = np.c_[avgtrop, avgspd, maxtrop, mintrop, maxvel, minvel, divtrop, difftrop, diffvel]
    feature_names = ['avgtrop', 'avgspd', 'maxtrop', 'mintrop', 'maxvel', 'minvel', 'divtrop', 'difftrop', 'diffvel']
    x = np.c_[x, trops[:, 0]]
    feature_names += ['logtrop0']

    return x, feature_names


def get_optimal_threshold_roc(y1, s1s):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    for s1 in s1s:
        fpr, tpr, thresholds = roc_curve(y1, s1)
        #  Youden’s J statistic
        J = tpr - fpr
        idx = np.nanargmax(J)
        opt_thresh = thresholds[idx]

        thresh1.append(opt_thresh)

    thresh1 = np.median(thresh1)

    return thresh1


def get_optimal_threshold_pr(y1, s1s):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    for s1 in s1s:
        precision, recall, thresholds = precision_recall_curve(y1, s1)
        #  F-Measure
        fscore = (2 * precision * recall) / (precision + recall)
        idx = np.nanargmax(fscore)
        opt_thresh = thresholds[idx]

        thresh1.append(opt_thresh)

    thresh1 = np.median(thresh1)

    return thresh1