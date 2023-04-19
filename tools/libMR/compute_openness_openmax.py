import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.covariance import ledoit_wolf
import time
import torch
from scipy.special import xlogy
import torch.nn.functional as F
import scipy.spatial.distance as spd
import libmr

def parse_args():
    '''Command instruction:
        source activate mmaction
        python experiments/compare_openness.py --ind_ncls 101 --ood_ncls 51
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--base_model', default='i3d', help='the backbone model name')
    parser.add_argument('--baselines', nargs='+', default=['I3D_Dropout_BALD', 'I3D_BNN_BALD', 'I3D_EDLlog_EDL', 'I3D_EDLlogAvUC_EDL'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.000423, 0.000024, 0.495783, 0.495783])
    parser.add_argument('--styles', nargs='+', default=['-b', '-k', '-r', '-g', '-m'])
    parser.add_argument('--ind_ncls', type=int, default=101, help='the number of classes in known dataset')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    parser.add_argument('--ood_data', default='HMDB', help='the name of OOD dataset.')
    parser.add_argument('--num_rand', type=int, default=10, help='the number of random selection for ood classes')
    parser.add_argument('--result_png', default='F1_openness_compare_HMDB.png')
    parser.add_argument('--analyze', default=False, help="analyze score distribution")
    args = parser.parse_args()
    return args


def main():

    result_file = "/mnt/data-nas/jcenaa/output/test/tsm_maha_distance_sc_final.npz"
    result_file_softmax = "/mnt/data-nas/jcenaa/output/test/tsm_maha_distance_sc_ce_logits.npz"
    png_file = "/mnt/data-nas/jcenaa/output/test/test.png"
    plt.figure(figsize=(8,5))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    # result_file = "/mnt/data-nas/jcenaa/output/test/tsm_softmax_" + baseline + ".npz"
    print(result_file)
    assert os.path.exists(result_file), "File not found! Run ood_detection first!"
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    results_logits = np.load(result_file_softmax, allow_pickle=True)
    ind_logits = results_logits['ind_unctt']
    ood_logits = results_logits['ood_unctt']
    print(ind_logits.shape, ood_logits.shape)

    repeated_clss = [35, 29, 15, 26, 30, 34, 43, 31]
    index_repeated = np.zeros_like(ood_results)
    for i in repeated_clss:
        index_repeated[ood_labels == i] = 1
    index_no_repeated = 1 - index_repeated
    ood_uncertainties = ood_uncertainties[index_no_repeated==1]
    ood_results = ood_results[index_no_repeated==1]
    ood_labels = ood_labels[index_no_repeated==1]

    acc = accuracy_score(ind_labels, ind_results)
    # open-set auc-roc (binary class)

    # mav_dist_list = compute_mav_dist(ind_logits, ind_labels, ind_results)
    mav_dist_list = []
    for cls_gt in range(101):
        mav_dist_file = "/mnt/data-nas/jcenaa/output/test/mav_dist_2_"+str(cls_gt)+".npz"
        mav_dist_list.append(mav_dist_file)
    
    # print("Weibull fitting...")
    # weibull_model = weibull_fitting(mav_dist_list)
    # ind_uncertainties_score = []
    # ood_uncertainties_score = []
    # for i in range(ind_logits.shape[0]):
    #     print(i)
    #     openmax_prob_ind = openmax_recalibrate(weibull_model, ind_uncertainties, ind_logits[i])
    #     ind_uncertainties_score.append(openmax_prob_ind[0][-1])
    # for i in range(ood_logits.shape[0]):
    #     print(i)
    #     openmax_prob_ood = openmax_recalibrate(weibull_model, ood_uncertainties, ood_logits[i])
    #     ood_uncertainties_score.append(openmax_prob_ood[0][-1])
    # print(len(openmax_prob_ind), len(openmax_prob_ood))
    # ind_uncertainties_score = np.array(ind_uncertainties_score)
    # ood_uncertainties_score = np.array(ood_uncertainties_score)
    # np.savez('/mnt/data-nas/jcenaa/output/test/openmax.npz', ind_uncertainties=ind_uncertainties_score, ood_uncertainties=ood_uncertainties_score)
    results_openmax = np.load('/mnt/data-nas/jcenaa/output/test/openmax.npz', allow_pickle=True)
    ind_uncertainties_score = results_openmax['ind_uncertainties']
    ood_uncertainties_score = results_openmax['ood_uncertainties']
    ood_uncertainties_score = ood_uncertainties_score[index_no_repeated==1]
    uncertains = np.concatenate((ind_uncertainties_score, ood_uncertainties_score))
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))


    auroc = roc_auc_score(labels, uncertains)
    aupr = average_precision_score(labels, uncertains)
    fpr, tpr, _ = roc_curve(labels, uncertains)
    print('Model: %s, ClosedSet Accuracy (multi-class): %.3lf, OpenSet AUC (bin-class): %.3lf, OpenSet AUPR (bin-class): %.3lf'%('openmax', acc * 100, auroc * 100, aupr * 100))
    print('FPR95 is: ', fpr[tpr > 0.95][0]*100)

    # plt.plot(fpr, tpr)
        
def compute_mav_dist(features, labels, preds):
    num_cls = 101
    mav_dist_list = []
    for cls_gt in range(num_cls):
        mav_dist_file = "/mnt/data-nas/jcenaa/output/test/mav_dist_2_"+str(cls_gt)+".npz"
        mav_dist_list.append(mav_dist_file)
        # extract MAV features
        features_cls = features[(labels==cls_gt)*(preds==cls_gt)]
        mav_train = np.mean(features, axis=0)
        # compute distance
        eucos_dist, eu_dist, cos_dist = compute_distance(mav_train, features_cls)
        print(cls_gt, eucos_dist.shape, eu_dist.shape, cos_dist.shape)
        # save MAV and distances
        np.savez(mav_dist_file, mav=mav_train, eucos=eucos_dist, eu=eu_dist, cos=cos_dist)
    return mav_dist_list

def compute_distance(mav, features):
    # extract features and compute distances for each class
    num_channels = mav.shape[0]
    eucos_dist, eu_dist, cos_dist = [], [], []
    for feat in features:
        # compute distance of each channel
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        for c in range(num_channels):
            # eu_channel += [spd.euclidean(mav[c, :], feat[c, :])/200.]
            # cos_channel += [spd.cosine(mav[c, :], feat[c, :])]
            # eu_cos_channel += [spd.euclidean(mav[c, :], feat[c, :]) / 200. 
            #                  + spd.cosine(mav[c, :], feat[c, :])]  # Here, 200 is from the official OpenMax code
            eu_channel += [spd.euclidean(mav[c], feat[c])/200.]
            cos_channel += [spd.cosine(mav[c], feat[c])]
            eu_cos_channel += [spd.euclidean(mav[c], feat[c]) / 200. 
                             + spd.cosine(mav[c], feat[c])]  # Here, 200 is from the official OpenMax code
        eu_dist += [eu_channel]
        cos_dist += [cos_channel]
        eucos_dist += [eu_cos_channel]
    return np.array(eucos_dist), np.array(eu_dist), np.array(cos_dist)

def weibull_fitting(mav_dist_list, distance_type='eucos', tailsize=20):
    weibull_model = {}
    for cls_gt in range(len(mav_dist_list)):
        # load the mav_dist file
        cache = np.load(mav_dist_list[cls_gt], allow_pickle=True)
        mav_train = cache['mav']
        distances = cache[distance_type]

        weibull_model[cls_gt] = {}
        weibull_model[cls_gt]['mean_vec'] = mav_train

        # weibull fitting for each channel
        weibull_model[cls_gt]['weibull_model'] = []
        num_channels = mav_train.shape[0]
        for c in range(num_channels):
            mr = libmr.MR()
            tailtofit = sorted(distances[:, c])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[cls_gt]['weibull_model'] += [mr]
    return weibull_model

def openmax_recalibrate(weibull_model, feature, score, rank=1, distance_type='eucos'):
    num_channels = 101
    num_cls = 101
    # get the ranked alpha
    alpharank = min(num_cls, rank)
    ranked_list = np.mean(score, axis=0).argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = np.zeros((num_cls,))
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    # calibrate
    openmax_score, openmax_score_u = [], []
    for c in range(num_channels):
        channel_scores = score
        openmax_channel = []
        openmax_unknown = []
        for cls_gt in range(num_cls):
            # get distance between current channel and mean vector
            mav_train = weibull_model[cls_gt]['mean_vec']
            category_weibull = weibull_model[cls_gt]['weibull_model']
            # channel_distance = compute_channel_distance(mav_train[c, :], feature[c, :], distance_type=distance_type)
            channel_distance = compute_channel_distance(mav_train[c], feature[c], distance_type=distance_type)
            # obtain w_score for the distance and compute probability of the distance
            wscore = category_weibull[c].w_score(channel_distance)
            modified_score = channel_scores[cls_gt] * ( 1 - wscore*ranked_alpha[cls_gt] )
            openmax_channel += [modified_score]
            openmax_unknown += [channel_scores[cls_gt] - modified_score]
        # gather modified scores for each channel
        openmax_score += [openmax_channel]
        openmax_score_u += [openmax_unknown]
    openmax_score = np.array(openmax_score)
    openmax_score_u = np.array(openmax_score_u)
    # Pass the recalibrated scores into openmax
    openmax_prob = compute_openmax_prob(openmax_score, openmax_score_u)


    return openmax_prob

def compute_openmax_prob(openmax_score, openmax_score_u):
    num_channels, num_cls = openmax_score.shape
    prob_scores, prob_unknowns = [], []
    for c in range(num_channels):
        channel_scores, channel_unknowns = [], []
        for gt_cls in range(num_cls):
            channel_scores += [np.exp(openmax_score[c, gt_cls])]
        
        total_denominator = np.sum(np.exp(openmax_score[c, :])) + np.exp(np.sum(openmax_score_u[c, :]))
        prob_scores += [channel_scores/total_denominator ]
        prob_unknowns += [np.exp(np.sum(openmax_score_u[c, :]))/total_denominator]
        
    prob_scores = np.array(prob_scores)
    prob_unknowns = np.array(prob_unknowns)

    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    assert len(modified_scores) == num_cls + 1
    modified_scores = np.expand_dims(np.array(modified_scores), axis=0)
    return modified_scores

def compute_channel_distance(mav_channel, feat_channel, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mav_channel, feat_channel)/200. + spd.cosine(mav_channel, feat_channel)
    elif distance_type == 'eu':
        query_distance = spd.euclidean(mav_channel, feat_channel)/200.
    elif distance_type == 'cos':
        query_distance = spd.cosine(mav_channel, feat_channel)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance

if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()