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

    result_file = "../../output/test/tsm_maha_distance.npz"
    plt.figure(figsize=(8,5))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    for style, baseline in zip(args.styles, args.baselines):
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
        ind_uncertainties = ind_uncertainties[::2]
        ood_uncertainties = ood_uncertainties[::2]
        print(ind_uncertainties.shape, ood_uncertainties.shape, ind_results.shape, ood_results.shape, ind_labels.shape, ood_labels.shape)

        repeated_clss = [35, 29, 15, 26, 30, 34, 43, 31]
        index_repeated = np.zeros_like(ood_results)
        for i in repeated_clss:
            index_repeated[ood_labels == i] = 1
        index_no_repeated = 1 - index_repeated
        ood_uncertainties = ood_uncertainties[index_no_repeated==1]
        ood_results = ood_results[index_no_repeated==1]
        ood_labels = ood_labels[index_no_repeated==1]

        acc = accuracy_score(ind_labels, ind_results)

        train_uncertainties = results['train_unctt']
        train_labels = results['train_label']
        d_in, d_ood, preds_maha_in, preds_maha_ood = get_eval_results(
        np.copy(train_uncertainties),
        np.copy(ind_uncertainties),
        np.copy(ood_uncertainties),
        np.copy(train_labels),
        clusters=1,
        )
        uncertains = np.concatenate((d_in, d_ood), axis=0)
        labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))

        auroc = roc_auc_score(labels, uncertains)
        aupr = average_precision_score(labels, uncertains)
        fpr, tpr, _ = roc_curve(labels, uncertains)
        print('Model: %s, ClosedSet Accuracy (multi-class): %.3lf, OpenSet AUC (bin-class): %.3lf, OpenSet AUPR (bin-class): %.3lf'%(baseline, acc * 100, auroc * 100, aupr * 100))
        print('FPR95 is: ', fpr[tpr > 0.95][0]*100)

        plt.plot(fpr, tpr)

def get_eval_results(ftrain, ftest, food, labelstrain, clusters=1):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    d_in, d_ood, preds_maha_in, preds_maha_ood = get_scores(ftrain, ftest, food, labelstrain, clusters)

    return d_in, d_ood, preds_maha_in, preds_maha_ood

def get_scores(ftrain, ftest, food, labelstrain, clusters=1):
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        ypred = labelstrain
        # ypred = get_clusters(ftrain, args.clusters)
        return get_scores_multi_cluster_cuda(ftrain, ftest, food, ypred)

def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred

def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]
    start = time.time()

    din_all = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood_all = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    print(time.time()-start)
    din = np.min(din_all, axis=0)
    dood = np.min(dood_all, axis=0)
    preds_maha_in = np.argmin(din_all, axis=0)
    preds_maha_ood = np.argmax(dood_all, axis=0)

    return din, dood, preds_maha_in, preds_maha_ood

def get_scores_multi_cluster_cuda(ftrain, ftest, food, ypred):
    torch.set_default_dtype(torch.double)
    torch.set_printoptions(precision=8)
    ftrain = torch.from_numpy(ftrain).cuda()
    ftest = torch.from_numpy(ftest).cuda()
    food = torch.from_numpy(food).cuda()
    ypred = torch.from_numpy(ypred).cuda()
    xc = [ftrain[ypred == i] for i in torch.unique(ypred)]

    din_all = [
        torch.sum(
            (ftest - torch.mean(x, axis=0, keepdims=True))
            * (
                torch.matmul(torch.linalg.pinv(cov(x.T, bias=True)), (ftest - torch.mean(x, axis=0, keepdims=True)).T)
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    din_all = torch.stack(din_all, axis=1)

    dood_all = [
        torch.sum(
            (food - torch.mean(x, axis=0, keepdims=True))
            * (
                torch.matmul(torch.linalg.pinv(cov(x.T, bias=True)), (food - torch.mean(x, axis=0, keepdims=True)).T)
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood_all = torch.stack(dood_all, axis=1)

    din, preds_maha_in = torch.min(din_all, axis=1)
    dood, preds_maha_ood = torch.min(dood_all, axis=1)


    return din.cpu().numpy(), dood.cpu().numpy(), preds_maha_in.cpu().numpy(), preds_maha_ood.cpu().numpy()

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2)

def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dtrain = np.sum(
        (ftrain - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftrain - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )
    # print(np.max(dtrain), np.min(dtrain))
    # print(np.max(dtest), np.min(dtest))
    # print(np.max(dood), np.min(dood))

    return dtest, dood, None, None

def analyze_prototype(prototypes, ind_features, ind_labels, cls, mean_dis_pro, var_dis_pro, var_feature):
    prototype_cls = prototypes[cls]
    prototype_cls = prototype_cls / np.linalg.norm(prototype_cls)
    ind_features_cls = ind_features[ind_labels == cls]
    # print('Features.var for class 50: ', np.var(ind_features_cls, axis=1))
    sim_cls = np.matmul(ind_features_cls, prototype_cls.T)
    # print('similarity for class 50:', sim_cls)
    ind_features_cls_mean = np.mean(ind_features_cls, axis=0)
    distance = np.matmul(ind_features_cls, prototype_cls.T)
    var_feature_cls = np.var(ind_features_cls, axis=0)
    # print('distance with prototype: ', distance)
    # print('mean distance with prototype', np.mean(distance))
    # fig = plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.bar(range(len(prototype_cls)), prototype_cls)
    # plt.ylim([-0.5,0.6])
    # plt.subplot(1, 2, 2)
    # plt.bar(range(len(prototype_cls)), ind_features_cls_mean)
    # plt.ylim([-0.5,0.6])
    # png_file = "/mnt/data-nas/jcenaa/output/test"
    # plt.savefig(os.path.join(png_file, str(cls) + '_prototype_analyze.png'))
    mean_dis_pro.append(np.mean(distance))
    var_dis_pro.append(np.var(distance))
    var_feature.append(np.mean(var_feature_cls))
    

def get_eval_results_prototype(features_ind, features_ood, prototypes_in):
    features_ind = torch.from_numpy(features_ind)
    features_ood = torch.from_numpy(features_ood)
    prototypes = torch.from_numpy(prototypes_in).float()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    print("prototypes_norm.var: ", torch.var(prototypes, dim=1))

    # prototypes = torch.eye(101)
    # prototypes = torch.cat((prototypes, torch.zeros(101, features_ind.shape[1] - 101)), dim=1)
    sim_ind = torch.matmul(features_ind,prototypes.T)
    sim_ood = torch.matmul(features_ood,prototypes.T)
    preds_in = torch.max(sim_ind, dim=1)[-1]
    preds_ood = torch.max(sim_ood, dim=1)[-1]

    uncertains_ind_sim = 1 - torch.max(sim_ind, dim=1)[0]
    uncertains_ood_sim = 1 - torch.max(sim_ood, dim=1)[0]

    anchor_dot_contrast_ind = torch.div(sim_ind, 0.2)
    anchor_dot_contrast_ood = torch.div(sim_ood, 0.2)

    softmax_layer = torch.nn.Softmax(dim=1)
    anchor_dot_contrast_ind = softmax_layer(anchor_dot_contrast_ind)
    anchor_dot_contrast_ood = softmax_layer(anchor_dot_contrast_ood)

    # print(torch.sum(anchor_dot_contrast_ind,dim=1))
    # uncertains_ind = - torch.sum(xlogy(anchor_dot_contrast_ind, anchor_dot_contrast_ind), dim=1)
    # uncertains_ood = - torch.sum(xlogy(anchor_dot_contrast_ood, anchor_dot_contrast_ood), dim=1)

    uncertains_ind_softmax = 1 - torch.max(anchor_dot_contrast_ind, dim=1)[0]
    uncertains_ood_softmax = 1 - torch.max(anchor_dot_contrast_ood, dim=1)[0]

    # uncertains_ind = uncertains_ind_sim + uncertains_ind_softmax
    # uncertains_ood = uncertains_ood_sim + uncertains_ood_softmax

    uncertains_ind = uncertains_ind_sim
    uncertains_ood = uncertains_ood_sim

    uncertains_ind = uncertains_ind_softmax
    uncertains_ood = uncertains_ood_softmax

    # uncertains_ind = torch.sum(sim_ind, dim=1)
    # uncertains_ood = torch.sum(sim_ood, dim=1)

    return uncertains_ind.numpy(), uncertains_ood.numpy(), preds_in.numpy(), preds_ood.numpy()

def get_eval_results_prototype_unk(features_ind, features_ood, prototypes_in):
    features_ind = torch.from_numpy(features_ind)
    features_ood = torch.from_numpy(features_ood)
    prototypes = torch.from_numpy(prototypes_in).float()
    prototypes = F.normalize(prototypes, p=2, dim=1)

    sim_ind = torch.matmul(features_ind,prototypes.T)
    sim_ood = torch.matmul(features_ood,prototypes.T)

    uncertains_ind_sim = torch.sum(sim_ind[:,-3:], dim=1)
    uncertains_ood_sim = torch.sum(sim_ood[:,-3:], dim=1)

    # uncertains_ind_sim = torch.max(sim_ind[:,-3:], dim=1)[0]
    # uncertains_ood_sim = torch.max(sim_ood[:,-3:], dim=1)[0]

    print(torch.max(sim_ind, dim=1)[1])
    print(torch.max(sim_ind, dim=1)[0])
    # print(uncertains_ood_sim)

    anchor_dot_contrast_ind = torch.div(sim_ind, 0.2)
    anchor_dot_contrast_ood = torch.div(sim_ood, 0.2)

    softmax_layer = torch.nn.Softmax(dim=1)
    anchor_dot_contrast_ind = softmax_layer(anchor_dot_contrast_ind)
    anchor_dot_contrast_ood = softmax_layer(anchor_dot_contrast_ood)

    uncertains_ind_softmax = torch.sum(anchor_dot_contrast_ind[:,-3:], dim=1)
    uncertains_ood_softmax = torch.sum(anchor_dot_contrast_ood[:,-3:], dim=1)

    uncertains_ind_softmax = torch.max(anchor_dot_contrast_ind[:,-3:], dim=1)[0]
    uncertains_ood_softmax = torch.max(anchor_dot_contrast_ood[:,-3:], dim=1)[0]

    uncertains_ind = uncertains_ind_sim
    uncertains_ood = uncertains_ood_sim

    uncertains_ind = uncertains_ind_softmax
    uncertains_ood = uncertains_ood_softmax

    return uncertains_ind.numpy(), uncertains_ood.numpy(), None, None

if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()