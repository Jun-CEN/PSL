import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from compute_openness import get_eval_results, get_scores, get_clusters, get_scores_multi_cluster, get_scores_one_cluster, get_eval_results_prototype
from sklearn import manifold

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
    parser.add_argument('--t_SNE', default=False, help='plot the embedding')
    parser.add_argument('--t_SNE_cls', default=False, help='plot the embedding of an ind class')
    parser.add_argument('--analyze', default=False, help="analyze score distribution")
    args = parser.parse_args()
    return args


def main():

    result_file = "/mnt/data-nas/jcenaa/output/test/tsm_maha_distance_sc_l_prototype_5050_n_i_6.npz"
    result_file_softmax = "/mnt/data-nas/jcenaa/output/test/tsm_softmax_sc_final.npz"
    png_file = "/mnt/data-nas/jcenaa/output/test"
    plt.figure(figsize=(8,5))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    for style, baseline in zip(args.styles, args.baselines):
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

        if baseline == "maha_distance":
            train_uncertainties = results['train_unctt']
            train_labels = results['train_label']
            ind_uncertainties, ood_uncertainties, _, _ = get_eval_results(
            np.copy(train_uncertainties),
            np.copy(ind_uncertainties),
            np.copy(ood_uncertainties),
            np.copy(train_labels),
            clusters=1,
            )
            uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
            ind_uncertainties = (ind_uncertainties-np.min(uncertains)) / (np.max(uncertains) - np.min(uncertains)) # normalize
            ood_uncertainties = (ood_uncertainties-np.min(uncertains)) / (np.max(uncertains) - np.min(uncertains)) # normalize
        elif baseline == "prototype":
            try:
                results['prototypes']
            except KeyError:
                prototypes = np.eye(101)
                prototypes = np.concatenate((prototypes, np.zeros((101, ind_uncertainties.shape[1] - 101), dtype=float)), axis=1)
            else:
                prototypes = results['prototypes']
            print("Prototypes: ", prototypes)
            ind_uncertainties, ood_uncertainties, preds_maha_in, preds_maha_ood = get_eval_results_prototype(
            np.copy(ind_uncertainties),
            np.copy(ood_uncertainties),
            prototypes_in = prototypes)
            uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
            # print("Prototypes.norm: ", np.linalg.norm(prototypes, axis=1))
            # acc = accuracy_score(ind_labels, preds_maha_in)
            ind_uncertainties = (ind_uncertainties-np.min(uncertains)) / (np.max(uncertains) - np.min(uncertains)) # normalize
            ood_uncertainties = (ood_uncertainties-np.min(uncertains)) / (np.max(uncertains) - np.min(uncertains)) # normalize
        else:
            ind_uncertainties = (ind_uncertainties-np.min(uncertains)) / (np.max(uncertains) - np.min(uncertains)) # normalize
            ood_uncertainties = (ood_uncertainties-np.min(uncertains)) / (np.max(uncertains) - np.min(uncertains)) # normalize
        dataName_ind = 'UCF-101'
        dataName_ood = args.ood_data

        if dataName_ood == 'MIT':
            dataName_ood = 'MiT-v2'
        if dataName_ood == 'HMDB':
            dataName_ood = 'HMDB-51'
        
        try:
            result_file_softmax
        except NameError:
            pass
        else:
            results_softmax = np.load(result_file_softmax, allow_pickle=True)
            ind_uncertainties_softmax = results_softmax['ind_unctt']  # (N1,)
            ood_uncertainties_softmax = results_softmax['ood_unctt']
            ind_uncertainties_softmax = (ind_uncertainties_softmax-np.min(ind_uncertainties_softmax)) / (np.max(ind_uncertainties_softmax) - np.min(ind_uncertainties_softmax)) # normalize
            ood_uncertainties_softmax = (ood_uncertainties_softmax-np.min(ood_uncertainties_softmax)) / (np.max(ood_uncertainties_softmax) - np.min(ood_uncertainties_softmax)) # normalize
            # ind_uncertainties = ind_uncertainties_softmax
            # ood_uncertainties = ood_uncertainties_softmax
        
        uncertainties_mean_all = []
        preds_ood = []
        for unknown_cls in np.unique(ood_labels):
            uncertainties_mean = np.mean(ood_uncertainties[ood_labels == unknown_cls])
            uncertainties_mean_all.append(uncertainties_mean)
            preds_ood.append(np.unique(ood_results[ood_labels == unknown_cls], return_counts=True))
        sorted_id = sorted(range(len(uncertainties_mean_all)), key=lambda k: uncertainties_mean_all[k], reverse=True)


        plt.figure(figsize=(5,4))  # (w, h)
        plt.rcParams["font.family"] = "Arial"  # Times New Roman
        fontsize = 15
        plt.hist([ind_uncertainties, ood_uncertainties], 50, 
                density=True, histtype='bar', color=['blue', 'red'], 
                label=['in-distribution (%s)'%(dataName_ind), 'out-of-distribution (%s)'%(dataName_ood)])
        plt.legend(fontsize=fontsize)
        plt.xlabel('uncertainty', fontsize=fontsize)
        plt.ylabel('density', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.xlim(0, 1.01)
        plt.ylim(0, 10.01)
        plt.tight_layout()
        plt.savefig(os.path.join(png_file, baseline + '_distribution.png'))
        plt.savefig(os.path.join(png_file, baseline + '_distribution.pdf'))

        draw_eigenvalue(results['ind_unctt'], results['ind_label'])

        if args.t_SNE_cls == True:
            prototypes = np.divide(prototypes , np.expand_dims(np.linalg.norm(prototypes, axis=1), 1).repeat(prototypes.shape[1], 1))
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            ind_features = results['ind_unctt']  # (N1,)
            ood_features = results['ood_unctt']
            X = []
            Y = []
            P = []
            Y_P = []
            for i in range(10):
                cls_analyze = i*10
                ind_features_cls = ind_features[ind_labels == cls_analyze]
                x = ind_features_cls
                print(x.shape)
                y = np.ones((ind_features_cls.shape[0])) * cls_analyze
                X.append(x)
                Y.append(y)
                P.append(prototypes[cls_analyze][np.newaxis,:])
                Y_P.append(cls_analyze)
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            P = np.concatenate(P, axis=0)
            X_in = np.concatenate((X, P), axis=0)
            X = np.concatenate((X_in, ood_features), axis=0)
            Y = np.concatenate((Y, Y_P), axis=0)
            X_tsne = tsne.fit_transform(X)
            print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_in.shape[0]-5):
                # plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.Set1(Y[i]/41), 
                #         fontdict={'weight': 'bold', 'size': 9})
                plt.scatter(X_norm[i, 0], X_norm[i, 1], 5, color=plt.cm.Set1(Y[i]/91))
            for i in range(X_in.shape[0]-10, X_in.shape[0]):
                plt.scatter(X_norm[i, 0], X_norm[i, 1], 80, color=plt.cm.Set1(Y[i]/91))
            for i in range(X_in.shape[0], X_norm.shape[0]):
                plt.scatter(X_norm[i, 0], X_norm[i, 1], 50, color=plt.cm.Set1(Y[-1]/91), marker='1')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(png_file, baseline + '_tSNE_cls.png'))

        if args.t_SNE:
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            ind_features = results['ind_unctt']  # (N1,)
            ood_features = results['ood_unctt']
            X = np.concatenate((ind_features, ood_features), axis=0)
            print(X.shape)
            X_tsne = tsne.fit_transform(X)
            y = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))

            print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                        fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(png_file, baseline + '_tSNE_0.png'))

def draw_eigenvalue(features_ind, labels_ind):
    labels_interest = 0
    features_ind = features_ind[labels_ind == labels_interest]
    features_ind_mean = np.mean(features_ind, axis=0)

    print(features_ind_mean.shape)
    features_ind = features_ind - features_ind_mean
    features_ind = np.expand_dims(features_ind, axis=-1)
    features_ind_T = features_ind.swapaxes(2,1)
    va_matrix = np.matmul(features_ind,features_ind_T)
    va_matrix = np.mean(va_matrix, axis=0)
    u,s,v = np.linalg.svd(va_matrix)
    s = np.log(s)
    print(s)
    print(s.mean())



if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()