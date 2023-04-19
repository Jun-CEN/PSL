#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_ft_hmdb/TSM_openmax_ft.npz" \  # openmax
              "results/TSM_DNN_BALD_${OOD_DATA}_result.npz" \                   # mc dropout
              "results/TSM_BNN_BALD_${OOD_DATA}_result.npz" \                   # bnn svi
              "results_baselines/openmax/TSM_OpenMax_${OOD_DATA}_result.npz" \  # softmax
              "results_baselines/rpl/TSM_RPL_${OOD_DATA}_result.npz" \          # rpl
              "results/TSM_EDLNoKLAvUCDebias_EDL_${OOD_DATA}_result.npz" \      # dear 
              "results_ft_hmdb/TSM_PSL_80_n_s_shu_ft_score.npz")       # PSL(ours)
THRESHOLDS=(0.1 \
            0.000022 \
            0.000003 \
            0.999683 \
            0.999167 \
            0.004549 \
            0.1)

# OOD Detection comparison by using thresholds
echo 'Results by using a specific threshold:'
python experiments/compare_openness_new_hmdb.py \
    --base_model tsm \
    --ood_data ${OOD_DATA} \
    --thresholds ${THRESHOLDS[@]} \
    --baseline_results ${RESULT_FILES[@]}
# OOD Detection comparison
# The folders `results/` and `results_baselines` are in the `experiments/tsm/` folder.
echo 'Results by using all thresholds:'
python experiments/compare_openness_new_hmdb.py \
    --base_model tsm \
    --ood_data ${OOD_DATA} \
    --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"