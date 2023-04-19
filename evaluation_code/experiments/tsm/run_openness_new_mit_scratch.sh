#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_scratch_mit/TSM_openmax_mit.npz" \  # openmax
              "results_scratch_mit/TSM_dropout_mit.npz" \                   # mc dropout
              "results_scratch_mit/TSM_bnn_mit.npz" \                   # bnn svi
              "results_scratch_mit/TSM_softmax_mit.npz" \  # softmax
              "results_scratch_mit/TSM_rpl_mit.npz" \          # rpl
              "results_scratch_mit/TSM_dear_mit.npz" \      # dear
              "results_scratch_mit/TSM_PSL_80_n_s_shu_mit_score.npz")       # PSL (ours)
THRESHOLDS=(0.1 \
            0.000022 \
            0.000003 \
            0.999683 \
            0.999167 \
            0.004549 \
            0.1)

# OOD Detection comparison by using thresholds
echo 'Results by using a specific threshold:'
python experiments/compare_openness_new_mit_scratch.py \
    --base_model tsm \
    --ood_data ${OOD_DATA} \
    --thresholds ${THRESHOLDS[@]} \
    --baseline_results ${RESULT_FILES[@]}

# OOD Detection comparison
# The folders `results/` and `results_baselines` are in the `experiments/tsm/` folder.
echo 'Results by using all thresholds:'
python experiments/compare_openness_new_mit_scratch.py \
    --base_model tsm \
    --ood_data ${OOD_DATA} \
    --baseline_results ${RESULT_FILES[@]}


cd $pwd_dir
echo "Experiments finished!"