cd ../

OOD_DATA=HMDB  # HMDB or MiT
case ${OOD_DATA} in
    HMDB)
    NUM_CLASSES=51
    ;;
    MiT)
    NUM_CLASSES=305
    ;;
    *)
    echo "Invalid OOD Dataset: "${OOD_DATA}
    exit
    ;;
esac

# OOD Detection comparison
python compute_openness.py \
    --base_model i3d \
    --baselines maha_distance \
    --thresholds 0.000433 \
    --styles b \
    --ood_data ${OOD_DATA} \
    --ood_ncls ${NUM_CLASSES} \
    --ind_ncls 101 \
    --result_png F1_openness_compare_${OOD_DATA}.png \
    --analyze true
    

cd $pwd_dir
echo "Experiments finished!"