# for i in {1..3}; do
#      taskset -c "51" python main.py --config "algorithms/mDSDI/configs/PACS_photo.json" --exp_idx $i --gpu_idx "1"
# done

# tensorboard --logdir "/home/ubuntu/mDSDI/algorithms/mDSDI/results/tensorboards/PACS_photo_1"
# python utils/tSNE_plot.py --plotdir "algorithms/mDSDI/results/plots/PACS_photo_1/"

# rm -r algorithms/mDSDI/results/checkpoints/*
# rm -r algorithms/mDSDI/results/logs/*
# rm -r algorithms/mDSDI/results/plots/*
# rm -r algorithms/mDSDI/results/tensorboards/*
#!/bin/bash

'./algorithms/ERM/configs/RCC_igc.json' './algorithms/ERM/configs/RCC_nci.json' './algorithms/ERM/configs/RCC_mskcc.json' './algorithms/ERM/configs/RCC_mixed.json' './algorithms/mDSDI/configs/RCC_igc.json' './algorithms/mDSDI/configs/RCC_nci.json' './algorithms/mDSDI/configs/RCC_mskcc.json' './algorithms/mDSDI/configs/RCC_mixed.json' './algorithms/mHSHA/configs/RCC_igc.json' './algorithms/mHSHA/configs/RCC_nci.json' './algorithms/mHSHA/configs/RCC_mskcc.json' './algorithms/mHSHA/configs/RCC_mixed.json'



for arg in './algorithms/ERM/configs/RCC_igc.json' './algorithms/ERM/configs/RCC_nci.json' './algorithms/ERM/configs/RCC_mskcc.json' './algorithms/ERM/configs/RCC_mixed.json' './algorithms/mDSDI/configs/RCC_igc.json' './algorithms/mDSDI/configs/RCC_nci.json' './algorithms/mDSDI/configs/RCC_mskcc.json' './algorithms/mDSDI/configs/RCC_mixed.json' './algorithms/mHSHA/configs/RCC_igc.json' './algorithms/mHSHA/configs/RCC_nci.json' './algorithms/mHSHA/configs/RCC_mskcc.json' './algorithms/mHSHA/configs/RCC_mixed.json'

do
    python main.py --config $arg
done