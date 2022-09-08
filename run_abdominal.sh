#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate DG_medical
python main_abdominal.py --GIN --GIN_ch 2 --noise --adv --MI_weight 10.0