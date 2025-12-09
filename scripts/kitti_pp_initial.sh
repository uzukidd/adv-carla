if [ -z "$1" ]; then
    echo "Fatal error: please provide a dataset name"
    exit 1
fi

DATASET="$1"
CONFIG_HPARAMS="physicalADV_hparams.yaml"

if [ "$2" == "-cf" ]; then
    CONFIG_HPARAMS="physicalADV_hparams_constant_reflectness.yaml"
fi

python lightning_main.py \
    --config="cfgs/physical_attack/${DATASET}/${CONFIG_HPARAMS}" \
    test \
    --config="cfgs/physical_attack/${DATASET}/physicalADV_pp.yaml"