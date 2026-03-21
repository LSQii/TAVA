ROOT=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/
CKPT=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/
OUT_DIR=$CKPT/testing
LOAD_CKPT_FILE=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/basetraining/B2N_ssv2_froster/wa_checkpoints/swa_2_16.pth

# TEST_FILE can be set as val.csv (base set) or test.csv (novel set).
# rephrased_file can be set as train_rephrased.json (base set) or test_rephrased.json (novel set)
# NUM_CLASSES can be set as 26 (base set) or 25 (novel set)
B2N_ssv2_file=B2N_ssv2
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv
rephrased_file=test_rephrased.json
NUM_CLASSES=87

cd $ROOT
# MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_SSV2.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$B2N_ssv2_file \
    DATA.PATH_PREFIX /data/datasets/something-something-v2/20bn-something-something-v2/ \
    DATA.PATH_LABEL_SEPARATOR , \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/B2N_ssv2/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 32 \
    NUM_GPUS 2 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES $NUM_CLASSES \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 0 \
