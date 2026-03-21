ROOT=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/
CKPT=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/

B2N_ssv2_file=B2N_ssv2
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv

cd $ROOT

TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
  --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_SSV2.yaml \
  --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$B2N_ssv2_file \
  TRAIN_FILE $TRAIN_FILE \
  VAL_FILE $VAL_FILE \
  TEST_FILE $TEST_FILE \
  DATA.PATH_PREFIX /data/datasets/something-something-v2/20bn-something-something-v2/ \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/B2N_ssv2/train_rephrased.json \
  DATA.PROXY_LABEL_MAPPING_FILE  /data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/zs_label_db/B2N_ssv2/test_rephrased.json \
  DATA.ACTION_ATTRIBUTE_MAPPING_FILE /data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/final_visual_attributes_SSv2.json \
  TRAIN.ENABLE True \
  OUTPUT_DIR $CKPT/basetraining/B2N_ssv2_froster \
  TRAIN.BATCH_SIZE 12 \
  TEST.BATCH_SIZE 48 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  NUM_GPUS 4 \
  SOLVER.MAX_EPOCH 16 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  TRAIN.MIXED_PRECISION True \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES 87 \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 6 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  TRAIN.LINEAR_CONNECT_CLIMB False \
  TRAIN.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
  TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.0 \
  MODEL.RAW_MODEL_DISTILLATION True \
  MODEL.KEEP_RAW_MODEL True \
  MODEL.DISTILLATION_RATIO 2.0