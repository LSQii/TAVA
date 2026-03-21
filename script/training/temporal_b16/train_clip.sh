export CUDA_VISIBLE_DEVICES=0,1,2,3
#ROOT=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/
#CKPT=/data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/
ROOT=/root/TAVA/
CKPT=/root/autodl-tmp


cd $ROOT

TRAIN_FILE=train_tc_clip.csv
VAL_FILE=train_tc_clip.csv
TEST_FILE=val_tc_clip.csv

TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
  --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
  --opts DATA.PATH_TO_DATA_DIR $ROOT/label_db/weng_compress_full_splits \
  DATA.PATH_PREFIX /data/datasets/K400  \
  TRAIN_FILE $TRAIN_FILE \
  VAL_FILE $VAL_FILE \
  TEST_FILE $TEST_FILE \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_rephrase/k400_rephrased_classes.json\
  DATA.PROXY_LABEL_MAPPING_FILE  /data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/retrive_augmented_files/proxy_label_cross_dataset_k400_final.json \
  DATA.ACTION_ATTRIBUTE_MAPPING_FILE /data/users/liz/OV-HAR/attribute_end_to_end/FROSTER-main/retrive_augmented_files/attribute_k400_cross_dataset.json \
  TRAIN.ENABLE True \
  OUTPUT_DIR $CKPT/basetraining/froster \
  TRAIN.BATCH_SIZE 14 \
  TEST.BATCH_SIZE 12 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  NUM_GPUS 2 \
  SOLVER.MAX_EPOCH 22 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  TRAIN.MIXED_PRECISION True \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES 400 \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 13 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  TRAIN.LINEAR_CONNECT_CLIMB False \
  TRAIN.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \
  TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.0 \
  MODEL.RAW_MODEL_DISTILLATION True \
  MODEL.KEEP_RAW_MODEL True \
  MODEL.DISTILLATION_RATIO 2.0 \