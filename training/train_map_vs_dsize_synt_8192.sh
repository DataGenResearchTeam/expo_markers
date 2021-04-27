# mAP vs dataset size
##########

python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/map_vs_size/expo_synt_v8_r3_8192_round000 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/map_vs_size/expo_synt_v8_r3_8192_round001 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/map_vs_size/expo_synt_v8_r3_8192_round002 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/map_vs_size/expo_synt_v8_r3_8192_round003 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/map_vs_size/expo_synt_v8_r3_8192_round004 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'