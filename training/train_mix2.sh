# Mix
##########

python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Real_India', '/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-750', '0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/mix2/expo_real_india_0-750_AND_expo_synt_v8_r3_0-8192__round000 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Real_India', '/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-750', '0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/mix2/expo_real_india_0-750_AND_expo_synt_v8_r3_0-8192__round001 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Real_India', '/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-750', '0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/mix2/expo_real_india_0-750_AND_expo_synt_v8_r3_0-8192__round002 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Real_India', '/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-750', '0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/mix2/expo_real_india_0-750_AND_expo_synt_v8_r3_0-8192__round003 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Real_India', '/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Synt_V8_R3_Batch2',)" \
DG_TRAIN_SET_INDS "('0-750', '0-4096', '0-4096',)" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/training_logs/mix2/expo_real_india_0-750_AND_expo_synt_v8_r3_0-8192__round004 \
SOLVER.MAX_ITER 40000 \
SOLVER.STEPS '(35000, 37000)'
