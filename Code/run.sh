##### CodeBERT
ATTRS=("relevance" "informativeness" "expression")
FOLDS=(1 2 3 4 5)

for ATTR in "${ATTRS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    TS=$(date +"%Y%m%d%H%M%S")

    TRAIN_FILE="../data/five_fold_upsampled/train_fold_${FOLD}_${ATTR}_upsampled.csv"
    VALIDATION_FILE="../data/five_fold_upsampled/val_fold_${FOLD}.csv"
    TEST_FILE="../data/five_fold_upsampled/test_fold_${FOLD}.csv"
    RESULT_FILE="results/fold_${FOLD}_${ATTR}_codebert_${TS}.txt"
    OUTPUT_DIR="save/fold_${FOLD}_${ATTR}_codebert"

    CUDA_VISIBLE_DEVICES=4 python run_evaluator.py \
      --seed 1234 \
      --model_name_or_path "microsoft/codebert-base" \
      --task_name $ATTR \
      --label_name $ATTR \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --test_file $TEST_FILE \
      --result_file $RESULT_FILE \
      --max_length 512 \
      --per_device_train_batch_size 32 \
      --learning_rate 1e-5 \
      --num_train_epochs 10 \
      --output_dir $OUTPUT_DIR \
      --pad_to_max_length \
      --checkpointing_steps "epoch" \
      --num_warmup_steps 100
  done
done


##### CodeReviewer
ATTRS=("relevance" "informativeness" "expression")
FOLDS=(1 2 3 4 5)

for ATTR in "${ATTRS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    TS=$(date +"%Y%m%d%H%M%S")

    TRAIN_FILE="../data/five_fold_upsampled/train_fold_${FOLD}_${ATTR}_upsampled.csv"
    VALIDATION_FILE="../data/five_fold_upsampled/val_fold_${FOLD}.csv"
    TEST_FILE="../data/five_fold_upsampled/test_fold_${FOLD}.csv"
    RESULT_FILE="results/fold_${FOLD}_${ATTR}_codereviewer_${TS}.txt"
    OUTPUT_DIR="save/fold_${FOLD}_${ATTR}_codereviewer"

    CUDA_VISIBLE_DEVICES=4 python run_evaluator.py \
      --seed 1234 \
      --model_name_or_path "microsoft/codereviewer" \
      --task_name $ATTR \
      --label_name $ATTR \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --test_file $TEST_FILE \
      --result_file $RESULT_FILE \
      --max_length 512 \
      --per_device_train_batch_size 32 \
      --learning_rate 8e-4 \
      --num_train_epochs 10 \
      --output_dir $OUTPUT_DIR \
      --pad_to_max_length \
      --checkpointing_steps "epoch" \
      --num_warmup_steps 100
  done
done