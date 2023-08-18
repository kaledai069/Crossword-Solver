TRAIN_FILE=$1
VALID_FILE=$2
OUTPUT_DIR=$3
EPOCH_=$4
BATCH_SIZE=$5

python DPR/train_dense_encoder.py \
	--batch_size ${BATCH_SIZE} \
	--dev_batch_size ${BATCH_SIZE} \
	--sequence_length 32 \
	--warmup_steps 1237 \
	--max_grad_norm 2.0 \
	--learning_rate 2e-05 \
	--seed 1 \
	--num_train_epochs ${EPOCH_} \
	--eval_per_epoch 3 \
	--do_lower_case \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--train_file ${TRAIN_FILE} \
	--dev_file ${VALID_FILE} \
	--output_dir ${OUTPUT_DIR}