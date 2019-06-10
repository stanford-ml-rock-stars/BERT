#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 python3.5 run_squad_fix_mt.py \
	--bert_model=bert-base-uncased \
	--train_file=../huggingface_bert/drop_train_add_sub.json \
	--predict_file=../huggingface_bert/drop_dev_add_sub.json \
	--output_dir=tempdir6 \
	--version_2_with_negative \
	--learning_rate=3e-5 \
	--num_train_epochs=1.0 \
	--max_seq_length=278 \
	--train_batch_size=20 \
	--null_score_diff_threshold=7.0 \
	--doc_stride=128 \
	--do_lower_case \
	--do_predict \
	--do_train
	#--fp16 \
