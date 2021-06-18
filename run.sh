## preprocessing
# monolingual
python preprocess.py --setting en
python preprocess.py --setting zh
# bilingual
python preprocess.py --setting en_zh
# cross lingual training set: only use 10% data for target language
python preprocess.py --setting en2zh
python preprocess.py --setting zh2en
# generate data for mix language training
python preprocess.py --setting en --pretraining_prefix en2zh_trainsfer
python preprocess.py --setting zh --pretraining_prefix zh2en_trainsfer


## training:
# t5 zh
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path google/mt5-small \
--do_train \
--do_eval \
--train_file data/preprocessed/zh_train.json \
--validation_file data/preprocessed/zh_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 8 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--save_steps 2000 \
--output_dir save/zh_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3
# t5 en
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path google/mt5-small \
--do_train \
--do_eval \
--train_file data/preprocessed/en_train.json \
--validation_file data/preprocessed/en_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 8 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--save_steps 2000 \
--output_dir save/en_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3
# t5 en_zh
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path google/mt5-small \
--do_train \
--do_eval \
--train_file data/preprocessed/en_zh_train.json \
--validation_file data/preprocessed/en_zh_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 8 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--save_steps 2000 \
--output_dir save/en_zh_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3

# t5 en2zh mix language pretraining
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path google/mt5-small \
--do_train \
--do_eval \
--train_file data/preprocessed/en2zh_trainsferen_train.json \
--validation_file data/preprocessed/en2zh_trainsferen_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 8 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--save_steps 2000 \
--output_dir save/mix_en2zh_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3

# t5 zh2en mix language pretraining
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path google/mt5-small \
--do_train \
--do_eval \
--train_file data/preprocessed/zh2en_trainsferzh_train.json \
--validation_file data/preprocessed/zh2en_trainsferzh_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 8 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--save_steps 2000 \
--output_dir save/mix_zh2en_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3


# Note: before finetuning, please delete the files that might cache the training status of the pre-trained model
cd save/mix_en2zh_mt5_5e-4
rm -rf train*
rm -rf checkpoin*

# t5 en2zh finetuning
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path save/mix_en2zh_mt5_5e-4 \
--do_train \
--do_eval \
--train_file data/preprocessed/en2zh_train.json \
--validation_file data/preprocessed/en2zh_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 10 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--output_dir save/mix_transfer_en2zh_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3

# Note: before finetuning, please delete the files that might cache the training status of the pre-trained model
cd save/mix_zh2en_mt5_5e-4
rm -rf train*
rm -rf checkpoin*

# t5 zh2en finetuning
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--model_name_or_path save/mix_zh2en_mt5_5e-4  \
--do_train \
--do_eval \
--train_file data/preprocessed/zh2en_train.json \
--validation_file data/preprocessed/zh2en_valid.json \
--learning_rate 5e-4  \
--num_train_epochs 10 \
--source_lang en_XX \
--target_lang en_XX \
--logging_steps 100 \
--output_dir save/mix_transfer_zh2en_mt5_5e-4 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 8 \
--overwrite_output_dir \
--predict_with_generate \
--fp16 \
--sharded_ddp zero_dp_3



# evaluation
# t5
CUDA_VISIBLE_DEVICES=6 python evaluate.py --model_path save/zh_mt5_5e-4 --setting zh --reference_file_path data/zh_test.json --save_prefix t5_
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path save/en_mt5_5e-4 --setting en --reference_file_path data/en_test.json --save_prefix t5_
CUDA_VISIBLE_DEVICES=6 python evaluate.py --model_path save/en_zh_mt5_5e-4 --setting en_zh --reference_file_path data/zh_test.json__data/en_test.json --save_prefix t5_
CUDA_VISIBLE_DEVICES=2 python evaluate.py --model_path save/mix_transfer_en2zh_mt5_5e-4 --setting en2zh --reference_file_path data/zh_test.json --save_prefix mix_t5_
CUDA_VISIBLE_DEVICES=3 python evaluate.py --model_path save/mix_transfer_zh2en_mt5_5e-4 --setting zh2en --reference_file_path data/en_test.json --save_prefix mix_t5_


# evaluate predictions
python evaluate.py --eval_mode eval_file --prediction_file_path result/zh_end2end_predictions.json --setting zh --reference_file_path data/zh_test.json
python evaluate.py --eval_mode eval_file --prediction_file_path result/en_end2end_predictions.json --setting en --reference_file_path data/en_test.json
python evaluate.py --eval_mode eval_file --prediction_file_path result/t5_en_zh_end2end_predictions.json --setting en --reference_file_path data/en_test.json --save_prefix bi_t5_
python evaluate.py --eval_mode eval_file --prediction_file_path result/t5_en_zh_end2end_predictions.json --setting zh --reference_file_path data/zh_test.json --save_prefix bi_t5_
