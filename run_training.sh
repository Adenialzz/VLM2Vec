output_dir=/data/vlm2vec_training

model_path=/mnt/data/user/tc_ai/klara/models/open_mllm/qwen25_vl_3b/train-model/
data_path=/data/data/MMEB-train
image_root=/data/data/MMEB-train  # img_path 里呆了 images/ 前缀，这里不用加

torchrun --nproc_per_node=1 --master_port=22447 --max_restarts=0 train.py \
 --model_name ${model_path} \
 --model_backbone qwen2_5_vl \
 --pooling last \
 --bf16 \
 --dataset_name $data_path \
 --subset_name ImageNet_1K N24News HatefulMemes InfographicsVQA ChartQA Visual7W VisDial CIRR NIGHTS WebQA MSCOCO \
 --num_sample_per_subset 50000 \
 --image_dir $image_root \
 --max_len 8192 --num_crops 4 --output_dir $output_dir --logging_steps 1 \
 --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
 --warmup_steps 200 --save_steps 1000 --normalize True \
 --temperature 0.02 --per_device_train_batch_size 8 \
 --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 \
 --lora --lora_r 16 \
 --dataset_split original \
 --remove_unused_columns False
 # --model_backbone qwen25_vl \

