# model_name=TIGER-Lab/VLM2Vec-Full
# model_type=phi3_v
model_name=/mnt/data/user/tc_ai/klara/models/open_mllm/vlm2vec_qwen2vl/train-model
model_type=qwen2_vl
data_path=/home/jeeves/JJ_Projects/songjunjie/playground/mm_emb/evaluation/MMEB-eval
image_root=/home/jeeves/JJ_Projects/songjunjie/playground/mm_emb/evaluation/MMEB-eval/eval_images

CLASSIFICATION_DATAS="ImageNet-1K N24News HatefulMemes VOC2007 SUN397 Place365 ImageNet-A ImageNet-R ObjectNet Country211"
VISUAL_QUESTION_ANSWERING_DATAS="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA VizWiz GQA TextVQA"
INFORMATION_RETRIEVAL_DATAS="VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA OVEN FashionIQ EDIS Wiki-SS-NQ"
VISUAL_GROUNDING_DATAS="MSCOCO Visual7W-Pointing RefCOCO RefCOCO-Matching"

python eval.py \
  --model_name ${model_name} \
  --model_type ${model_type} \
  --encode_output_path outputs_$model_type \
  --num_crops 4 \
  --max_len 256 \
  --pooling last \
  --normalize True \
  --dataset_name ${data_path} \
  --image_dir ${image_root} \
  --subset_name ${VISUAL_QUESTION_ANSWERING_DATAS} \
  --dataset_split test \
  --per_device_eval_batch_size 32 \
