model_name=models/Qwen3-8B
model_url=http://localhost:8000/v1/chat/completions

mkdir -p logs/$model_name
mkdir -p Figs/$model_name

# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AIME24 --dataset_path data/aime24.parquet --conf_type logits --max_new_tokens 3000 > logs/$model_name/AIME24_Main_logits.log
# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AIME24 --dataset_path data/aime24.parquet --conf_type verbal --max_new_tokens 3000 > logs/$model_name/AIME24_Main_verbal.log
# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AIME25 --dataset_path data/aime25/aime2025.jsonl --conf_type logits --max_new_tokens 3000 > logs/$model_name/AIME25_Main_logits.log
# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AIME25 --dataset_path data/aime25/aime2025.jsonl --conf_type verbal --max_new_tokens 3000 > logs/$model_name/AIME25_Main_verbal.log

# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AMC23 --dataset_path data/AMC/data/amc23-00000-of-00001.parquet --conf_type logits --max_new_tokens 3000 > logs/$model_name/AMC23_Main_logits.log
# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AMC23 --dataset_path data/AMC/data/amc23-00000-of-00001.parquet --conf_type verbal --max_new_tokens 3000 > logs/$model_name/AMC23_Main_verbal.log
# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AMC24 --dataset_path data/AMC/data/amc24-00000-of-00001.parquet --conf_type logits --max_new_tokens 3000 > logs/$model_name/AMC24_Main_logits.log
# python Uncertainty/calibration_main.py --model_name $model_name --dataset_name AMC24 --dataset_path data/AMC/data/amc24-00000-of-00001.parquet --conf_type verbal --max_new_tokens 3000 > logs/$model_name/AMC24_Main_verbal.log

python Uncertainty/calibration_main.py --model_name $model_name --dataset_name MATH-500 --dataset_path data/MATH-500/test.jsonl --model_url $model_url --conf_type logits --max_new_tokens 3000 > logs/$model_name/MATH-500_Main_logits.log
python Uncertainty/calibration_main.py --model_name $model_name --dataset_name MATH-500 --dataset_path data/MATH-500/test.jsonl --model_url $model_url --conf_type verbal --max_new_tokens 3000 > logs/$model_name/MATH-500_Main_verbal.log
