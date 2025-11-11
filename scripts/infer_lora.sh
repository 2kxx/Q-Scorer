export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=./:$PYTHONPATH

python src/evaluate/iqa_eval.py \
	--model-path checkpoints/Qscorer_lora_5_1 \
	--model-base xxx/mplug-owl2-llama2-7b \
	--save-dir results/res_Qscorer_lora_5_1/ \
	--preprocessor-path ./preprocessor/ \
	--root-dir /xxx/ \
	--meta-paths /xxx/koniq/metas/test_koniq_2k.json \
				 /xxx/spaq/metas/test_spaq_2k.json \
				 /xxx/kadid10k/metas/test_kadid_2k.json \
				 /xxx/LIVE-WILD/metas/test_livew_1k.json \
				 /xxx/AGIQA3K/metas/test_agiqa_3k.json \
				 /xxx/csiq/metas/test_csiq_866.json \
