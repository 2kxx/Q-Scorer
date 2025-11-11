export PYTHONPATH=./:$PYTHONPATH

res_dir=./results/res_Qscorer_lora_5_1
gt_dir=/xxx/

python src/evaluate/cal_plcc_srcc.py \
	--pred_paths $res_dir/test_koniq_2k.json \
				 $res_dir/test_spaq_2k.json \
				 $res_dir/test_kadid_2k.json \
				 $res_dir/test_livew_1k.json \
				 $res_dir/test_agiqa_3k.json \
				 $res_dir/test_csiq_866.json \
	--gt_paths  $gt_dir/koniq/metas/test_koniq_2k.json \
				$gt_dir/spaq/metas/test_spaq_2k.json \
				$gt_dir/kadid10k/metas/test_kadid_2k.json \
				$gt_dir/LIVE-WILD/metas/test_livew_1k.json \
				$gt_dir/AGIQA3K/metas/test_agiqa_3k.json \
				$gt_dir/csiq/metas/test_csiq_866.json \
		
