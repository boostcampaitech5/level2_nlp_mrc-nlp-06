python ./src/inference.py \
--output_dir ./predictions/eval \
--dataset_name ./data/train_dataset \
--model_run_name 0620_03:49:08_GE \
--do_eval  \
--eval_retrieval \
--search_mode basic \
--valid_elastic_dir es_valid_top40.csv \
--test_elastic_dir es_test_top40.csv