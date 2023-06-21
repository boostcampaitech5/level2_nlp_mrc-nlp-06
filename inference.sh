python ./src/inference.py \
--output_dir ./predictions/test \
--dataset_name ./data/test_dataset \
--model_run_name 0620_22:16:26_GE \
--eval_retrieval \
--do_predict \
--search_mode elastic \
--valid_elastic_dir es_valid_top20.csv \
--test_elastic_dir es_test_noun_top20.csv