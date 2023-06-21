python ./src/inference.py \
--output_dir ./predictions/test \
--dataset_name ./data/test_dataset \
--model_run_name 0616_13:32:48_JH \
--eval_retrieval \
--do_predict \
--search_mode elastic \
--valid_elastic_dir retrieval/es_valid_top20.csv \
--test_elastic_dir retrieval/es_test_noun_top20.csv