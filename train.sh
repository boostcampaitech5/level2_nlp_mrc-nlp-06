python ./src/train.py \
--output_dir ./models \
--num_train_epochs 1 \
--eval_steps 300 \
--save_strategy steps \
--save_steps 300 \
--evaluation_strategy steps \
--save_total_limit 2 \
--logging_steps 100 \
--do_train \
--do_eval \
--load_best_model_at_end True \
--metric_for_best_model exact_match \
--clf_layer lstm \
--model_name_or_path "klue/roberta-large"

python ./src/train.py \
--output_dir ./models \
--num_train_epochs 1 \
--eval_steps 300 \
--save_strategy steps \
--save_steps 300 \
--evaluation_strategy steps \
--save_total_limit 2 \
--logging_steps 100 \
--do_train \
--do_eval \
--load_best_model_at_end True \
--metric_for_best_model exact_match \
--model_name_or_path "klue/roberta-large"