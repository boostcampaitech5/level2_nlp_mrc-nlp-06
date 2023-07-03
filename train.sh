python ./src/train.py \
--output_dir ./models \
--preprocessing False \
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
--model_name_or_path "klue/roberta-large" \
--use_add_data False \
--clf_layer linear # default: linear | other options: [lstm, bi_lstm, mlp, SDS_cnn] \
--distill True \
--distill_dir ./models/[add_data]0622_23:30:08_JH \
