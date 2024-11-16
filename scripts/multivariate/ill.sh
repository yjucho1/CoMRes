if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=36
model_name=PathFormer

root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom
unlabeled=1
unlabel_type=interpolate
consistency=1

for random_seed in 22 123 456 99 999
do
for pred_len in 24 36 48 60
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_size_list 16 6 2 24 \
      --num_nodes 7 \
      --layer_nums 1 \
      --k 2\
      --d_model 16 \
      --d_ff 128 \
      --train_epochs 100\
      --closs_epoch 20\
      --patience 30\
      --lradj 'constant'\
      --unlabeled $unlabeled\
      --unlabel_type $unlabel_type\
      --consistency $consistency\
      --batch_size 32 --learning_rate 0.0025 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$unlabeled'_'$unlabel_type.log
done
done
