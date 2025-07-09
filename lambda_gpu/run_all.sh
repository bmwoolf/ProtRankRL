#!/bin/bash

python retrain_model.py --timesteps 100000 --model_name ppo_100k   | tee logs/ppo_100k.log
python retrain_model.py --timesteps 500000 --model_name ppo_500k   | tee logs/ppo_500k.log
python retrain_model.py --timesteps 1000000 --model_name ppo_1M    | tee logs/ppo_1M.log
python retrain_model.py --timesteps 2000000 --model_name ppo_2M    | tee logs/ppo_2M.log