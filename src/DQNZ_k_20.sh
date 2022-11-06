for ((i=1; i<= 3; i++));do
let "seed=$i*100"
python run.py --agt 6 \
--usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
--experience_replay_pool_size 5000 --episodes 500 --simulation_epoch_size 100 \
--run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
--goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
--warm_start 1 --warm_start_epochs 100 --mcts 1 \
--write_model_dir ./deep_dialog/checkpoints/Test_DDQ_k20_all_mcts_run$i \
--planning_steps 20 --torch_seed $seed --grounded 0 --boosted 1 --train_world_model 1
done
