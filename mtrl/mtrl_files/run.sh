mkdir -p ./trainlogs

#### MT10 ####

### use an inner gradient descent to solve the optimization (cagrad_method=cagrad), the default method ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=cagrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.cagrad_method=cagrad agent.builder.agent_cfg.cagrad_c=0.9 > trainlogs/mt10_cagrad-c9e-1_sd1.log 2>&1 &

### fast method, cagrad_method=cagrad_fastk, where k is the number of gradients we calculate ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=cagrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.cagrad_method=cagrad_fast4 agent.builder.agent_cfg.cagrad_c=0.9 > trainlogs/mt10_cagrad_fast-c9e-1_sd1.log 2>&1 &


#### MT50 ####

### use an inner gradient descent to solve the optimization (cagrad_method=cagrad), the default method ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt50 agent=cagrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=50 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.cagrad_method=cagrad agent.builder.agent_cfg.cagrad_c=0.5 > trainlogs/mt50_cagrad-c5e-1_sd1.log 2>&1 &

### fast method, cagrad_method=cagrad_fastk, where k is the number of gradients we calculate ###
export CUDA_VISIBLE_DEVICES=0 && python -u main.py setup=metaworld env=metaworld-mt10 agent=cagrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=1 replay_buffer.batch_size=1280 agent.multitask.num_envs=50 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.cagrad_method=cagrad_fast8 agent.builder.agent_cfg.cagrad_c=0.5 > trainlogs/mt50_cagrad_fast-c5e-1_sd1.log 2>&1 &
