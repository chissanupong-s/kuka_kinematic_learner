- [1. Data Generation](#1-data-generation)
  - [1.1. Command for Data generation using ISAAC LAB](#11-command-for-data-generation-using-isaac-lab)
  - [1.2. Command for Data generation using torch (Much Faster)](#12-command-for-data-generation-using-torch-much-faster)
- [2. Training Stage](#2-training-stage)
  - [2.1. Command for training one DOF](#21-command-for-training-one-dof)
  - [2.2. Command for simple MAML with Fixed at Q1-Q7](#22-command-for-simple-maml-with-fixed-at-q1-q7)
- [3. Tensorboard Visualise](#3-tensorboard-visualise)
- [4. Utility](#4-utility)
  - [4.1. Compute Actual Error from LOSS](#41-compute-actual-error-from-loss)
  - [4.2. Evaluate Model from any dataset](#42-evaluate-model-from-any-dataset)
  - [4.3. Check and Fix the dataset (only pt file)](#43-check-and-fix-the-dataset-only-pt-file)


## 1. Data Generation

### 1.1. Command for Data generation using ISAAC LAB
ISL -p generate_iiwa14_dynamic_dataset.py     --num_envs 100     --num_samples 1000     --output_csv data/5DOF/iiwa14_dynamic_dataset_100k.csv

**Reminder**
- Change the robot_configuration selection in the python file and save it before running the command
- change the output directory before run the command
- change the directory of the CLI to Original/scripts/data first

### 1.2. Command for Data generation using torch (Much Faster)

python3 datagen_pt.py \
--urdf_path (path to urdf) \    
--step_deg 15   \
--batch_size 65536   \
--output_csv (path, "" for no csv)   \
--output_pt_prefix (path directory + prefix_name, "" for no pt) \
--pt_shard_size (data per pt part) \
--cuda

**Reminder**
- source ./fk_urdf_env/bin/activate  activate environment first
- change directory to ext_script
- change urdf path
- check output path
  
## 2. Training Stage

### 2.1. Command for training one DOF
ISL -p train_with_eval.py   \\
--csv (path to .csv or .pt)   \\
--mode fk   \\
--epochs 200   \\
--batch_size 8192   \\
--hidden_dim 1024   \\
--lr 5e-4   \\
--device cuda   \\
--num_workers 8   \\
--weight_decay 1e-5   \\
--log_dir .*/runs/**(path)***

**Reminder**
- Select the **csv/pt** file to be correct to your desired dof e.g.for 6 dof the csv path is ../6DOF/iiwa14_dynamic_dataset_100k.csv
- changing the mode fk, ik to compute the forward kinematic or inverse kinematics
- change the log directory in --log_dir to correct to your desired DOF and name the folder e.g. 6 DOF is runs/6DOF/pose_with_orientation_2 for 2nd attempt
- change the directory of the CLI to Original/scripts/data/Training first




### 2.2. Command for simple MAML with Fixed at Q1-Q7
ISL -p train_MAML_avg.py \
  --mode ik \
  --task_5dof /path/to/5DOF.pt \
  --task_6dof /path/to/6DOF.pt \
  --task_7dof /path/to/7DOF.pt \
  --init_ckpt_5dof /path/to/ik_5dof_best.pt \
  --init_ckpt_6dof /path/to/ik_6dof_best.pt \
  --init_ckpt_7dof /path/to/ik_7dof_best.pt \
  --meta_iters 2000 \
  --inner_steps 20 \
  --inner_lr 1e-3 \
  --meta_lr 1e-4 \
  --batch_size 8192 \
  --device cuda

ISL -p adapt_meta_model.py   --meta_checkpoint '/home/wish/isaaclab/kuka_14_kinematic_learner/original/ext_script/scripts/train/runs/meta_2/ik_reptile_5_6_7dof/meta_ik_5_6_7dof_best.pt'     --data '/home/wish/isaaclab/kuka_14_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt'       --support_size 20000   --query_size 50000   --batch_size 8192   --inner_lr 5e-3   --steps "0,5,10,20,50,200,500"   --num_workers 8   --device cuda




**Reminder**
- Select the mode to be trained, ik or fk
- init checkpiont from the single task 7DOF best pt
- The --csv_5,6,7 is for the dataset path of 5DOF, 6DOF, 7DOF respectively
- change the log directory in --log_dir to correct to your desired DOF and name the folder which is runs/567DOF/multitask_learning_2000 for 2000 epochs
- change the directory of the CLI to Original/scripts/data/Training first

## 3. Tensorboard Visualise
ISL -p -m tensorboard.main --logdir (path) --port 6005

**Reminder**
- Change the output file for pulling into the tensorboard
- set the port is optionally for remember





## 4. Utility
### 4.1. Compute Actual Error from LOSS
ISL -p compute_error.py \
  --path (path to csv or pt) \
  --fk_loss 0.028 \
  --ik_loss 0.05

### 4.2. Evaluate Model from any dataset
ISL -p eval_model.py \
  --csv (path to dataset.csv/.pt)  \
  --checkpoint (path to model.pt)  \
  --batch_size 8192 \
  --device cuda

### 4.3. Check and Fix the dataset (only pt file)

python3 check_dataset.py --path (path to dataset.pt)