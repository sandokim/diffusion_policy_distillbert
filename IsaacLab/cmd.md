python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0
python scripts/reinforcement_learning/skrl/play.py --task=Isaac-Ant-v0

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard

./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device keyboard --dataset_file ./datasets/dataset_skillgen.hdf5 --num_demos 2

# franka