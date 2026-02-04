from dataclasses import MISSING
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.envs.mdp as base_mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
LOCAL_PROPS_DIR = os.environ.get(
    "LOCAL_PROPS_DIR",
    "/home/hyeseong/diffusion_policy/assets/Props",
)

COBOTTA_USD_PATH = os.environ.get(
    "COBOTTA_USD_PATH",
    "/home/hyeseong/diffusion_policy/assets/Robots/cobotta_isaac.usd",
)
COBOTTA_ROOT_PRIM = os.environ.get(
    "COBOTTA_ROOT_PRIM",
    "/base_link",
)
COBOTTA_BASE_Z = float(os.environ.get("COBOTTA_BASE_Z", "0.76"))

ZED_X_USD_PATH = os.environ.get(
    "ZED_X_USD_PATH",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc",
)
ZED_X_CAMERA_PRIM = os.environ.get(
    "ZED_X_CAMERA_PRIM",
    "{ENV_REGEX_NS}/cam_high/base_link/ZED_X/CameraLeft",
)
ZED_X_ROOT_PRIM = os.environ.get(
    "ZED_X_ROOT_PRIM",
    "{ENV_REGEX_NS}/cam_high",
)


def _parse_tuple(env_key: str, default: tuple[float, float, float]):
    value = os.environ.get(env_key, None)
    if value is None:
        return default
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return default

from . import mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene config with bimanual robot, a table, a rice-cooker placeholder, and cam_high."""

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # table (local asset)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{LOCAL_PROPS_DIR}/table.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # block (peg block)
    peg_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PegBlock",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.10, 0.50, 0.98], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{LOCAL_PROPS_DIR}/block.usd",
            scale=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
    )

    # pegboard placeholder can be added later

    # two Cobotta robots (bimanual setup)
    robot_left: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotLeft",
        articulation_root_prim_path=COBOTTA_ROOT_PRIM,
        spawn=UsdFileCfg(
            usd_path=COBOTTA_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.55, 0.70, COBOTTA_BASE_Z),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
                "joint_gripper": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint_gripper"],
                stiffness=200.0,
                damping=40.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=20.0,
                velocity_limit_sim=1.0,
        ),
        },
    )

    robot_right: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotRight",
        articulation_root_prim_path=COBOTTA_ROOT_PRIM,
        spawn=UsdFileCfg(
            usd_path=COBOTTA_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.70, COBOTTA_BASE_Z),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.0, # joint_3 limits [0.314, 2.443]
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
                "joint_gripper": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint_gripper"],
                stiffness=200.0,
                damping=40.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=20.0,
                velocity_limit_sim=1.0,
        ),
        },
    )

    # ZED_X camera USD (spawn as asset, then attach sensor to camera prim)
    zed_x = AssetBaseCfg(
        prim_path=ZED_X_ROOT_PRIM,
        init_state=AssetBaseCfg.InitialStateCfg(
            # default: higher top-down view centered on table (use table pos as reference)
            pos=_parse_tuple("ZED_X_POS", (0.0, 0.55, 1.8)),
            rot=_parse_tuple("ZED_X_ROT", (0.5, -0.5, 0.5, 0.5)),
        ),
        spawn=UsdFileCfg(
            usd_path=ZED_X_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False, # 손목 링크 아래에 또 다른 RigidBody가 달려있기 때문이야. 그래서 ZED_X 자산의 RigidBody를 비활성화했어. 이제 계층 충돌이 없어짐.
                disable_gravity=True,
            ),
        ),
    )

    # cam_high (single camera) - assumes ZED_X.usd contains a Camera prim at ZED_X_CAMERA_PRIM
    cam_high = CameraCfg(
        prim_path=ZED_X_CAMERA_PRIM,
        update_period=0.0333,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    left_gripper_action: ActionTerm = MISSING
    right_gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # joint positions (bimanual arms only)
        qpos_left = ObsTerm(
            func=base_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot_left",
                    joint_names=[
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                        "joint_gripper",
                    ],
                )
            },
        )

        qpos_right = ObsTerm(
            func=base_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot_right",
                    joint_names=[
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                        "joint_gripper",
                    ],
                )
            },
        )

        cam_high = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_high"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_done_peg_block)


@configclass
class EventsCfg:
    """Events config."""

    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
        },
    )
    randomize_cooker = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("peg_block"),
        },
    )


@configclass
class RewardsCfg:
    """Empty rewards config (no rewards)."""
    pass


@configclass
class CommandsCfg:
    """Empty commands config (no commands)."""
    pass


@configclass
class CurriculumCfg:
    """Empty curriculum config (no curriculum)."""
    pass


@configclass
class DexVLAEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal DexVLA env for evaluation (bimanual Franka + cam_high)."""

    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers (empty configs)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # basic sim settings
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 100
        self.sim.render_interval = 2

        # actions: IK (6DoF) + gripper (1DoF) per arm => 14-dim total
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_left",
            joint_names=["joint_[1-6]"],
            body_name="gripper_base",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_right",
            joint_names=["joint_[1-6]"],
            body_name="gripper_base",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        self.actions.left_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot_left",
            joint_names=["joint_gripper"],
            open_command_expr={"joint_gripper": 0.04},
            close_command_expr={"joint_gripper": 0.0},
        )
        self.actions.right_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot_right",
            joint_names=["joint_gripper"],
            open_command_expr={"joint_gripper": 0.04},
            close_command_expr={"joint_gripper": 0.0},
        )
