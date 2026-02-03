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

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
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

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd",
            scale=(1.0, 1.0, 1.3),
        ),
    )

    # placeholder object for "rice cooker"
    rice_cooker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RiceCooker",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.10, 0.50, 0.98], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
            scale=(1.0, 1.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
    )

    # two Franka robots (bimanual setup)
    robot_left: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/RobotLeft",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.55, 0.70, 0.76),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    robot_right: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/RobotRight",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.70, 0.76),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
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
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
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
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
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
    success = DoneTerm(func=mdp.task_done_cook_rice)


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
            "asset_cfg": SceneEntityCfg("rice_cooker"),
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
    """Minimal DexVLA env for evaluation (bimanual GR1T2 + cam_high)."""

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

        # actions: joint position for bimanual franka arms (14-dim total)
        self.actions.left_arm_action = JointPositionActionCfg(
            asset_name="robot_left",
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            scale=1.0,
            use_default_offset=True,
            preserve_order=True,
        )
        self.actions.right_arm_action = JointPositionActionCfg(
            asset_name="robot_right",
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            scale=1.0,
            use_default_offset=True,
            preserve_order=True,
        )
