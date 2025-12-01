# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/02/27     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************

from __future__ import annotations
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from diff.lab.terrains import TerrainImporterCfg as diff_TerrainImporterCfg
from diff.lab_tasks.tasks.racing.terrains import *
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from diff.lab_assets.quadcopter import DRONE_CFG, DRONE_NO_COLLIDER_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, patterns, TiledCameraCfg
import diff.lab_tasks.tasks.racing.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from diff.lab.envs.manager_based_diff_rl_env_cfg import ManagerBasedDiffRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from diff.lab.managers import LossTermCfg as LossTerm
from diff.lab.controllers.controller_diff_cfg import CTBRControllerCfg

import os
STAGE = int(os.environ.get("TRAINING_STAGE", 0)) # 0: pre-training, 1: training, 2: testing

@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = diff_TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator = RacingComplexTerrainCfg,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        )
    if STAGE == 0:
        robot: ArticulationCfg = DRONE_NO_COLLIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")    # type: ignore
    else:
        contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/body", history_length=3, track_air_time=True, debug_vis=False)
        robot: ArticulationCfg = DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")    # type: ignore

    sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )


    front_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera",
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.01, 0.0, 0.0),
            rot=(0.991, 0, -0.131, 0.),
            convention="world"
        ),
        attach_yaw_only=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            width=96,
            height=72,
            intrinsic_matrix=[388.963 / (640 / 96), 0.0, 317.04 / (640 / 96), 0.0, 388.963 / (480 / 72), 241.99 / (480 / 72), 0.0, 0.0, 1.0],
        ),
        data_types=["distance_to_image_plane"],
        max_distance=10.0,
        depth_clipping_behavior="max"
    )

@configclass
class CommandsCfg:
    next_gate_pose = mdp.RacingCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        consecutive_commands=True,
        noise_ranges=mdp.RacingCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.1, 0.1),
            pos_z=(-0.1, 0.1),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.1, 0.1)
        ) if (STAGE == 0 or STAGE == 1) else mdp.RacingCommandCfg.Ranges(
            pos_x=(-0.5, 0.5),
            pos_y=(-0.5, 0.5),
            pos_z=(-0.5, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-0.5, 0.5)
        ),
        add_noise= (STAGE != 0),
        update_threshold=0.35
    )

@configclass
class ActionsCfg:
    force_torque = mdp.DiffActionCfg(asset_name="robot", 
                                     command_type="CTBRController", 
                                     controller_cfg=CTBRControllerCfg(
                                         rate_gain_p=[35, 35, 35],
                                         rate_gain_i=[0.0, 0.0, 0.0],
                                         rate_gain_d=[0.0005, 0.0005, 0.0003],
                                         body_rate_bound=[-6, 6],
                                         thrust_ctrl_delay=0.03,
                                         torque_ctrl_delay=(0.03, 0.03, 0.03)
                                         ),
                                         random_drag=True,
                                         action_lag=1)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.modified_base_lin_vel, params={"add_noise": True})
        base_orientation = ObsTerm(func=mdp.base_orientation_r, params={"add_noise": True})
        target_cmd = ObsTerm(func=mdp.modified_generated_commands, params={"command_name": "next_gate_pose"})
        last_action = ObsTerm(func=mdp.modified_last_action, params={"action_name": "force_torque"})
        image = ObsTerm(func=mdp.depth_image, params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "distance_to_image_plane", "add_noise": True})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.modified_base_lin_vel, params={"add_noise": False})
        base_orientation = ObsTerm(func=mdp.base_orientation_r, params={"add_noise": False})
        target_cmd = ObsTerm(func=mdp.modified_generated_commands_gt, params={"command_name": "next_gate_pose"})
        last_action = ObsTerm(func=mdp.modified_last_action, params={"action_name": "force_torque"})
        image = ObsTerm(func=mdp.depth_image, params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "distance_to_image_plane", "add_noise": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    critic: CriticCfg = CriticCfg()

    @configclass
    class AuxiliaryCfg(ObsGroup):
        cross_obs = ObsTerm(func=mdp.cross_obs, params={"reward_name": "success_cross"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    auxiliary: AuxiliaryCfg = AuxiliaryCfg()
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_racing,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5),
                "z":(-0.5, 0.5), 
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.7, 0.7)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_articulation_mass_and_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.02, 0.02),
            "mass_operation": "add",
            "inertia_distribution_params": (0.9, 1.1),
            "inertia_operation": "scale"
        },
    )

    randomize_rate_controller_gain = EventTerm(
        func=mdp.randomize_rate_controller_gain_and_thrust_delay,
        mode="startup",
        params={
            "action_name": "force_torque",
            "pid_scale_factor": (0.9, 1.1),
            "thrust_delay_scale_factor": (0.8, 1.3)
        }
    )

    reset_terrain = EventTerm(
        func=mdp.reset_terrain_period,
        interval_range_s = (0.03 * 24 * 5000, 0.03 * 24 * 5000),
        mode="interval",
    )
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="body"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 2.0),
    #     params={"velocity_range": {
    #         "x": (-0.0, 0.0), 
    #         "y": (-0.0, 0.0),
    #         "z": (-0.0, 0.0),
    #         }},
    # )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    if STAGE == 0:
        outofbound = DoneTerm(func=mdp.out_of_bound,params={"bounds":(0.00, 10.0)})
    else:
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"), "threshold": 1.0},
        )
        bad_pose = DoneTerm(
            func=mdp.bad_pose,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.racing_terrain_levels, params={
        "cmd_name": "next_gate_pose",
        "move_on_threshold": 3,
        "move_down_threshold": 2,
    })

    if STAGE == 1:
        command_noise_level = CurrTerm(func=mdp.racing_cmd_noise_levels, params={
            "cmd_name": "next_gate_pose",
            "enhance_threshold": 4,
            "decay_threshold": 3,
            "enhance_percent": 0.01,
            "decay_percent": 0.02,
        })

@configclass
class RewardsCfg:
    progress_rewards = RewTerm(
        func=mdp.progress_reward_mine, 
        weight=1.0,
        params={"command_name": "next_gate_pose"}
    )
    
    # penalty on command body rate
    command_bodyrate_penalty = RewTerm(
        func=mdp.command_body_rate_penalty,
        weight=-0.02 if STAGE == 0 else -0.1,
        params={"action_name": "force_torque"}
    )
    # penalty on command rate
    action_rate = RewTerm(
        func=mdp.command_rate_penalty,
        weight=-0.01 if STAGE == 0 else -0.05,
        params={"action_name": "force_torque"})
   
    if STAGE == 0:
        collision_penalty = RewTerm(
            func=mdp.collision_penalty_custom, 
            weight=-50.0,
        )
    else:
        collision_penalty = RewTerm(
            func=mdp.undesired_contacts, 
            weight=-100.0,
            params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
            "threshold": 1.0,})
        
    perception_reward = RewTerm(
        func=mdp.perception_reward,
        weight=0.1,
        params={"command_name": "next_gate_pose"}
    )
    success_cross = RewTerm(
        func=mdp.success_cross,
        weight=10.0 if STAGE == 0 else 20.0,
        params={"command_name": "next_gate_pose", "threshold": 0.35}
    )
    ####
    if STAGE == 1:
        bad_pose_penalty = RewTerm(
            func=mdp.penalize_bad_pose,
            weight=-30.0,
        )

@configclass
class LossesCfg:
    # move_towards_goal = LossTerm(
    #     func=mdp.racing_target_diff,
    #     weight=1.0,
    #     params={"command_name": "next_gate_pose"},
    #     use_diff_states=True,
    # )
    move_towards_goal = LossTerm(
        func=mdp.racing_target_diff,
        weight=1.0,
        params={"command_name": "next_gate_pose"},
        use_diff_states=True,
    )
    falling = LossTerm(
        func=mdp.racing_vel_diff,
        weight=0.05,
        use_diff_states=True,
    )
    falling_speed = LossTerm(
        func=mdp.racing_falling_diff,
        weight=0.5,
        use_diff_states=True,
    )

@configclass
class QuadcopterRacingCTBREnvCfg(ManagerBasedDiffRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    losses: LossesCfg = LossesCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:

        """Post initialization."""
        self.is_differentiable_physics = False
        
        # general settings
        self.decimation = 3
        self.episode_length_s = 6.0 if STAGE != 2 else 8.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing=True

        if hasattr(self.scene, "terrain"):
            self.sim.physics_material = self.scene.terrain.physics_material

        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.scene, "front_camera", None) is not None:
            self.scene.front_camera.update_period = 0.04

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False