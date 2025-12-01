"""Configuration for the quadcopters"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from diff.lab_assets import DIFFLAB_ASSETS_ARTICULATIONS_DIR
##
# Configuration
##

DRONE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{DIFFLAB_ASSETS_ARTICULATIONS_DIR}/drone_175_v8.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "m.*": 0.0,
        },
        joint_vel={
            "m1": 0.0,
            "m2": 0.0,
            "m3": 0.0,
            "m4": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=["m.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

DRONE_NO_COLLIDER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{DIFFLAB_ASSETS_ARTICULATIONS_DIR}/drone_175_v8.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "m.*": 0.0,
        },
        joint_vel={
            "m1": 0.0,
            "m2": 0.0,
            "m3": 0.0,
            "m4": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=["m.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the drone"""