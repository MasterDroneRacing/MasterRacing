import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.utils import configclass

from .commands import UniformWorldPoseCommand, RacingCommand

@configclass
class WorldPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformWorldPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["cuboid"].scale = (0.25, 0.25, 0.25)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

@configclass
class RacingCommandCfg(CommandTermCfg):
    """Specify the next gate's position and yaw."""

    class_type: type = RacingCommand
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    add_noise: bool = False
    """whether add noise on the real gate pose"""

    consecutive_commands: bool = False
    """Whether to use consecutive two commands"""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""
        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]
    
    noise_ranges: Ranges = MISSING
    """Ranges for the noise on commands."""

    update_threshold: float = 0.1       # [m]
    """minimum distance threshold to decide whether update to next gate"""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    goal_pose_gt_visualizer_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose_gt")
    """The configuration for the goal pose ground truth visualization marker. Defaults to CUBOID_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)
    goal_pose_gt_visualizer_cfg.markers["cuboid"].scale = (0.25, 0.25, 0.25)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)
