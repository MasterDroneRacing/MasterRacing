from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.core.utils.stage as stage_utils
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from isaaclab.sensors import SensorBase
from isaaclab.sensors import ImuData
from .imu_noise import *

if TYPE_CHECKING:
    # from isaaclab.sensors import ImuCfg
    from .imu_cfg import HifiImuCfg


class HifiImu(SensorBase):
    """The High-fidelity Inertia Measurement Unit (IMU) sensor.
    NOTE: Add noise models and other features in the future.
    """

    cfg: HifiImuCfg
    """The configuration parameters."""

    def __init__(self, cfg: HifiImuCfg):
        """Initializes the Imu sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = ImuData()
        # Set noise parameters
        self.noise_cfg = cfg.noise

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Imu sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> ImuData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_instances(self) -> int:
        return self._view.count

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.quat_w[env_ids] = 0.0
        self._data.lin_vel_b[env_ids] = 0.0
        self._data.ang_vel_b[env_ids] = 0.0
        self._data.lin_acc_b[env_ids] = 0.0
        self._data.ang_acc_b[env_ids] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        self._dt = dt
        # execute updating
        # force compute, but updata data according to update period
        super().update(dt, True)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the imu prim is not a RigidBodyPrim
        """
        # Initialize parent class
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is a rigid prim
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # check if it is a RigidBody Prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
        else:
            raise RuntimeError(f"Failed to find a RigidBodyAPI for the prim paths: {self.cfg.prim_path}")

        # Create internal buffers
        self._initialize_buffers_impl()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # check if self._dt is set (this is set in the update function)
        # if not hasattr(self, "_dt"):
        #     raise RuntimeError(
        #         "The update function must be called before the data buffers are accessed the first time."
        #     )
        # # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        # obtain the poses of the sensors
        pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = math_utils.convert_quat(quat_w, to="wxyz")

        # store the poses
        self._data.pos_w[env_ids] = pos_w + math_utils.quat_rotate(quat_w, self._offset_pos_b[env_ids])
        self._data.quat_w[env_ids] = math_utils.quat_mul(quat_w, self._offset_quat_b[env_ids])

        # get the offset from COM to link origin
        com_pos_b = self._view.get_coms().to(self.device).split([3, 4], dim=-1)[0]
        
        # obtain the velocities of the link COM
        lin_vel_w, ang_vel_w = self._view.get_velocities()[env_ids].split([3, 3], dim=-1)
        # if an offset is present or the COM does not agree with the link origin, the linear velocity has to be
        # transformed taking the angular velocity into account
        lin_vel_w += torch.linalg.cross(
            ang_vel_w, math_utils.quat_rotate(quat_w, self._offset_pos_b[env_ids] - com_pos_b[env_ids]), dim=-1
        )

        # obtain the acceleration of the link COM
        lin_acc_w, ang_acc_w = self._view.get_accelerations()[env_ids].split([3, 3], dim=-1)
        # # if an offset is present or the COM does not agree with the link origin, the linear acceleration has to
        # # be transformed taking the angular velocity and acceleration into account
        lin_acc_w += torch.cross(ang_acc_w, math_utils.quat_rotate(quat_w, self._offset_pos_b[env_ids] - com_pos_b[env_ids]), dim=-1) + torch.cross(
            ang_vel_w, torch.cross(ang_vel_w, math_utils.quat_rotate(quat_w, self._offset_pos_b[env_ids] - com_pos_b[env_ids]), dim=-1), dim=-1
        )

        # add gravity
        lin_acc_w += self._gravity_bias_w[env_ids]
        
        # add noise
        update_dt = self.cfg.update_period
        if self.noise_cfg.add_noise:
            M = pos_w.shape[0]
            if torch.any(self._accl_bias_corr == 0):
                raise ValueError("accl_bias_corr cannot have any zeros")
            if torch.any(self._gyro_bias_corr == 0):
                raise ValueError("gyro_bias_corr cannot have any zeros")
            
            acc_wn = white_noise(self._accl_std, M)
            acc_b_dyn = noise_b_dyn(self._prev_accl_dyn_noise[env_ids], self._accl_bias_corr, self._accl_bias_dynamic, update_dt)
            acc_rrw = noise_rrw(self._prev_vrrw_noise[env_ids], self._vrrw_std, update_dt)
            lin_acc_w += acc_wn + acc_b_dyn + acc_rrw + self._accl_bias_static

            gyro_wn = white_noise(self._gyro_std, M)
            gyro_b_dyn = noise_b_dyn(self._prev_gyro_dyn_noise[env_ids], self._gyro_bias_corr, self._gyro_bias_dynamic, update_dt)
            gyro_rrw = noise_rrw(self._prev_arrw_noise[env_ids], self._arrw_std, update_dt)
            ang_vel_w += gyro_wn + gyro_b_dyn + gyro_rrw + self._gyro_bias_static

            # store dynamic noise and rrw noise
            self._prev_accl_dyn_noise[env_ids] = acc_b_dyn
            self._prev_vrrw_noise[env_ids] = acc_rrw
            self._prev_gyro_dyn_noise[env_ids] = gyro_b_dyn
            self._prev_arrw_noise[env_ids] = gyro_rrw
            
        # store the velocities
        self._data.lin_vel_b[env_ids] = math_utils.quat_rotate_inverse(self._data.quat_w[env_ids], lin_vel_w)
        self._data.ang_vel_b[env_ids] = math_utils.quat_rotate_inverse(self._data.quat_w[env_ids], ang_vel_w)
        # store the accelerations
        self._data.lin_acc_b[env_ids] = math_utils.quat_rotate_inverse(self._data.quat_w[env_ids], lin_acc_w)
        self._data.ang_acc_b[env_ids] = math_utils.quat_rotate_inverse(self._data.quat_w[env_ids], ang_acc_w)

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        # data buffers
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.quat_w[:, 0] = 1.0
        self._data.lin_vel_b = torch.zeros_like(self._data.pos_w)
        self._data.ang_vel_b = torch.zeros_like(self._data.pos_w)
        self._data.lin_acc_b = torch.zeros_like(self._data.pos_w)
        self._data.ang_acc_b = torch.zeros_like(self._data.pos_w)

        # store sensor offset transformation
        self._offset_pos_b = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        self._offset_quat_b = torch.tensor(list(self.cfg.offset.rot), device=self._device).repeat(self._view.count, 1)
        # set gravity bias
        self._gravity_bias_w = torch.tensor(list(self.cfg.gravity_bias), device=self._device).repeat(
            self._view.count, 1
        )

        # set sensor noise
        if self.noise_cfg.add_noise:
            # white noise
            self._gyro_std = torch.tensor(list(self.noise_cfg.g_std), device=self._device)
            self._accl_std = torch.tensor(list(self.noise_cfg.a_std), device=self._device)
            # static bias (static after initialization)
            self._gyro_bias_static = torch.tensor(list(self.noise_cfg.gb_sta), device=self._device)
            self._accl_bias_static = torch.tensor(list(self.noise_cfg.ab_sta), device=self._device)
            self._gyro_bias_static = noise_b_sta(self._gyro_bias_static, self._view.count)
            self._accl_bias_static = noise_b_sta(self._accl_bias_static, self._view.count)
            # random walk noise
            self._arrw_std = torch.tensor(list(self.noise_cfg.arrw_std), device=self._device)
            self._vrrw_std = torch.tensor(list(self.noise_cfg.vrrw_std), device=self._device)
            # bias instability
            self._gyro_bias_dynamic = torch.tensor(list(self.noise_cfg.gb_dyn), device=self._device)
            self._accl_bias_dynamic = torch.tensor(list(self.noise_cfg.ab_dyn), device=self._device)
            self._gyro_bias_corr = torch.tensor(list(self.noise_cfg.gb_corr), device=self._device)
            self._accl_bias_corr = torch.tensor(list(self.noise_cfg.ab_corr), device=self._device)
            # rrw and bias instability initialization
            self._prev_arrw_noise = torch.zeros(self._view.count, 3, device=self._device)
            self._prev_vrrw_noise = torch.zeros(self._view.count, 3, device=self._device)
            self._prev_gyro_dyn_noise = torch.tensor(list(self.noise_cfg.gb_dyn), device=self._device).repeat(self._view.count, 1)
            self._prev_accl_dyn_noise = torch.tensor(list(self.noise_cfg.ab_dyn), device=self._device).repeat(self._view.count, 1)


    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.acceleration_visualizer.set_visibility(True)
        else:
            if hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self._view is None:
            return
        # get marker location
        # -- base state
        base_pos_w = self._data.pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales
        default_scale = self.acceleration_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self._data.lin_acc_b.shape[0], 1)
        # get up axis of current stage
        up_axis = stage_utils.get_stage_up_axis()
        # arrow-direction
        quat_opengl = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(
                self._data.pos_w,
                self._data.pos_w + math_utils.quat_rotate(self._data.quat_w, self._data.lin_acc_b),
                up_axis=up_axis,
                device=self._device,
            )
        )
        quat_w = math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world")
        # display markers
        self.acceleration_visualizer.visualize(base_pos_w, quat_w, arrow_scale)
