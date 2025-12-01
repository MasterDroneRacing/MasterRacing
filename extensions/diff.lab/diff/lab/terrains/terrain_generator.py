from isaaclab.terrains import TerrainGenerator as TerrainGen, SubTerrainBaseCfg
import trimesh
import numpy as np
from isaaclab.utils.dict import dict_to_md5_hash
import os
from isaaclab.utils.io import dump_yaml, load_yaml
from scipy.spatial.transform import Rotation as R

class TerrainGenerator(TerrainGen):
    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        # copy the configuration
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")
        sub_terrain_gate_info_filename = os.path.join(sub_terrain_cache_dir, "gate_info.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            if os.path.exists(sub_terrain_gate_info_filename):
                if not hasattr(self, "extras"):
                    self.extras = {"gate_pose":[], "next_gate_id": []}  
                gate_info = load_yaml(sub_terrain_gate_info_filename)
                self.extras["gate_pose"].append(np.array(gate_info["gate_pose"]))
                self.extras["next_gate_id"].append(gate_info["next_gate_id"])
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        # meshes, origin = cfg.function(difficulty, cfg)
        val = cfg.function(difficulty, cfg)
        if len(val) == 2:
            meshes, origin = val
        else:
            meshes, origin, _extras = val
            if not isinstance(_extras, dict):
                raise ValueError("The extra information returned by terrain_cfg function is not a dict.")
            gate_pose = _extras["gate_pose"]
            next_gate_id = _extras["next_gate_id"]
            new_gate_pose = np.zeros((gate_pose.shape[0], 7))
            new_gate_pose[:,:3] = gate_pose[:,:3] - origin
            # Y-Z-X convention
            ori_quat = R.from_euler('YXZ', np.stack([gate_pose[:, 3], -gate_pose[:, 4], gate_pose[:, 5]], axis=1), degrees=True)
            transform_offset = R.from_euler('XYZ', [-90, -90, 0], degrees=True)
            quat = (ori_quat * transform_offset).as_quat()
            new_gate_pose[:, 3] = quat[:, -1]
            new_gate_pose[:, 4:] = quat[:, :-1] 
            if not hasattr(self, "extras"):
                self.extras = {"gate_pose":[], "next_gate_id": []}  
            self.extras["gate_pose"].append(new_gate_pose)
            self.extras["next_gate_id"].append(next_gate_id)

        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
            if hasattr(self, "extras"):
                _extras = {}
                _extras["gate_pose"] = new_gate_pose.tolist()
                _extras["next_gate_id"] = next_gate_id
                dump_yaml(sub_terrain_gate_info_filename, _extras)
        # return the generated mesh
        return mesh, origin


