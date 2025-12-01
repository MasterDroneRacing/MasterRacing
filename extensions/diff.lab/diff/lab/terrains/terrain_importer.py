from isaaclab.terrains import TerrainImporter as terrain_importer
from .terrain_generator import TerrainGenerator
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
import torch
import numpy as np
import trimesh
from pxr import UsdGeom
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from diff.lab.utils.utils import create_prim_from_mesh
from .trimesh.utils import make_plane
import warp
class TerrainImporter(terrain_importer):
    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore
        
        # create a dict of meshes
        self.diff_meshes = {}
        self.diff_warp_meshes = {}
        self.terrain_prim_paths = list()
        self.env_origins = None
        self.terrain_origins = None
        # private variables
        self._terrain_flat_patches = dict()

        #
        self.extras = {}

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = TerrainGenerator(cfg=self.cfg.terrain_generator, device=self.device)
            # obtain extra information
            if hasattr(terrain_generator, "extras"):
                if terrain_generator.cfg.curriculum:
                    gate_pose = torch.tensor(np.stack(terrain_generator.extras["gate_pose"]), device=self.device).reshape(terrain_generator.cfg.num_cols, terrain_generator.cfg.num_rows, -1, 7)
                    self.extras["gate_pose"] = gate_pose
                    next_gate_id = torch.tensor(terrain_generator.extras["next_gate_id"], device=self.device).reshape(terrain_generator.cfg.num_cols, terrain_generator.cfg.num_rows)
                    self.extras["next_gate_id"] = next_gate_id
                else:
                    gate_pose = torch.tensor(np.stack(terrain_generator.extras["gate_pose"]), device=self.device).reshape(terrain_generator.cfg.num_rows, terrain_generator.cfg.num_cols, -1, 7).transpose(0, 1)
                    self.extras["gate_pose"] = gate_pose
                    next_gate_id = torch.tensor(terrain_generator.extras["next_gate_id"], device=self.device).reshape(terrain_generator.cfg.num_rows, terrain_generator.cfg.num_cols).transpose(0, 1)
                    self.extras["next_gate_id"] = next_gate_id

    
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)    

    
    def import_ground_plane(self, key: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Add a plane to the terrain importer.

        Args:
            key: The key to store the mesh.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # check if key exists
        if key in self.diff_meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.diff_meshes.keys()}.")
        # create a plane
        mesh = make_plane(size, height=0.0, center_zero=True)
        # store the mesh
        self.diff_meshes[key] = mesh
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.diff_warp_meshes[key] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        # get the mesh
        ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size)
        ground_plane_cfg.func(self.cfg.prim_path, ground_plane_cfg)

    def import_mesh(self, key: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            key: The key to store the mesh.
            mesh: The mesh to import.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # check if key exists
        if key in self.diff_meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.diff_meshes.keys()}.")
        # store the mesh
        self.diff_meshes[key] = mesh
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.diff_warp_meshes[key] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        # get the mesh
        mesh = self.diff_meshes[key]
        mesh_prim_path = self.cfg.prim_path + f"/{key}"
        # import the mesh
        create_prim_from_mesh(
            mesh_prim_path,
            mesh,
            visual_material=self.cfg.visual_material,
            physics_material=self.cfg.physics_material,
        )

    def import_usd(self, key: str, usd_path: str):
        """Import a mesh from a USD file.

        We assume that the USD file contains a single mesh. If the USD file contains multiple meshes, then
        the first mesh is used. The function mainly helps in registering the mesh into the warp meshes
        and the meshes dictionary.

        Note:
            We do not apply any material properties to the mesh. The material properties should
            be defined in the USD file.

        Args:
            key: The key to store the mesh.
            usd_path: The path to the USD file.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # add mesh to the dict
        if key in self.diff_meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.diff_meshes.keys()}.")
        # add the prim path
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func(self.cfg.prim_path + f"/{key}", cfg)

        # traverse the prim and get the collision mesh
        # THINK: Should the user specify the collision mesh?
        mesh_prim = sim_utils.get_first_matching_child_prim(
            self.cfg.prim_path + f"/{key}", lambda prim: prim.GetTypeName() == "Mesh"
        )
        # check if the mesh is valid
        if mesh_prim is None:
            raise ValueError(f"Could not find any collision mesh in {usd_path}. Please check asset.")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        self.diff_meshes[key] = trimesh.Trimesh(vertices=vertices, faces=faces)
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.diff_warp_meshes[key] = convert_to_warp_mesh(vertices, faces, device=device)


    