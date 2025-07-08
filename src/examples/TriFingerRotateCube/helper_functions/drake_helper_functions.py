from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.all import RevoluteJoint
from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.geometry import (
    Cylinder,
    HalfSpace,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    Role,
    SceneGraph,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    DiscreteContactApproximation,
    MultibodyPlant,
    MultibodyPlant_,
)
from pydrake.systems.framework import DiagramBuilder

from .misc_helper_functions import get_src_folder_absolute_path


# ---------------- Utility functions to create MultibodyPlant from urdf ---------------- #
def setup_package_paths(
    parser: Parser, package_map: Dict[str, str], src_folder_abs_path: Path
) -> None:
    """Configure package paths for URDF model parsing.

    Adds package name to path mappings to the parser's package map.
    Paths are resolved relative to the source folder absolute path.

    Args:
        parser (Parser): URDF parser to configure
        package_map (Dict[str, str]): Mapping of package names to relative paths
        src_folder_abs_path (Path): Absolute path to the source folder

    Raises:
        ValueError: If package_map is empty or contains invalid paths
    """
    if not package_map:
        raise ValueError("Package map cannot be empty")

    for name, path in package_map.items():
        parser.package_map().Add(name, (src_folder_abs_path / path).as_posix())


def add_urdf_models(
    parser: Parser, urdfs: List[str], src_folder_abs_path: Path
) -> None:
    """Add URDF models to the parser.

    Loads URDF models from specified file paths, resolving them
    relative to the source folder absolute path.

    Args:
        parser (Parser): URDF parser to add models to
        urdfs (List[str]): List of URDF file paths
        src_folder_abs_path (Path): Absolute path to the source folder

    Raises:
        ValueError: If urdfs list is empty
        FileNotFoundError: If any URDF file cannot be found
    """
    if not urdfs:
        raise ValueError("URDF list cannot be empty")

    for urdf in urdfs:
        full_path = src_folder_abs_path / urdf
        if not full_path.exists():
            raise FileNotFoundError(f"URDF file not found: {full_path}")
        parser.AddModels(full_path.as_posix())


def add_ground_surface(
    plant: MultibodyPlant,
    static_friction_coeff: float = 0.5,
    dynamic_friction_coeff: float = 0.5,
    name: str = "world_ground_plane",
) -> None:
    """Add ground surface to the plant with friction.

    Raises:
        RuntimeError: If plant is already finalized.
    """
    if plant.is_finalized():
        raise RuntimeError("Cannot modify a finalized plant by adding ground surface.")

    halfspace_transform = RigidTransform()
    friction = CoulombFriction(static_friction_coeff, dynamic_friction_coeff)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        halfspace_transform,
        HalfSpace(),
        name,
        friction,
    )


def weld_links(
    plant: MultibodyPlant, welded_links: Optional[List[Dict[str, str]]]
) -> None:
    """Weld specified links to the world or to each other.

    Args:
        plant (MultibodyPlant): Multibody plant to modify
        welded_links (Optional[List[Dict[str, str]]]): Links to weld, where each dict contains:
            - 'parent_link': Name of the parent link
            - 'body_of_parent_link': Model instance of the parent link
            - 'child_link': Name of the child link to be welded
            - 'body_of_child_link': Model instance of the child link

    Raises:
        RuntimeError: If plant is already finalized.
    """
    if plant.is_finalized():
        raise RuntimeError("Cannot modify a finalized plant by welding links.")

    if not welded_links:
        return

    for welded_link in welded_links:
        X_WI = RigidTransform.Identity()
        plant.WeldFrames(
            plant.GetFrameByName(
                welded_link["parent_link"],
                plant.GetModelInstanceByName(welded_link["body_of_parent_link"]),
            ),
            plant.GetFrameByName(
                welded_link["child_link"],
                plant.GetModelInstanceByName(welded_link["body_of_child_link"]),
            ),
            X_WI,
        )


def add_joints(
    plant: MultibodyPlant, joint_specs: Optional[List[Dict[str, Any]]]
) -> None:
    """Add joints to the plant.

    Args:
        plant (MultibodyPlant): Multibody plant to modify
        joint_specs (Optional[List[Dict[str, Any]]]): Joints to add, where each dict contains:
            - 'name': Name of the joint
            - 'parent': Model instance name of the parent body
            - 'child': Model instance name of the child body
            - 'frame_on_parent': Name of the frame on the parent body
            - 'frame_on_child': Name of the frame on the child body
            - 'axis': Rotation axis for the revolute joint
            - 'pos_lower_limit': Lower position limit of the joint
            - 'pos_upper_limit': Upper position limit of the joint
            - 'damping': Damping coefficient for the joint

    Raises:
        RuntimeError: If plant is already finalized.
        KeyError: If required joint configuration keys are missing.
    """
    if plant.is_finalized():
        raise RuntimeError("Cannot modify a finalized plant by adding joints.")

    if not joint_specs:
        return

    for joint in joint_specs:
        # Optional: Add explicit key validation if needed
        required_keys = [
            "name",
            "parent",
            "child",
            "frame_on_parent",
            "frame_on_child",
            "axis",
            "pos_lower_limit",
            "pos_upper_limit",
            "damping",
        ]
        for key in required_keys:
            if key not in joint:
                raise KeyError(f"Missing required joint configuration key: {key}")

        plant.AddJoint(
            RevoluteJoint(
                name=joint["name"],
                frame_on_parent=plant.GetFrameByName(
                    joint["frame_on_parent"],
                    plant.GetModelInstanceByName(joint["parent"]),
                ),
                frame_on_child=plant.GetFrameByName(
                    joint["frame_on_child"],
                    plant.GetModelInstanceByName(joint["child"]),
                ),
                axis=joint["axis"],
                pos_lower_limit=joint["pos_lower_limit"],
                pos_upper_limit=joint["pos_upper_limit"],
                damping=joint["damping"],
            )
        )


def create_plant_from_urdfs(
    builder: DiagramBuilder,
    package_map: Dict[str, str],
    urdfs: List[str],
    sim_dt: float,
    welded_links: Optional[List[Dict[str, str]]] = None,
    joint_specs: Optional[List[Dict[str, Any]]] = None,
    meshcat: Optional[Meshcat] = None,
    show_collision: bool = False,
    visualizer_publish_period: float = -1,
) -> Tuple[MultibodyPlant, SceneGraph, MeshcatVisualizer, Optional[MeshcatVisualizer]]:
    """Create a multibody plant with specified URDF models and configurations.

    Args:
        builder (DiagramBuilder): Diagram builder for plant creation
        package_map (Dict[str, str]): Package name to path mapping
        urdfs (List[str]): List of URDF file paths
        sim_dt (float): Simulation time step
        welded_links (Optional[List[Dict[str, str]]], optional): Links to weld. Defaults to None.
        joint_specs (Optional[List[Dict[str, Any]]], optional): Specs of joints to add. Defaults to None.
        meshcat (Optional[Meshcat], optional): Meshcat instance. Defaults to None.
        show_collision (bool, optional): Show collision geometry. Defaults to False.
        visualizer_publish_period (float, optional): Visualizer publish period. Defaults to -1.

    Returns:
        Tuple containing MultibodyPlant, SceneGraph, MeshcatVisualizer, and optional MeshcatVisualizer
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, sim_dt)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)

    parser = Parser(plant)
    parser.SetAutoRenaming(True)

    src_folder_abs_path = get_src_folder_absolute_path()
    setup_package_paths(parser, package_map, src_folder_abs_path)
    add_urdf_models(parser, urdfs, src_folder_abs_path)

    # perform various operations on the plant
    weld_links(plant, welded_links)
    add_ground_surface(plant)
    add_joints(plant, joint_specs)

    plant.Finalize()

    visual_visualizer = None
    collision_visualizer = None

    if meshcat is not None:
        visual_visualizer, collision_visualizer = setup_drake_visualizers(
            builder, scene_graph, meshcat, show_collision, visualizer_publish_period
        )

    return plant, scene_graph, visual_visualizer, collision_visualizer


# ------------------------------------------------------------------------------------#


# ---------------- Utility functions to set up visualizers in Drake ---------------- #
def initialize_meshcat() -> Meshcat:
    """Initialize and prepare Meshcat visualizer.

    Returns:
        Configured Meshcat instance
    """
    meshcat = StartMeshcat()
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    return meshcat


def set_camera_position_in_meshcat(
    meshcat: Meshcat,
    camera_position: Optional[npt.NDArray[np.float64]] = None,
    camera_target: Optional[npt.NDArray[np.float64]] = None,
) -> None:
    """Configure default camera position.

    Args:
        meshcat (Meshcat): Meshcat visualizer to configure
        camera_position (npt.NDArray[np.float64]): Camera position in world coordinates
        camera_target (npt.NDArray[np.float64]): Camera target in world coordinates
    """
    if camera_position is None:
        camera_position = np.array([0.4, 0.4, 0.4])
    if camera_target is None:
        camera_target = np.zeros(3)
    meshcat.SetCameraPose(camera_position, camera_target)


def add_meshcat_visualizer(
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    meshcat: Meshcat,
    meshcat_visualizer_params: MeshcatVisualizerParams,
) -> MeshcatVisualizer:
    """Add visual visualizer to the diagram builder.

    Args:
        builder (DiagramBuilder): Diagram builder
        scene_graph (SceneGraph): Scene graph to visualize
        meshcat (Meshcat): Meshcat instance
        meshcat_visualizer_params (MeshcatVisualizerParams): Visualizer parameters

    Returns:
        Created MeshcatVisualizer
    """
    if builder.already_built():
        raise RuntimeError("Cannot add visualizer to an already built diagram")

    return MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_visualizer_params
    )


def setup_drake_visualizers(
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    meshcat: Meshcat,
    show_collision: bool = False,
    publish_period: float = -1,
) -> Tuple[MeshcatVisualizer, Optional[MeshcatVisualizer]]:
    """Set up Drake visualizer if enabled.

    Args:
        builder (DiagramBuilder): Diagram builder
        scene_graph (SceneGraph): Scene graph
        meshcat (Meshcat): Meshcat instance
        show_collision (bool): Show collision geometry
        publish_period (float): Period to publish visualizer updates. Defaults to -1.

    Returns:
        (MeshcatVisualizer, MeshcatVisualizer | None)
    """
    if publish_period == -1:
        raise RuntimeError("publish_period must be > 0")
    set_camera_position_in_meshcat(meshcat)

    collision_visualizer = None

    visual_params = MeshcatVisualizerParams(
        role=Role.kPerception, prefix="visual", publish_period=publish_period
    )
    collision_params = MeshcatVisualizerParams(
        role=Role.kProximity, prefix="collision", publish_period=publish_period
    )
    visual_visualizer = add_meshcat_visualizer(
        builder, scene_graph, meshcat, visual_params
    )

    if show_collision:
        collision_visualizer = add_meshcat_visualizer(
            builder, scene_graph, meshcat, collision_params
        )

    return visual_visualizer, collision_visualizer


def draw_frame_axes(
    meshcat: Meshcat,
    str_path: str,
    frame_quat: Optional[np.ndarray] = None,
    frame_pos: Optional[np.ndarray] = None,
    transparency: float = 1.0,
) -> None:
    if not meshcat.HasPath(f"{str_path}/x_axis"):
        meshcat.SetObject(
            f"{str_path}/x_axis",
            Cylinder(0.001, 0.05),
            Rgba(1.0, 0.0, 0.0, transparency),
        )
        meshcat.SetObject(
            f"{str_path}/y_axis",
            Cylinder(0.001, 0.05),
            Rgba(0.0, 1.0, 0.0, transparency),
        )
        meshcat.SetObject(
            f"{str_path}/z_axis",
            Cylinder(0.001, 0.05),
            Rgba(0.0, 0.0, 1.0, transparency),
        )
        x_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([0, 1, 0])), np.array([0.025, 0, 0])
        )
        y_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([1, 0, 0])), np.array([0, 0.025, 0])
        )
        z_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([0, 0, 1])), np.array([0, 0, 0.025])
        )
        meshcat.SetTransform(f"{str_path}/x_axis", x_axis_transform)
        meshcat.SetTransform(f"{str_path}/y_axis", y_axis_transform)
        meshcat.SetTransform(f"{str_path}/z_axis", z_axis_transform)

    if frame_quat is not None and frame_pos is not None:
        meshcat.SetTransform(
            str_path,
            RigidTransform(quaternion=Quaternion(frame_quat), p=frame_pos),
        )


# ------------------------------------------------------------------------------------#
def angle_axis_to_quaternion(angle: float, axis: np.ndarray) -> np.ndarray:
    """
    Convert angle-axis representation to quaternion using Drake functions.

    Args:
        angle: Rotation angle in radians
        axis: 3D unit vector representing rotation axis

    Returns:
        Quaternion as [w, x, y, z]
    """
    # For z-axis rotation, use Drake's MakeZRotation directly
    if np.allclose(axis, [0, 0, 1]):
        rotation_matrix = RotationMatrix.MakeZRotation(angle)
    else:
        # For other axes, use the general approach
        rotation_matrix = RotationMatrix.MakeFromOneVector(axis, 2)
        rotation_matrix = rotation_matrix.multiply(RotationMatrix.MakeZRotation(angle))
    quaternion = Quaternion(rotation_matrix.matrix())
    return np.array([quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()])


def draw_frame_axes(
    meshcat: Meshcat,
    str_path: str,
    frame_quat: Optional[np.ndarray] = None,
    frame_pos: Optional[np.ndarray] = None,
    transparency: float = 1.0,
) -> None:
    if not meshcat.HasPath(f"{str_path}/x_axis"):
        meshcat.SetObject(
            f"{str_path}/x_axis",
            Cylinder(0.001, 0.05),
            Rgba(1.0, 0.0, 0.0, transparency),
        )
        meshcat.SetObject(
            f"{str_path}/y_axis",
            Cylinder(0.001, 0.05),
            Rgba(0.0, 1.0, 0.0, transparency),
        )
        meshcat.SetObject(
            f"{str_path}/z_axis",
            Cylinder(0.001, 0.05),
            Rgba(0.0, 0.0, 1.0, transparency),
        )
        x_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([0, 1, 0])), np.array([0.025, 0, 0])
        )
        y_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([1, 0, 0])), np.array([0, 0.025, 0])
        )
        z_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([0, 0, 1])), np.array([0, 0, 0.025])
        )
        meshcat.SetTransform(f"{str_path}/x_axis", x_axis_transform)
        meshcat.SetTransform(f"{str_path}/y_axis", y_axis_transform)
        meshcat.SetTransform(f"{str_path}/z_axis", z_axis_transform)

    if frame_quat is not None and frame_pos is not None:
        meshcat.SetTransform(
            str_path,
            RigidTransform(quaternion=Quaternion(frame_quat), p=frame_pos),
        )
