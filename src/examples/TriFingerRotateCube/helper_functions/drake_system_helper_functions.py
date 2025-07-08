from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from pydrake.autodiffutils import AutoDiffXd
from pydrake.geometry import Meshcat, MeshcatVisualizer, SceneGraph
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlant_
from pydrake.systems.framework import Context, Context_, DiagramBuilder, DiagramBuilder_

from .drake_helper_functions import create_plant_from_urdfs, initialize_meshcat


def load_env_configs(env_config_path: str) -> Dict[str, Any]:
    """Load environment configuration from YAML file.

    Args:
        env_config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing environment configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_path = Path(env_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {env_config_path}")

    try:
        with open(config_path, "r") as f:
            env_configs = yaml.safe_load(f)
        return env_configs
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {env_config_path}: {e}")


def validate_env_configs(env_configs: Dict[str, Any]) -> None:
    """Validate that all required configuration keys are present.

    Args:
        env_configs: Environment configuration dictionary

    Raises:
        KeyError: If required configuration keys are missing
        ValueError: If configuration values are invalid
    """
    required_keys = [
        "package_map",
        "urdfs",
        "dt",
        "welded_links",
        "additional_joints",
        "show_collision_geometries",
    ]

    missing_keys = [key for key in required_keys if key not in env_configs]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

    # Validate specific configuration values
    if env_configs["dt"] <= 0:
        raise ValueError("dt must be positive")

    if not isinstance(env_configs["urdfs"], list) or not env_configs["urdfs"]:
        raise ValueError("urdfs must be a non-empty list")


def setup_drake_system(
    env_config_path: str,
) -> Tuple[
    MultibodyPlant,
    MultibodyPlant_[AutoDiffXd],
    Context,
    Context_[AutoDiffXd],
    DiagramBuilder,
    Context,
    DiagramBuilder_[AutoDiffXd],
    Context_[AutoDiffXd],
    Meshcat,
    SceneGraph,
    Optional[MeshcatVisualizer],
    Optional[MeshcatVisualizer],
    List[Dict[str, str]],
    List[float],
]:
    """Create a Drake environment from configuration file.

    This function creates a complete Drake simulation environment including:
    - Multibody plant with URDF models
    - AutoDiffXd version for optimization
    - Scene graph for visualization
    - Visualizers for debugging

    Args:
        env_config_path: Path to the YAML configuration file containing:
            - package_map: Mapping of package names to paths
            - urdfs: List of URDF file paths
            - sim_dt: Simulation time step
            - welded_links: Links to weld together
            - additional_joints: Specifications for additional joints
            - show_collision_geometries: Whether to show collision geometry

    Returns:
        Tuple containing:
            - plant: MultibodyPlant for simulation
            - plant_ad: AutoDiffXd version of plant for optimization
            - plant_context: Context for the plant
            - plant_context_ad: AutoDiffXd context for the plant
            - plant_diagram: Built diagram
            - plant_diagram_context: Context for the diagram
            - plant_diagram_ad: AutoDiffXd version of the diagram
            - plant_diagram_ad_context: AutoDiffXd context for the diagram
            - scene_graph: Scene graph for visualization
            - visual_visualizer: Visual geometry visualizer
            - collision_visualizer: Collision geometry visualizer

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML file is malformed
        KeyError: If required configuration keys are missing
        ValueError: If configuration values are invalid
        RuntimeError: If plant creation fails
    """
    # Load and validate configuration
    env_configs = load_env_configs(env_config_path)
    validate_env_configs(env_configs)

    # Create diagram builder and plant
    builder = DiagramBuilder()
    meshcat = initialize_meshcat()

    try:
        plant, scene_graph, visual_visualizer, collision_visualizer = (
            create_plant_from_urdfs(
                builder,
                env_configs["package_map"],
                env_configs["urdfs"],
                env_configs["dt"],
                env_configs["welded_links"],
                env_configs["additional_joints"],
                meshcat,
                env_configs["show_collision_geometries"],
                env_configs["dt"],
            )
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create plant from URDFs: {e}")

    # Build the diagram and create contexts
    plant_diagram = builder.Build()
    plant_diagram_context = plant_diagram.CreateDefaultContext()

    # Convert the diagram to AutoDiffXd for optimization
    plant_diagram_ad = plant_diagram.ToAutoDiffXd()
    plant_diagram_ad_context = plant_diagram_ad.CreateDefaultContext()

    # Get mutable subsystem contexts
    plant_context = plant_diagram.GetMutableSubsystemContext(
        plant, plant_diagram_context
    )
    plant_ad = plant_diagram_ad.GetSubsystemByName("plant")
    plant_ad_context = plant_diagram_ad.GetMutableSubsystemContext(
        plant_ad, plant_diagram_ad_context
    )

    # Get contact pairs
    contact_pairs = env_configs["contact_geoms"]
    contact_friction_coeffs = env_configs["contact_friction_coeffs"]

    return (
        plant,
        plant_ad,
        plant_context,
        plant_ad_context,
        plant_diagram,
        plant_diagram_context,
        plant_diagram_ad,
        plant_diagram_ad_context,
        meshcat,
        scene_graph,
        visual_visualizer,
        collision_visualizer,
        contact_pairs,
        contact_friction_coeffs,
    )


class DrakeSystem:
    """Container class for all Drake system components.

    This class holds all the components returned by setup_drake_system function,
    providing a clean interface for accessing the Drake simulation environment.

    The class encapsulates:
    - MultibodyPlant instances (regular and AutoDiffXd versions)
    - Context objects for state management
    - Diagram builders and contexts
    - Scene graph for visualization
    - Visualizers for debugging
    - Contact pairs and friction coefficients for contact detection
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        plant_ad: MultibodyPlant_[AutoDiffXd],
        plant_context: Context,
        plant_ad_context: Context_[AutoDiffXd],
        plant_diagram: DiagramBuilder,
        plant_diagram_context: Context,
        plant_diagram_ad: DiagramBuilder_[AutoDiffXd],
        plant_diagram_ad_context: Context_[AutoDiffXd],
        meshcat: Meshcat,
        scene_graph: SceneGraph,
        visual_visualizer: Optional[MeshcatVisualizer],
        collision_visualizer: Optional[MeshcatVisualizer],
        contact_pairs: List[Dict[str, str]],
        contact_friction_coeffs: List[float],
    ):
        """Initialize DrakeSystem with all Drake components.

        Args:
            plant: MultibodyPlant for simulation
            plant_ad: AutoDiffXd version of plant for optimization
            plant_context: Context for the plant
            plant_ad_context: AutoDiffXd context for the plant
            plant_diagram: Built diagram
            plant_diagram_context: Context for the diagram
            plant_diagram_ad: AutoDiffXd version of the diagram
            plant_diagram_ad_context: AutoDiffXd context for the diagram
            meshcat: Meshcat instance for visualization
            scene_graph: Scene graph for visualization
            visual_visualizer: Visual geometry visualizer
            collision_visualizer: Collision geometry visualizer
            contact_pairs: List of contact geometry pairs for collision detection
            contact_friction_coeffs: List of friction coefficients for contact pairs
        """
        self.plant = plant
        self.plant_ad = plant_ad
        self.plant_context = plant_context
        self.plant_ad_context = plant_ad_context
        self.plant_diagram = plant_diagram
        self.plant_diagram_context = plant_diagram_context
        self.plant_diagram_ad = plant_diagram_ad
        self.plant_diagram_ad_context = plant_diagram_ad_context
        self.meshcat = meshcat
        self.scene_graph = scene_graph
        self.visual_visualizer = visual_visualizer
        self.collision_visualizer = collision_visualizer
        self.contact_pairs = contact_pairs
        self.contact_friction_coeffs = contact_friction_coeffs

    @classmethod
    def from_config(cls, env_config_path: str) -> "DrakeSystem":
        """Create a DrakeSystem from a configuration file.

        Args:
            env_config_path: Path to the YAML configuration file

        Returns:
            DrakeSystem instance with all components initialized

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML file is malformed
            KeyError: If required configuration keys are missing
            ValueError: If configuration values are invalid
            RuntimeError: If plant creation fails
        """
        components = setup_drake_system(env_config_path)
        return cls(*components)
