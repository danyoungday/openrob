import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects.primitive import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_controller_config

# =============================================================================
# 1. THE CUSTOM ENVIRONMENT (4 BLOCKS)
# =============================================================================
class FourBlockEnv(Stack):
    """
    A custom environment extending 'Stack' to have 4 colored blocks.
    """
    def _load_model(self):
        """
        Overriding the model loading to spawn 4 blocks instead of 2.
        """
        # A. Setup Robot and Arena (Standard Robosuite logic)
        self.mujoco_robot.robot_model.set_base_xpos([0, 0, 0])
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),
        )
        self.mujoco_arena.set_origin([0.8, 0, 0])

        # B. Define The 4 Blocks
        box_size = [0.02, 0.02, 0.02] # 2cm half-size (4cm total)
        
        # Standard Red/Green from Stack
        self.cubeA = BoxObject(name="CubeA", size=box_size, rgba=[1, 0, 0, 1]) # Red
        self.cubeB = BoxObject(name="CubeB", size=box_size, rgba=[0, 1, 0, 1]) # Green
        
        # NEW: Blue and Yellow
        self.cubeC = BoxObject(name="CubeC", size=box_size, rgba=[0, 0, 1, 1]) # Blue
        self.cubeD = BoxObject(name="CubeD", size=box_size, rgba=[1, 1, 0, 1]) # Yellow

        self.mujoco_objects = [self.cubeA, self.cubeB, self.cubeC, self.cubeD]

        # C. Create the Task
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[self.mujoco_robot.robot_model],
            mujoco_objects=self.mujoco_objects,
        )

    def _reset_internal(self):
        """
        Overriding reset to randomize positions of all 4 blocks.
        """
        # Initialize standard robot/arena reset
        super(Stack, self)._reset_internal() # Skip Stack's reset, go to parent

        # Create a sampler for all 4 objects
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.mujoco_objects,
            x_range=[-0.20, 0.20], # Bounds on the table
            y_range=[-0.20, 0.20],
            rotation=None,
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.mujoco_arena.table_top_abs,
            z_offset=0.01,
        )

        # Sample and Apply Positions
        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_pos, obj_quat]))

    def reward(self, action=None):
        return 0 # Open-ended task, no reward needed

# =============================================================================
# 2. THE ROBOT PRIMITIVES (API FOR LLM)
# =============================================================================
class RobotPrimitives:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.object_names = ["CubeA", "CubeB", "CubeC", "CubeD"]
        
        # Cache body IDs for position lookup
        self.body_ids = {}
        for name in self.object_names:
            # Robosuite naming: {name}_main is the visual/physics body
            self.body_ids[name] = self.sim.model.body_name2id(f"{name}_main")
        
        # Gripper ID (Panda right gripper)
        # Note: In some versions this might be "gripper0_grip_site"
        self.eef_site_id = self.sim.model.site_name2id("gripper0_right_grip_site") 

    def get_pos(self, obj_name):
        return np.array(self.sim.data.body_xpos[self.body_ids[obj_name]])

    def get_ee_pos(self):
        return np.array(self.sim.data.site_xpos[self.eef_site_id])

    def move_to(self, target_pos, grasp=False):
        """
        Basic P-Controller to move robot to XYZ.
        grasp=True closes gripper, False opens it.
        """
        gripper_val = 1 if grasp else -1
        
        for _ in range(80): # Timeout/Steps
            current_pos = self.get_ee_pos()
            delta = target_pos - current_pos
            
            if np.linalg.norm(delta) < 0.01:
                break
            
            # Action: [x, y, z, roll, pitch, yaw, gripper]
            action = np.zeros(7)
            action[:3] = delta * 4.0 # P-gain
            action[6] = gripper_val
            
            self.env.step(action)
            self.env.render()

    def pick(self, obj_name):
        print(f"PRIMITIVE: Pick({obj_name})")
        pos = self.get_pos(obj_name)
        
        # Hover, Down, Grasp, Up
        self.move_to(pos + [0, 0, 0.20], grasp=False)
        self.move_to(pos + [0, 0, 0.005], grasp=False)
        self.move_to(pos + [0, 0, 0.005], grasp=True)
        for _ in range(20): self.env.step([0,0,0,0,0,0,1]) # Tighten grip
        self.move_to(pos + [0, 0, 0.20], grasp=True)

    def place(self, position):
        print(f"PRIMITIVE: Place({position})")
        # Hover, Down, Release, Up
        self.move_to(position + [0, 0, 0.20], grasp=True)
        self.move_to(position + [0, 0, 0.05], grasp=True)
        self.move_to(position + [0, 0, 0.05], grasp=False)
        self.move_to(position + [0, 0, 0.20], grasp=False)


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Initializing 4-Block Robosuite Environment...")
    
    # Use OSC_POSE for easy XYZ control
    config = load_controller_config(default_controller="OSC_POSE")

    # Instantiate our Custom Class directly
    env = FourBlockEnv(
        robots="Panda",
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()

    # Initialize Primitives
    robot = RobotPrimitives(env)
    
    # ---------------------------------------------------------
    # THIS IS WHERE YOUR LLM CODE WOULD GO
    # ---------------------------------------------------------
    print("\n>>> Task: Stack Red (A) on Blue (C)")
    
    try:
        # 1. Get positions
        pos_blue = robot.get_pos("CubeC")
        
        # 2. Pick Red
        robot.pick("CubeA")
        
        # 3. Place on Blue (Stacking offset ~5cm)
        robot.place(pos_blue + [0, 0, 0.05])
        
        print(">>> Task Complete. Use Ctrl+C to exit.")
        while True:
            env.render()
            
    except KeyboardInterrupt:
        env.close()