import numpy as np
import robosuite as suite
# from robosuite.controllers import load_controller_config

class RobotPrimitives:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        self.robot_name = env.robots[0].robot_model.name
        
        # --- 1. Dynamic Gripper Site Lookup ---
        # We look for a site that contains 'grip_site' in the model
        all_site_names = [self.sim.model.site_id2name(i) for i in range(self.sim.model.nsite)]
        self.eef_site_name = None
        
        # Priority: Try to find the one explicitly mentioned in your error first
        possible_names = ["gripper0_right_grip_site", "gripper0_grip_site", "grip_site", "ee_site"]
        
        for name in possible_names:
            if name in all_site_names:
                self.eef_site_name = name
                break
        
        # Fallback: search for any site with 'grip_site' in the name
        if self.eef_site_name is None:
            for name in all_site_names:
                if "grip_site" in name:
                    self.eef_site_name = name
                    break
                    
        if self.eef_site_name is None:
            raise ValueError(f"Could not find a valid gripper site. Available: {all_site_names}")
        
        print(f"DEBUG: Using End-Effector Site: {self.eef_site_name}")

        # --- 2. Dynamic Object Lookup ---
        # Robosuite 'Stack' env stores objects in env.model.mujoco_objects
        # We extract their names to find their physics bodies later
        self.object_names = [obj.name for obj in env.model.mujoco_objects if "cube" in obj.name]
        
        # Store body IDs for "God Mode" (Absolute Coordinate Access)
        self.obj_body_ids = {}
        for name in self.object_names:
            # In Robosuite, the main physical body is often named "{name}_main"
            # We try "{name}_main" first, then just "{name}"
            body_name = f"{name}_main"
            try:
                self.obj_body_ids[name] = self.sim.model.body_name2id(body_name)
            except Exception:
                # Fallback if _main suffix doesn't exist
                try:
                    self.obj_body_ids[name] = self.sim.model.body_name2id(name)
                except:
                    print(f"Warning: Could not find physics body for {name}")

        print(f"DEBUG: Found Objects: {list(self.obj_body_ids.keys())}")

        # Gripper state definitions (Standard for Panda: 1=closed, -1=open)
        self.GRIPPER_OPEN = -1
        self.GRIPPER_CLOSED = 1
        self.current_gripper_action = self.GRIPPER_OPEN

    def get_eef_pos(self):
        """Returns the current (x,y,z) of the end-effector."""
        site_id = self.sim.model.site_name2id(self.eef_site_name)
        return np.array(self.sim.data.site_xpos[site_id])

    def get_object_pos(self, obj_name):
        """Oracle function to get an object's position directly from MuJoCo."""
        if obj_name not in self.obj_body_ids:
            # Fallback: try to find a key that *contains* the name (e.g. 'CubeA' inside 'CubeA')
            found = False
            for key in self.obj_body_ids:
                if obj_name in key:
                    obj_name = key
                    found = True
                    break
            if not found:
                raise ValueError(f"Object '{obj_name}' not found. Available: {list(self.obj_body_ids.keys())}")
        
        body_id = self.obj_body_ids[obj_name]
        return np.array(self.sim.data.body_xpos[body_id])

    def move_to(self, target_pos, tolerance=0.02, timeout=150):
        """
        Moves the robot to a target (x,y,z) position using the OSC controller.
        Blocking call that loops until the robot gets there.
        """
        for i in range(timeout):
            current_pos = self.get_eef_pos()
            delta = target_pos - current_pos
            dist = np.linalg.norm(delta)
            
            # 1. Check Success
            if dist < tolerance:
                break
                
            # 2. Calculate Action (P-Controller)
            # Action Space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action = np.zeros(7)
            
            # Simple Gain (KP)
            kp = 3.0
            # Limit max speed to prevent instability
            action[:3] = np.clip(delta * kp, -1.0, 1.0) 
            
            # Keep gripper state constant
            action[6] = self.current_gripper_action
            
            self.env.step(action)
            self.env.render() # Crucial for visualization

        # Stabilize
        for _ in range(5):
            self.env.step(np.zeros(7))
            self.env.render()

    def open_gripper(self):
        print(" > Opening Gripper")
        self.current_gripper_action = self.GRIPPER_OPEN
        # Wait for physics to actuate
        for _ in range(15):
            action = np.zeros(7)
            action[6] = self.current_gripper_action
            self.env.step(action)
            self.env.render()

    def close_gripper(self):
        print(" > Closing Gripper")
        self.current_gripper_action = self.GRIPPER_CLOSED
        for _ in range(15):
            action = np.zeros(7)
            action[6] = self.current_gripper_action
            self.env.step(action)
            self.env.render()

    # --- HIGH LEVEL API (The "Code as Policies" part) ---

    def pick(self, obj_name):
        print(f"Executing: pick({obj_name})")
        obj_pos = self.get_object_pos(obj_name)
        
        # 1. Hover
        hover_pos = obj_pos.copy()
        hover_pos[2] += 0.20  # Hover 20cm above
        self.open_gripper()
        self.move_to(hover_pos)
        
        # 2. Descend
        grasp_pos = obj_pos.copy()
        # grasp_pos[2] += 0.005 # Target slightly above center of object to avoid collision
        self.move_to(grasp_pos - 0.01)
        
        # 3. Grasp
        self.close_gripper()
        
        # 4. Lift
        self.move_to(hover_pos)

    def place(self, target_pos):
        print(f"Executing: place({target_pos})")
        
        # 1. Hover above target
        hover_pos = target_pos.copy()
        hover_pos[2] += 0.20
        self.move_to(hover_pos)
        
        # 2. Descend
        drop_pos = target_pos.copy()
        drop_pos[2] += 0.05 # Drop 5cm above target
        self.move_to(drop_pos)
        
        # 3. Release
        self.open_gripper()
        
        # 4. Retreat
        self.move_to(hover_pos)

    def place_on_object(self, bottom_obj_name):
        print(f"Executing: place_on_object({bottom_obj_name})")
        target_pos = self.get_object_pos(bottom_obj_name)
        # Stack height offset (cubes are ~5cm, so target is +5cm)
        target_pos[2] += 0.05 
        self.place(target_pos)


# ==========================================
#  MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Initializing Environment...")
    
    # Create the environment with OSC_POSE controller
    # This controller allows us to send [x,y,z] commands directly
    # config = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name="Stack",
        robots="Panda",
        # controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )
    env.reset()

    # Initialize Primitives
    try:
        robot = RobotPrimitives(env)
        
        # --- THE LLM SCRIPT ---
        # This simulates what the Code-as-Policies model would output
        print("\n>>> STARTING LLM GENERATED POLICY")
        
        # 1. Identify objects (Hardcoded for demo, but names are dynamic in class)
        # Usually CubeA is Red, CubeB is Green
        
        # 2. Pick CubeA
        robot.pick("cubeA")
        
        # 3. Stack on CubeB
        robot.place_on_object("cubeB")
        
        print(">>> DONE")
        
        # Keep window open
        while True:
            env.render()
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        env.close()