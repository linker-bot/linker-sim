## Target Spec (Isaac Sim RL Pick-and-Place)

This doc defines the *deterministic interface* between simulation, perception, and the RL policy. Fill in every field; avoid vague wording because downstream reward/termination logic depends on it.

### 1) Robot Hardware

**Robot model / URDF name:** `Rokae AR5 + Linkerhand L6`

**Arm**
- **Arm DoF:** `7`
- **Joint list (order matters):** `[base_rotation, shoulder, elbow, forearm1, forearm2, wrist1, wrist2]`
- **Joint limits (rad / deg; units must be consistent):**
  - `base_rotation: [{{min}}, {{max}}] {{units}}`
  - `{{joint_name}}: [{{min}}, {{max}}] {{units}}`
- **Actuation / control mode:** `position | velocity | torque` (choose one) = `{{fill}}`
- **Control input representation to policy/controller:**
  - Commanded values are interpreted as: `q_des | qdot_des | tau_des`
  - Saturation/clipping behavior: `{{fill}}`

**Hand / Gripper**
- **Hand DoF:** `{{fill}}`
- **Joint list (order matters):** `{{fill}}`
- **Joint limits:** `{{fill}}`
- **Actuation / control mode:** `position | velocity | torque` = `{{fill}}`
- **Grasping actuation notes (e.g., parallel jaw closure rule):** `{{fill}}`

**Coordinate frames**
- **World frame definition:** `{{fill}}`
- **Robot base frame definition:** `{{fill}}`
- **End-effector / hand frame definition:** `{{fill}}`
- **Object pose frame definition (what point on the object?):** `{{fill}}`

### 2) Control Loop Timing

- **Control frequency target:** `{{fill}}` Hz (e.g., 20 / 50 / 100)
- **Simulation step / policy step relation:**
  - Isaac Sim physics timestep: `{{fill}}` s
  - Policy action period: `{{fill}}` steps of physics
  - Action hold / interpolation: `{{fill}}`
- **Latency between perception and control:** `{{fill}}` s (or ms)
- **Receding-horizon behavior (if any):** `{{fill}}` (e.g., single-step MPC, fixed horizon)

### 3) Perception Inputs (Real-world Signals Emulated in Sim)

In real deployment, perception is imperfect and delayed. Specify exactly what the policy/controller receives.

- **Available perception inputs:** choose any and fill details
  - `object_pose`: 6DoF pose estimate (`T_world_object`), with units and rotation format (`quat` or `rotmat`), noise model: `{{fill}}`
  - `keypoints`: `N` object keypoints in either:
    - `{{fill}}` coordinate frame: `world | camera | robot_base`
    - format: `{{fill}}` (3D positions, 2D pixels + depth, heatmaps, etc.)
    - noise model: `{{fill}}`
  - `rgb` / `depth` / `segmentation` (if used): `{{fill}}`
  - `robot state`: joint positions/velocities, end-effector pose, gripper width: `{{fill}}`
- **Perception latency:** `{{fill}}` ms (distribution if stochastic)
- **Droop / missing data rate (if any):** `{{fill}}`
- **Update rate of perception:** `{{fill}}` Hz
- **Time alignment rule (how delayed pose/keypoints are synchronized to control step):** `{{fill}}`

### 4) Initial Task Set (Start Simple)

**Task ID 1:** `{{fill}}` (recommended: `pick_and_place_single`)

**Task description:**
- Initial condition:
  - Object spawn distribution: `{{fill}}` (positions/orientations)
  - Table height / workspace bounds: `{{fill}}`
  - Starting gripper state: `open | closed | pregrasp pose`: `{{fill}}`
- Goal specification:
  - Target pose representation: `{{fill}}` (object target pose? or keypoint target?)
  - Target distribution: `{{fill}}`
- Episode horizon (max steps): `{{fill}}`

**Success condition (high-level):**
- `{{fill}}` (e.g., grasp, lift, move to target region, release)

### 5) Success Metrics (Measurable + Deterministic)

Pick-and-place success must be evaluated with explicit thresholds.

**Pose threshold metrics** (if using object pose / end-effector pose)
- **Translation threshold:** `{{fill}}` meters (e.g., <= 2e-2 m)
- **Rotation threshold:** `{{fill}}` degrees or quaternion distance metric (specify which)
- **Reference frame for error:** `world | object_center | target_frame`: `{{fill}}`

**Keypoint distance metrics** (if using keypoints)
- **Number of keypoints (N):** `{{fill}}`
- **Distance metric:** `L2_3d | L1 | normalized_pixel | geodesic_on_mesh` = `{{fill}}`
- **Acceptable threshold:** `{{fill}}` (units must match coordinate frame)
- **Occlusion handling rule (if any):** `{{fill}}`

**Success steps / staged verification** (recommended for stability)
Define boolean checks that are robust to small noise. Example format:
- `success_steps[0] (grasp):` `{{condition}}` (e.g., gripper closed AND object attached/contact for >= K steps)
- `success_steps[1] (lift):` `{{condition}}` (e.g., object z > {{fill}} for >= K steps)
- `success_steps[2] (place):` `{{condition}}` (e.g., object within target region thresholds)
- `success_steps[3] (release):` `{{condition}}` (e.g., gripper open AND object not attached for >= K steps)

**Overall success aggregation:**
- `success = all(required_success_steps)` where `required_success_steps = {{fill}}`
- Alternative: success if `>= {{k}}` of `{{m}}` staged checks pass

### 6) Failure / Termination Conditions

Specify what ends an episode early (or marks failure) to avoid RL reward hacking.
- **Early failure conditions:**
  - Object dropped / falls below: `{{fill}}`
  - Timeout: `{{fill}}` steps
  - Collisions outside allowed contacts: `{{fill}}`
- **Safety constraints (sim-only):** `{{fill}}`

### 7) Notes for Reproducibility

- Random seed policy: `{{fill}}`
- Domain randomization knobs (initially minimal): `{{fill}}`
- Isaac Sim settings that must match real-world intent: `{{fill}}`

