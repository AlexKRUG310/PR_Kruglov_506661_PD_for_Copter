import mujoco
import mujoco.viewer
import numpy as np
import time
from PD import pd_controller


model = mujoco.MjModel.from_xml_path("x2.xml")
data = mujoco.MjData(model)

kp = [35.0, 0.0000003, 0.0000003]    
kd = [5.0, 0.0000003, 0.0000003]    
gravity_comp = 3.25

points =[
    np.array([0, 0, 0.8]),
    np.array([-5, 0, 2]),
    np.array([-10, 0, 3]),
    np.array([-15, 0, 1.2]), 
    np.array([-20, 0, 5]),
    np.array([-25, 0, 2])]

current_point_idx = 0

prev_errors = {'h': 0, 'x': 0, 'y': 0}
prev_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    
   viewer.cam.distance = 20.0     
   viewer.cam.azimuth = 140        
   viewer.cam.elevation = -20     
   viewer.cam.lookat[:] = [-7, 0, 1.2]
    
   while viewer.is_running():
        current_time = time.time()
        dt = 0.0015
        
        target = points[current_point_idx]
        
        base_thrust, x_control, y_control, prev_errors = pd_controller(
            target, data, prev_errors, dt, kp, kd, gravity_comp
        )
        
        distance = np.linalg.norm(target - data.qpos[0:3])
        
        data.ctrl[0] = base_thrust + x_control + y_control
        data.ctrl[1] = base_thrust + x_control - y_control
        data.ctrl[2] = base_thrust - x_control - y_control
        data.ctrl[3] = base_thrust - x_control + y_control
        
        if  distance < 0.5:  
           current_point_idx = (current_point_idx + 1) % len(points)
           print(f" Переход к точке {current_point_idx}")
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)