import mujoco
import mujoco.viewer
import numpy as np
import time

def pd_controller(target_point, data, prev_errors, dt, kp, kd, gravity_comp):
    """
    ПД-регулятор для управления дроном.
    
    Args:
        target_point: [x, y, z] целевая точка
        data: mujoco.MjData объект
        prev_errors: словарь предыдущих ошибок {'h':, 'x':, 'y':}
        dt: временной шаг
        kp: [kp_h, kp_x, kp_y] пропорциональные коэффициенты
        kd: [kd_h, kd_x, kd_y] дифференциальные коэффициенты
        gravity_comp: компенсация гравитации
    
    Returns:
        tuple: (base_thrust, x_control, y_control, new_errors)
    """
    
    h_error_p = target_point[2] - data.qpos[2]
    h_error_d = (h_error_p - prev_errors['h']) / dt
    high_control = kp[0] * h_error_p + kd[0] * h_error_d
    
    base_thrust = gravity_comp + high_control
    
    
    x_error_p = target_point[0] - data.qpos[0]
    x_error_d = (x_error_p - prev_errors['x']) / dt
    x_control = kp[1] * x_error_p + kd[1] * x_error_d
    

    y_error_p = target_point[1] - data.qpos[1]
    y_error_d = (y_error_p - prev_errors['y']) / dt
    y_control = kp[2] * y_error_p + kd[2] * y_error_d
    
    
    new_errors = {
        'h': h_error_p,
        'x': x_error_p,
        'y': y_error_p
    }
    
    return base_thrust, x_control, y_control, new_errors

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

# Начальные ошибки
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
        time.sleep(0.001)