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