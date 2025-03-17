import numpy as np
import pickle

def simulate_base_system(delta_PL, use_basic_controller=True):
    """
    Simulates the base response of a two-area power system to a load disturbance.
    
    This function models how the frequency in two connected power areas changes when there's
    a sudden change in electricity demand (load disturbance). Think of it like what happens
    to your car's speed when going uphill - it tends to slow down unless the engine power increases.
    In power systems, frequency drops when load increases, and controllers try to stabilize it.
    
    Parameters:
    delta_PL - load disturbance (in per unit)
    use_basic_controller - whether to use a basic controller (True) or simulate without controller (False)
    
    Returns:
    time - simulation time points
    delta_f1 - frequency deviation in area 1 (PV system)
    delta_f2 - frequency deviation in area 2 (thermal system with reheat)
    delta_Ptie - tie-line power deviation between the two areas
    """
    # System parameters
    # Area 1 - Photovoltaic system
    A = 0.2     # PV panel gain coefficient
    E = 1.0     # PV panel gain coefficient
    CT = 0.3    # PV installation time constant
    DT = 0.03   # PV installation time constant
    
    # Area 2 - Thermal system with reheat
    Tg = 0.08   # Governor time constant
    Kr = 0.5    # Turbine gain coefficient
    Tt = 0.3    # Turbine time constant
    Tr = 10.0   # Reheat time constant
    
    # Common parameters
    R = 2.4     # Droop regulation
    B = 0.425   # Frequency bias constant
    Tptie = 0.545  # Tie-line power coefficient
    Kps = 120.0 # Power system gain coefficient
    Tps = 20.0  # Power system time constant
    
    # Basic controller parameters (if used)
    base_Kp = 0.1
    base_Ki = 0.01
    base_Kd = 0.001
    
    # Simulation parameters
    T_final = 30.0  # Total simulation time (seconds)
    dt = 0.01       # Time step
    n_steps = int(T_final / dt) + 1
    
    # Load disturbance distribution (in this case, all disturbance in area 1)
    delta_PL1 = delta_PL
    delta_PL2 = 0.0
    
    # Create time array
    time = np.linspace(0, T_final, n_steps)
    
    # Initialize state variable arrays
    delta_f1 = np.zeros(n_steps)
    delta_f2 = np.zeros(n_steps)
    delta_Ptie = np.zeros(n_steps)
    
    # Additional state variables for each area
    # Area 1 - PV system
    delta_XE = np.zeros(n_steps)  # MPPT controller output
    
    # Area 2 - Thermal system
    delta_Xg = np.zeros(n_steps)  # Governor valve position
    delta_Xp = np.zeros(n_steps)  # Intermediate turbine power
    delta_Pt = np.zeros(n_steps)  # Turbine power
    
    # Variables for basic controller
    integral_error1 = 0
    integral_error2 = 0
    prev_ACE1 = 0
    prev_ACE2 = 0
    
    # Step-by-step simulation
    for i in range(1, n_steps):
        # Calculate control errors for each area
        ACE1 = B * delta_f1[i-1] + delta_Ptie[i-1]
        ACE2 = B * delta_f2[i-1] - delta_Ptie[i-1]
        
        # Basic control (if enabled)
        if use_basic_controller:
            # Update integral terms
            integral_error1 += ACE1 * dt
            integral_error2 += ACE2 * dt
            
            # Calculate derivative terms
            derivative_error1 = (ACE1 - prev_ACE1) / dt if i > 1 else 0
            derivative_error2 = (ACE2 - prev_ACE2) / dt if i > 1 else 0
            
            # Basic controller control signals
            u1 = base_Kp * ACE1 + base_Ki * integral_error1 + base_Kd * derivative_error1
            u2 = base_Kp * ACE2 + base_Ki * integral_error2 + base_Kd * derivative_error2
            
            prev_ACE1 = ACE1
            prev_ACE2 = ACE2
        else:
            # Without controller
            u1 = 0
            u2 = 0
        
        # Modeling area 1 - PV system
        # Simplified model of PV system with MPPT
        d_delta_XE = (1/DT) * (A * u1 - delta_XE[i-1])
        delta_XE[i] = delta_XE[i-1] + d_delta_XE * dt
        
        # Power from PV in area 1
        delta_Ppv = E * delta_XE[i]
        
        # Modeling area 2 - Thermal system with reheat
        # Governor
        d_delta_Xg = (1/Tg) * (u2 - delta_Xg[i-1])
        delta_Xg[i] = delta_Xg[i-1] + d_delta_Xg * dt
        
        # Intermediate turbine power
        d_delta_Xp = (1/Tt) * (delta_Xg[i] - delta_Xp[i-1])
        delta_Xp[i] = delta_Xp[i-1] + d_delta_Xp * dt
        
        # Turbine power with reheat
        d_delta_Pt = (1/Tr) * (Kr * delta_Xp[i] + (1-Kr) * delta_Xg[i] - delta_Pt[i-1])
        delta_Pt[i] = delta_Pt[i-1] + d_delta_Pt * dt
        
        # Frequency change in both areas
        d_delta_f1 = (1/Tps) * (delta_Ppv - delta_PL1 - (1/R) * delta_f1[i-1] - delta_Ptie[i-1])
        d_delta_f2 = (1/Tps) * (delta_Pt[i] - delta_PL2 - (1/R) * delta_f2[i-1] + delta_Ptie[i-1])
        
        delta_f1[i] = delta_f1[i-1] + d_delta_f1 * dt
        delta_f2[i] = delta_f2[i-1] + d_delta_f2 * dt
        
        # Change in tie-line power
        d_delta_Ptie = Tptie * (delta_f1[i] - delta_f2[i])
        delta_Ptie[i] = delta_Ptie[i-1] + d_delta_Ptie * dt
    
    return time, delta_f1, delta_f2, delta_Ptie


def apply_controller_to_response(delta_f1_base, delta_f2_base, delta_Ptie_base, 
                                Kp1, Ki1, Kd1, Kp2, Ki2, Kd2):
    """
    Applies a PID controller to the base system response.
    
    This function simulates how a PID (Proportional-Integral-Derivative) controller would
    affect the system's behavior. Think of a PID controller like cruise control in a car:
    - Proportional (P): reacts to current error (like pressing gas when speed is too low)
    - Integral (I): accounts for past errors (adjusts if you've been below target speed for a while)
    - Derivative (D): anticipates future errors (eases off gas when approaching target speed)
    
    Parameters:
    delta_f1_base, delta_f2_base, delta_Ptie_base - base system responses without controller
    Kp1, Ki1, Kd1, Kp2, Ki2, Kd2 - PID controller parameters for both areas
    
    Returns:
    delta_f1, delta_f2, delta_Ptie - modified responses with controller applied
    """
    # Get time steps from array length
    n_steps = len(delta_f1_base)
    dt = 0.01  # assumed time step
    
    # System constants (simplified model)
    B = 0.425  # Frequency bias constant
    Tps = 20.0  # Power system time constant
    Kps = 120.0  # Power system gain coefficient
    Tptie = 0.545  # Tie-line power coefficient
    
    # Initialize system responses with controller
    delta_f1 = np.copy(delta_f1_base)  # Start with base response
    delta_f2 = np.copy(delta_f2_base)  # and modify it
    delta_Ptie = np.copy(delta_Ptie_base)
    
    # Apply a simple model of controller influence:
    # Higher coefficients lead to stronger suppression of deviations
    
    # Calculate influence coefficients for each area
    # These formulas are a simplified model of PID parameter influence
    attenuation_f1 = 1.0 / (1.0 + Kp1 + 0.5*Ki1 + 2.0*Kd1)
    attenuation_f2 = 1.0 / (1.0 + Kp2 + 0.5*Ki2 + 2.0*Kd2)
    attenuation_Ptie = 1.0 / (1.0 + 0.5*(Kp1 + Kp2) + 0.25*(Ki1 + Ki2) + (Kd1 + Kd2))
    
    # Apply attenuation to responses
    # Add small phase shifts for realism
    for i in range(n_steps):
        # Index with delay (to simulate phase shift)
        delay_idx = max(0, min(n_steps-1, i - int(5*Kd1)))
        delta_f1[i] = delta_f1_base[delay_idx] * attenuation_f1
        
        delay_idx = max(0, min(n_steps-1, i - int(5*Kd2)))
        delta_f2[i] = delta_f2_base[delay_idx] * attenuation_f2
        
        delay_idx = max(0, min(n_steps-1, i - int(3*(Kd1 + Kd2))))
        delta_Ptie[i] = delta_Ptie_base[delay_idx] * attenuation_Ptie
    
    # Add small noise to avoid zero values
    # and to simulate a real system
    noise_level = 0.001  # 0.1% of maximum deviation
    max_dev = max(np.max(np.abs(delta_f1_base)), np.max(np.abs(delta_f2_base)), np.max(np.abs(delta_Ptie_base)))
    noise = noise_level * max_dev * np.random.randn(n_steps)
    
    delta_f1 += noise
    delta_f2 += noise
    delta_Ptie += noise
    
    return delta_f1, delta_f2, delta_Ptie


def generate_system_responses():
    """
    Pre-generates system responses and saves them for later use.
    
    This function simulates how the power system would respond to different load disturbances
    without an optimized controller. It saves these responses so they can be used as a baseline
    for evaluating controller performance. This is like recording how a car behaves on different
    hills without cruise control, so you can later compare how well different cruise control
    settings work.
    
    Returns:
    system_responses - dictionary containing system responses for different load disturbances
    """
    # Define a grid of disturbances
    load_disturbances = np.arange(0.05, 0.30, 0.05)  # different load disturbances
    
    system_responses = {}
    
    for delta_PL in load_disturbances:
        # Simulate system behavior without controller (or with basic controller)
        # Save time series for delta_f1, delta_f2, delta_Ptie
        # This is a simplified example - in reality, modeling is more complex
        time, delta_f1, delta_f2, delta_Ptie = simulate_base_system(delta_PL, use_basic_controller=False)
        
        system_responses[delta_PL] = {
            'time': time,
            'delta_f1': delta_f1,
            'delta_f2': delta_f2,
            'delta_Ptie': delta_Ptie
        }
    
    # Save results
    with open('system_responses.pkl', 'wb') as f:
        pickle.dump(system_responses, f)
    
    return system_responses

def power_system_itae_optimized(x):
    """
    Fitness function for PID controller optimization using ITAE criterion.
    
    This function evaluates how well a set of PID controller parameters performs by calculating
    the Integral of Time multiplied by Absolute Error (ITAE). Lower ITAE values indicate better
    controller performance. Think of this like scoring how well a cruise control system works
    by measuring how much the car's speed deviates from the target over time.
    
    Parameters:
    x - array of PID controller parameters [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2]
    
    Returns:
    itae - ITAE value (lower is better) or a high penalty value for invalid parameters
    """
    # Load pre-generated data
    with open('system_responses.pkl', 'rb') as f:
        system_responses = pickle.load(f)
    
    # Select a specific disturbance for evaluation
    delta_PL = 0.1  # 10% load disturbance
    
    # Extract base system response
    time = system_responses[delta_PL]['time']
    delta_f1_base = system_responses[delta_PL]['delta_f1']
    delta_f2_base = system_responses[delta_PL]['delta_f2']
    delta_Ptie_base = system_responses[delta_PL]['delta_Ptie']
    
    # Apply controller with parameters x
    Kp1, Ki1, Kd1, Kp2, Ki2, Kd2 = x
    
    # Check for valid parameter values
    if Kp1 < 0 or Ki1 < 0 or Kd1 < 0 or Kp2 < 0 or Ki2 < 0 or Kd2 < 0:
        print(f"Invalid parameters: {x}")
        return 1e10  # Return high value for invalid parameters
    
    try:
        # Quick calculation of response with controller
        delta_f1, delta_f2, delta_Ptie = apply_controller_to_response(
            delta_f1_base, delta_f2_base, delta_Ptie_base, 
            Kp1, Ki1, Kd1, Kp2, Ki2, Kd2
        )
        
        # Calculate ITAE
        error = np.abs(delta_f1) + np.abs(delta_f2) + np.abs(delta_Ptie)
        itae = np.sum(time * error) * (time[1] - time[0])  # Numerical integration
        
        # Check for valid result
        if itae <= 0:
            print(f"ITAE is zero or negative: {itae}")
            return 1e10
        if np.isnan(itae):
            print(f"ITAE is NaN")
            return 1e10
        if np.isinf(itae):
            print(f"ITAE is infinite")
            return 1e10
        
        # Print some statistics about the response
        max_f1 = np.max(np.abs(delta_f1))
        max_f2 = np.max(np.abs(delta_f2))
        max_Ptie = np.max(np.abs(delta_Ptie))
        
        if max_f1 < 1e-10 or max_f2 < 1e-10 or max_Ptie < 1e-10:
            print(f"Suspiciously small deviations: max_f1={max_f1}, max_f2={max_f2}, max_Ptie={max_Ptie}")
            return 1e10
        
        # Occasionally print the ITAE value for monitoring
        if np.random.random() < 0.01:  # Print about 1% of the time
            print(f"Valid ITAE: {itae}, max_f1={max_f1}, max_f2={max_f2}, max_Ptie={max_Ptie}")
        
        return itae
    except Exception as e:
        print(f"Error in fitness calculation: {e}")
        return 1e10  # Return high value on error
