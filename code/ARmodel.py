"""
This code models the dynamic behavior of a system consisting of six masses
connected by springs and subjected to viscous damping and an external force.
The system is represented by an ARMA (Auto-Regressive Moving Average) model,
which is used to predict the future positions and velocities of the masses
based on their current states and the applied force.

The script performs the following functions:
1. Sets up the main directories for saving figures and data.
2. Defines the AR_model function to compute the next state of the system.
3. Constructs the matrices A and B, which represent the system matrix and the
input matrix, respectively.
4. Implements the function to model the external force as a step or sinusoidal
input.
5. Evolves the system over time using the evolution function, which applies the
AR model iteratively to simulate the system's response to the external force.
6. Calculates the relative displacements between the masses, which could be
analogous to sensor readings in a physical system.
7. Plots the results, showing the temporal evolution of the system's response.
8. Optionally, saves the plot to the specified results directory.

The script is structured to allow easy modification of the system's parameters,
such as mass, spring constants, damping coefficients, and the characteristics
of the external force. This flexibility makes it suitable for simulating a wide
range of physical systems that can be modeled with an ARMA approach.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#----------------------------Setup the main directories------------------------#
script_dir = os.getcwd()                            #define current dir
main_dir = os.path.dirname(script_dir)              #go up of one directory
results_dir = os.path.join(main_dir, "figure")      #define figure dir
data_dir = os.path.join(main_dir, "data")           #define data dir

if not os.path.exists(results_dir):                 #if the directory does not
    os.mkdir(results_dir)                           # exist create it


if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#----------------------------------AR model------------------------------------#
def AR_model(y, A, B, u):
    """
    Computes the next state of the system using the ARMA model.

    Parameters
    ----------
    y : ndarray
        The current state vector, typically contains the positions and
        velocities of the bodies comprising the mechanical system.
    A : ndarray
        The system matrix that relates the current state to the next state.
    B : ndarray
        The input matrix that specifies how the system’s inputs affect the
        evolution of its state variables. Each column corresponds to a different
        input to the system, thus each element indicates the influence of each
        input on the rate of change of each state variable
    u : ndarray
        The input vector that represents external inputs applied to the system

    Returns
    -------
    ndarray
        The next state vector of the system.
    """
    return A @ y + B * u  #Return the next state of the system

#--------------------------------System's matrices-----------------------------#
def matrix(M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3, g4, g5, g6,
           dt):
    """
    Defines the matrices A and B based on the system's physical parameters.

    Parameters
    ----------
    M1, M2, M3, M4, M5, M6 : float
            Masses of the system components.
    K1, K2, K3, K4, K5, K6 : float
            Spring constants of the system.
    g2, g3, g4, g5, g6 : float
            Damping coefficients for the viscous friction force.
    dt : float
         Time step size.

    Returns
    -------
    tuple
        A tuple containing the system matrix A and the input matrix B.
    """
    # defne the matrices A and B
    Id = np.eye(6)
    V = np.array([[1-(dt*g2/M1), dt*g2/M1, 0, 0, 0, 0],
                       [dt*g2/M2, 1-dt*(g2+g3)/M2, dt*g3/M2, 0, 0, 0],
                       [0, dt*g3/M3, 1-dt*(g3+g4)/M3, dt*g4/M3, 0, 0],
                       [0, 0, dt*g4/M4, 1-dt*(g4+g5)/M4, dt*g5/M4, 0],
                       [0, 0, 0, dt*g5/M5, 1-dt*(g5+g6)/M5, dt*g6/M5],
                       [0, 0, 0, 0, dt*g6/M6, 1-dt*g6/M6]])
    X = dt * np.array([[-(K1+K2)/M1, K2/M1, 0, 0, 0, 0],
                       [K2/M2, -(K2+K3)/M2, K3/M2, 0, 0, 0],
                       [0, K3/M3, -(K3+K4)/M3, K4/M3, 0, 0],
                       [0, 0, K4/M4, -(K4+K5)/M4, K5/M4, 0],
                       [0, 0, 0, K5/M5, -(K5+K6)/M5, K6/M5],
                       [0, 0, 0, 0, K6/M6, -K6/M6]])
    A = np.block([[V, X],
                  [dt*Id, Id]])

    B = np.array((K1*dt/M1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    return A, B

#----------------------------------External force------------------------------#
def sin_function(t, F0, w):
    """
    Models a sinusoidal function representing the external force.

    Parameters
    ----------
    t : float
        The current time.
    F0 : float
        The amplitude of the sinusoidal force.
    w : float
        The angular frequency of the sinusoidal force.

    Returns
    -------
    float
        The value of the sinusoidal force at time t.
    """
    return F0 * np.sin(w*t)

#--------------------------------Temporal evolution----------------------------#
def evolution(evol_method, Nt_step, dt, physical_params, signal_params, F,
              file_name = None):
    """
    Simulates the temporal evolution of the system under the influence of an
    external force.

    Parameters
    ----------
    evol_method : function
        The function used to evolve the system (e.g. Euler or ARMA methods).
    Nt_step : int
        The number of temporal steps to simulate.
    dt : float
        The time step size.
    physical_params : list
        The list of physical parameters for the system.
    signal_params : list
        The list of parameters for the external force signal.
    F : function
        The function modeling the external force.
    file_name : str, optional
        The name of the file to save simulation data. Default is None.

    Returns
    -------
    tuple
        A tuple containing the time grid and the arrays of velocities
         and positions for each mass.
    """
    #Initialize the problem
    tmax = dt * Nt_step                             # total time of simulation
    tt = np.arange(0, tmax, dt)                     # temporal grid
    y0 = np.array((0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0.)) #initial condition
    y_t = np.copy(y0)                       # create a copy to evolve it in time
    F_signal = F(tt, *signal_params)        # external force applied over time

    # Initialize lists for velocities and positions
    v1, v2, v3, v4, v5, v6 = [[], [], [], [], [], []]
    x1, x2, x3, x4, x5, x6 = [[], [], [], [], [], []]

    # compute the system matrices
    A, B = matrix(*physical_params)

    # time evolution when the ext force is applied
    i = 0
    for t in tt:
        Fi = F_signal[i]  # evaluete the force at time t
        i = i + 1
        y_t = evol_method(y_t, A, B, Fi)   # evolve to step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        v3.append(y_t[2])
        v4.append(y_t[3])
        v5.append(y_t[4])
        v6.append(y_t[5])
        x1.append(y_t[6])
        x2.append(y_t[7])
        x3.append(y_t[8])
        x4.append(y_t[9])
        x5.append(y_t[10])
        x6.append(y_t[11])

    # save simulation's data (if a file name is provided)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v3, v4, v5, v6,
                                x1, x2, x3, x4, x5, x6))
        np.savetxt(os.path.join(data_dir, file_name), data,
                header='time, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6')

    return (tt, np.array(v1), np.array(v2), np.array(v3), np.array(v4),
                np.array(v5), np.array(v6), np.array(x1), np.array(x2),
                np.array(x3), np.array(x4), np.array(x5), np.array(x6))


if __name__ == '__main__':

    # Parameters of the simulation
    Nt_step = 2e5     #temporal steps
    dt = 1e-3         #temporal step size

    # Parameters of the system
    gamma = [5, 5, 5, 5, 5]                 # viscous friction coeff [kg/m*s]
    M = [160, 125, 120, 110, 325, 82]       # filter mass [Kg]
    K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]
    F0 = 1                                  # amplitude of the external force
    w = 2*np.pi*0.16                        # angular frequency of the ext force

    # External force applied to the system
    F = sin_function

    # Simulation
    physical_params = [*M, *K, *gamma, dt]
    signal_params = [F0, w]
    simulation_params = [AR_model, Nt_step, dt]
    tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6 = (
                        evolution(*simulation_params, physical_params,
                                  signal_params, F, file_name = None))

    # compute the relative displacement (similar to the LVDT reading)
    x0 = np.sin(w*tt)       # ground motion
    l1 = x1 - x0            #F0_LVDT
    l2 = x2 - x1            #F1_LVDT
    l3 = x3 - x2            #F2_LVDT
    l4 = x4 - x3            #F3_LVDT
    l5 = x5 - x4            #F4_LVDT
    l7 = x5 - x0            #F7_LVDT

    sumL7 = l1+l2+l3+l4+l5

    # ------------------------------Plot results-------------------------------#
    #Plot displacements
    #fig = plt.figure(figsize=(5,5))
    plt.title('Time evolution for SR', size=13)
    plt.xlabel('Time [s]', size=12)
    plt.ylabel('x [m]', size=12)
    plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='pink', label='x$_1$, M$_1$')
    plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='black', label='x$_2$, M$_2$')
    plt.plot(tt, x3, linestyle='-', linewidth=1, marker='', color='red', label='x$_3$, M$_3$')
    plt.plot(tt, x4, linestyle='-', linewidth=1, marker='', color='green', label='x$_4$, M$_4$')
    plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x$_5$, M$_7$')
    plt.plot(tt, x6, linestyle='-', linewidth=1, marker='', color='steelblue', label='x$_6$, M$_{pl}$')
    plt.legend()

    plt.show()

    #Plot relative diplacements
    #fig = plt.figure(figsize=(5,5))
    plt.title('Time evolution for SR', size=13)
    plt.xlabel('Time [s]', size=12)
    plt.ylabel('$\Delta x$ [m]', size=12)
    plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()
    plt.plot(tt, l1, linestyle='-', linewidth=1, marker='', color='pink', label='L$_0$, F0_LVDT')
    plt.plot(tt, l2, linestyle='-', linewidth=1, marker='', color='black', label='L$_1$, F1_LVDT')
    plt.plot(tt, l3, linestyle='-', linewidth=1, marker='', color='red', label='L$_2$, F2_LVDT')
    plt.plot(tt, l4, linestyle='-', linewidth=1, marker='', color='green', label='L$_3$, F3_LVDT')
    plt.plot(tt, l5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='L$_4$, F4_LVDT')
    plt.plot(tt, l7, linestyle='-', linewidth=1, marker='', color='steelblue', label='L$_7$, F7_LVDT')
    #plt.plot(tt, sumL7, linestyle='-', linewidth=1, marker='', color='orange', label='sum')
    plt.legend()

    plt.show()

    #Save the plot in the results dir
    #out_name = os.path.join(results_dir, "filename.png")
    #plt.savefig(out_name)
    #plt.show()

