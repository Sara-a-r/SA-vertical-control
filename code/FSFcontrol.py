"""
This code models the dynamic behavior of a controlled system consisting of six
masses connected by springs and subjected to viscous damping. The control
implemented on the system is the Full-State-Feedback.
The temporal evolution of the controlled system is represented by an ARMA
(Auto-Regressive Moving Average) model, which is used to predict the future
positions and velocities of the masses based on their current states.
It also computes the Transfer Function of the controlled system.

The script performs the following functions:
1. Implement the Full-State-Feedback with the pole placement technique
1. Compute the Transfer Function of the controlled system in state-space design
and calculate its poles.
6. Evolves the system over time using the evolution function, which applies the
AR model.
7. Plots the results, showing the temporal evolution of the system,the Transfer
function and the poles in the s-plane when the system is controlled.
8. Save data of the controlled Transfer Function.

The script is structured to allow easy modification of the system's parameters,
such as mass, spring constants, damping coefficients, and the characteristics
of the control force. This flexibility makes it suitable for simulating a wide
range of physical systems that can be modeled with an ARMA approach.
"""

import os
from scipy.linalg import eig
import control as ct
import numpy as np
import matplotlib.pyplot as plt


#----------------------------Setup the main directories------------------------#
script_dir = os.getcwd()                            #define current dir
main_dir = os.path.dirname(script_dir)              #go up of one directory
results_dir = os.path.join(main_dir, "figure")      #define figure dir
data_dir = os.path.join(main_dir, "data")           #define data dir

if not os.path.exists(results_dir):                 #if the directory does not
    os.mkdir(results_dir)                           # exist create it


if not os.path.exists(data_dir):
    os.mkdir(data_dir)


#--------------------------Transfer Function controlled------------------------#
def StateSpaceMatrix( M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3,
                      g4, g5, g6):
    """
     Defines the state-space representation of the system.

     Parameters
     ----------
     M1, M2, M3, M4, M5, M6 : float
        Masses of the system components.
     K1, K2, K3, K4, K5, K6 : float
        Spring constants of the system.
     g2, g3, g4, g5, g6 : float
        Damping coefficients for the viscous friction force.

     Returns
     -------
     tuple
         Matrices A, B, C, D representing the state-space model.
     """
    # define the matrices of the system from the state-space equations
    Id = np.eye(6)
    V = np.array([[-(g2 / M1), g2 / M1, 0, 0, 0, 0],
                  [g2 / M2, -(g2 + g3) / M2, g3 / M2, 0, 0, 0],
                  [0, g3 / M3, -(g3 + g4) / M3, g4 / M3, 0, 0],
                  [0, 0, g4 / M4, -(g4 + g5) / M4, g5 / M4, 0],
                  [0, 0, 0, g5 / M5, -(g5 + g6) / M5, g6 / M5],
                  [0, 0, 0, 0, g6 / M6, -g6 / M6]])
    X = np.array([[-(K1 + K2) / M1, K2 / M1, 0, 0, 0, 0],
                  [K2 / M2, -(K2 + K3) / M2, K3 / M2, 0, 0, 0],
                  [0, K3 / M3, -(K3 + K4) / M3, K4 / M3, 0, 0],
                  [0, 0, K4 / M4, -(K4 + K5) / M4, K5 / M4, 0],
                  [0, 0, 0, K5 / M5, -(K5 + K6) / M5, K6 / M5],
                  [0, 0, 0, 0, K6 / M6, -K6 / M6]])
    A = np.block([[V, X],
                  [Id, 0 * Id]])

    B = np.array([[K1/M1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    C = np.block([0 * Id, Id])
    D = np.array([[0], [0], [0], [0], [0], [0]])
    return A, B, C, D

def Kmatrix(A, B, C, D, dpoles):
    """
    Calculates the feedback gain matrix K.

    Parameters
    ----------
    A, B, C, D : ndarray
        State-space matrices.
    dpoles : list
        Desired poles for the system.

    Returns
    -------
    ndarray
        Gain matrix K.
    """

    # define desired poles
    desired_poles = np.array(dpoles)

    # Compute the gain. Use ct.place() in a broader case. It implements the Tits
    # and Yang algorithm. It handles SISO, MISO and MIMO systems
    k = ct.acker(A, B, desired_poles)
    return k

def TransferFunc(w, A, B, C, D, k, N, fmin, fmax):
    """
    Computes the transfer function of the controlled system.

    Parameters
    ----------
    w : ndarray
        Array of angular frequencies.
    A, B, C, D : ndarray
        State-space matrices.
    k : ndarray
        Gain matrix.
    N : float
        Scaling factor for the reference input.

    Returns
    -------
    tuple
        Transfer function matrix H and array of poles.
    """
    # Initialize the transfer matrix: the matrix has 6 rows (like the number of
    # output), and len(w) columns (all the range of frequencies). In each row
    # there is the TF of a single output.
    H = np.zeros((6, len(w)),dtype = 'complex_')

    w_min = 2 * np.pi * fmin
    w_max = 2 * np.pi * fmax
    # Compute the transfer function
    for i in range(len(w)):
        # apply the controller only in the frequency range fmin<f<fmax
        if (w[i] < w_min or w[i] > w_max):
            AA = A
            Nbar = 1
        else:
            AA = (A - (B @ k))
            Nbar = N

        # array of len=number of output whose elements are the values of the TF
        # of each output at given frequency w
        H_lenOUT = C @ np.linalg.inv((1j*w[i])*np.eye(12) - AA ) @ B * Nbar
        H_lenOUT = H_lenOUT.squeeze() # remove empty dimension

        #store each value of the TF in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]
        H[5][i] = H_lenOUT[5]

    # Compute poles
    poles, _ = eig((A-(B@k)))

    return H, poles

#------------------------Time evolution using ARMA model-----------------------#
def FSF_AR_model(y, A, B, K, N, r):
    """
    Computes the next state of the system, controlled by a Full State Feedback,
     using the ARMA model.

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
    K : ndarray
        Gain matrix.
    r : float
        Reference input.
    N : float
        Scaling factor for the reference input.

    Returns
    -------
    ndarray
        Next state vector of the system.
    """
    return (A - np.outer(B,K)) @ y + B * N * r  # Return the next temporal state

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

    B = np.array((dt*K1 / M1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    return A, B

def evolution(evol_method, Nt_step, dt, physical_params, k, control_params,
              file_name = None):
    """
        Simulates the temporal evolution of the system under control.

    Parameters
    ----------
    evol_method : function
        The function used to evolve the system (e.g. Euler or ARMA methods).
    Nt_step : int
        Number of temporal steps to simulate.
    dt : float
        Time step size.
    physical_params : list
        List of physical parameters for the system.
    k : ndarray
        Gain matrix.
    control_params : list
        List of control parameters including scaling factor N and reference r.
    file_name : str, optional
        Name of the file to save simulation data. Default is None.

    Returns
    -------
    tuple
        Time grid and arrays of velocities and positions for each mass.
    """
    # Initialize the problem
    tmax = dt * Nt_step  # total time of simulation
    tt = np.arange(0, tmax, dt)  # temporal grid
    y0 = np.array(
        (0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))  # initial condition
    y_t = np.copy(y0)             # create a copy to evolve it in time

    # Initialize lists for velocities and positions
    v1, v2, v3, v4, v5, v6 = [[], [], [], [], [], []]
    x1, x2, x3, x4, x5, x6 = [[], [], [], [], [], []]

    # compute the matrices of the system
    A, B = matrix(*physical_params)

    # temporal evolution of the controlled system
    i = 0
    for t in tt:
        i = i + 1
        y_t = evol_method(y_t, A, B, k, *control_params)   # step n+1
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
    gamma = [5, 5, 5, 5, 5]  # viscous friction coeff [kg/m*s]
    M = [160, 125, 120, 110, 325, 82]  # filter mass [Kg]
    K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]

    # create the array of frequencies in which evaluate the TF
    # Use arange() if you want to customize the frequency range
    #f = np.arange(1e-2,1e1,0.003)
    #w = 2*np.pi*f

    # Use loadtxt() to use frequencies extrapolated from the reference
    # measurement of the TF of the SR given by P.Ruggi. In this way in the code
    # 'rms' you can compute the product between the ASD of the seism and the TF
    # of the system at the same frequencies.
    freq = np.loadtxt('../data/freq.txt', unpack=True)
    wn = 2 * np.pi * freq

    # compute the state space matrices
    A, B, C, D = StateSpaceMatrix(*M, *K, *gamma)

    # define the desired poles
    dpoles = [-1.0302313 + 8.44834539j, -1.0302313 - 8.44834539j,  # 1.344Hz
              -0.1730504 + 7.11085663j, -0.1730504 - 7.11085663j,  # 1.131Hz
              -2.0692883 + 4.39542468j, -2.0692883 - 4.39542468j,  # 0.699Hz
              -0.159586 + 0.66927877j, -0.159586 - 0.66927877j,    # 0.106Hz
              -1.09151 + 2.24037402j, -1.09151 - 2.24037402j,      # 0.356Hz
              -0.1315067 + 3.00566475j, -0.1315067 - 3.00566475j]  # 0.478Hz

    # compute the gain matrix. k is a (1,n) matrix (in this case (1, 12) matrix,
    # i.e. row vector)
    k = Kmatrix(A, B, C, D, dpoles)

    # control parameters
    fmin, fmax = 0.05, 1.6  # frequency range to control
    r = 0   # reference input
    N = -1 / (C @ np.linalg.inv(A - (B @ k)) @ B)[0]  # scale the ref input

    # Simulation
    physical_params = [*M, *K, *gamma, dt]
    control_params = [N, r]
    simulation_params = [FSF_AR_model, Nt_step, dt]
    tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6 = (
            evolution(*simulation_params, physical_params, k,
                      control_params, file_name = None))

    # Compute the transfer function
    Tf, poles = TransferFunc(wn, A, B, C, D, k, N, fmin, fmax)
    # Compute the magnitude of the transfer function
    H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)

    #save H values in a file (for the first and last mass)
    np.savetxt(os.path.join(data_dir, 'TF_FSFcontrol.txt'),
       np.column_stack((freq, H[0], H[5])), header='f[Hz], H(x1/x0), H(xpl/x0)')

    # Extract imaginary and real part of poles
    real_p = np.real(poles)
    imag_p = np.imag(poles)

    print("Poles: ", poles)
    print('Real part is: sigma = ', real_p)
    print('Imaginary part is: w = ', imag_p)
    print('Normal frequencies are:', (imag_p[imag_p > 0] / (2 * np.pi)))

    #----------------------------------Plot poles------------------------------#
    plt.figure(figsize=(6,5))
    plt.title('Poles of the controlled system in $s$-plane', size=13)
    plt.xlabel('$\sigma$ (real part)', size=12)
    plt.ylabel('$j \omega$ (imaginary part)', size=12)
    plt.grid(True, which='both',ls='-', alpha=0.3, lw=0.5)

    plt.axhline(y=0, linestyle=':', color='black', linewidth=1.1)
    plt.axvline(x=0, linestyle=':', color='black', linewidth=1.1)
    plt.scatter(real_p, imag_p, marker='x', color='steelblue', linewidths=1)

    # --------------------------------Plot TF----------------------------------#
    #load data TF not controlled
    _, Tfnc_1, Tfnc_pl = np.loadtxt('../data/TFnoControl.txt',unpack=True)

    fig = plt.figure(figsize=(9, 5))
    plt.title('Transfer function (FSF control)', size=13)
    plt.xlabel('Frequency [Hz]', size=12)
    plt.ylabel('|x$_{out}$/x$_0$|', size=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both',ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    plt.plot(freq, Tfnc_pl, linestyle='-', linewidth=1, marker='', color='steelblue', label='no control')
    #plt.plot(freq, H[0], linestyle='-', linewidth=1, marker='', color='steelblue', label='FSF control')
    plt.plot(freq, H[5], linestyle='-', linewidth=1, marker='', color='coral', label='FSF, x$_{out}$ = x$_{pl}$')
    plt.legend()

    # ------------------------------Plot time evolution------------------------#
    #load data time evol not controlled
    #_, xnc_1, xnc_pl = np.loadtxt('../data/timeEvol_noControl.txt',unpack=True)

    fig = plt.figure(figsize=(5, 5))
    plt.title('Time evolution (FSF control)', size=13)
    plt.xlabel('Time [s]', size=12)
    plt.ylabel('x [m]', size=12)
    plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, M$_1$')
    plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='black', label='x2, M$_2$')
    plt.plot(tt, x3, linestyle='-', linewidth=1, marker='', color='red', label='x3, M$_3$')
    plt.plot(tt, x4, linestyle='-', linewidth=1, marker='', color='green', label='x4, M$_4$')
    plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x7, M$_7$')
    plt.plot(tt, x6, linestyle='-', linewidth=1, marker='', color='coral', label='x$_{pl}$, M$_{pl}$')

    #plt.plot(tt, xnc_pl, linestyle='-', linewidth=1, marker='', color='steelblue', label='no control, x$_{pl}$')
    plt.legend()
    plt.show()
