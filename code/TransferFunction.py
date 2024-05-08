"""
This code implements the transfer function of a system composed of 6 masses and 6 springs.
On the system acts a viscous friction force with a coefficient gamma.
The system is described in time domain by the equation dx/dt = A*x + B*u, y = C*x + D*u and
the transfer function is given by H(s) = C*(s*Id - A)^(-1) * B + D.
The code returns the transfer function for all the output (in this case X1/X0, X2/X0, X3/X0,
X4/X0, X5/X0) and also compare the results with real-word data.
"""

import numpy as np
import matplotlib.pyplot as plt

#-----------------------Transfer Function----------------------#
def TransferFunc (w, M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3, g4,
                  g5, g6):
    """
        Calculates the transfer function of the system and its poles when it is not
        controlled.

        Parameters
        ----------
        w : ndarray
            Array of angular frequencies.
        M1, M2, M3, M4, M5, M6 : float
            Masses of the system components.
        K1, K2, K3, K4, K5, K6 : float
            Spring constants of the system.
        g2, g3, g4, g5, g6 : float
            Damping coefficients for the viscous friction force.

        Returns
        -------
        tuple
            Transfer function matrix H.
        """
    #define the matrices of the system from the state-space equations
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
                  [0, 0, 0, 0, K6/ M6, -K6 / M6]])
    A = np.block([[V, X],
                  [Id, 0 * Id]])

    B = np.array((K1 / M1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    C = np.block([0*Id, Id])

    # Initialize the transfer matrix: the matrix has 6 rows (like the number of
    # output), and len(w) columns (all the range of frequencies). In each row
    # there is the TF of a single output.
    H = np.zeros((6, len(w)),dtype = 'complex_')
    for i in range(len(w)):
        # array of len=number of output whose elements are the values of the TF
        # of each output at given frequency w
        H_lenOUT = C @ np.linalg.inv((1j*w[i])*np.eye(12) - A) @ B

        #store each value of the TF in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]
        H[5][i] = H_lenOUT[5]
    return H


if __name__ == '__main__':
    # create an array of frequencies
    f = np.arange(1e-2,1e1,0.003)
    w = 2*np.pi*f

    # Parameters of the system
    gamma = [5, 5, 5, 5, 5]  # viscous friction coeff [kg/m*s]
    M = [160, 125, 120, 110, 325, 82]  # filter mass [Kg]
    K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]

    # compute the transfer function
    Tf = TransferFunc(w, *M, *K, *gamma)

    # compute the magnitude of the transfer function
    H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)

    #------------------------------Load real data------------------------------#
    ff, Tf_m = np.loadtxt('../../data/SR_verticalTF_Ruggi.txt',unpack=True)
    Tf_m = Tf_m * (1/402)  #rescaling factor

    # -------------------------------Plot results------------------------------#

    fig = plt.figure(figsize=(9, 5))
    plt.title('Data vs model: TF of SR chain',size=13)
    plt.xlabel('Frequency [Hz]', size=12)
    plt.ylabel('|x$_{out}$/x$_0$|', size=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both',ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    plt.plot(f, H[0], linestyle='-', linewidth=1, marker='', color='steelblue', label='output $x_1$')
    #plt.plot(f, H[1], linestyle='-', linewidth=1, marker='', color='violet', label='output $x_2$')
    #plt.plot(f, H[2], linestyle='-', linewidth=1, marker='', color='black', label='output $x_3$')
    #plt.plot(f, H[3], linestyle='-', linewidth=1, marker='', color='red', label='output $x_4$')
    #plt.plot(f, H[4], linestyle='-', linewidth=1, marker='', color='lime', label='output $x_7$')
    #plt.plot(f, H[5], linestyle='-', linewidth=1, marker='', color='pink', label='output $x_{pl}$')

    plt.plot(ff, Tf_m, linestyle='-', linewidth=1, marker='', color='tomato', label='open loop data')
    plt.legend()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "Tf_2M2K.png")
    #plt.savefig(out_name)
    plt.show()

