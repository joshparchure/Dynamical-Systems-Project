import numpy as np

from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import welch
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
from scipy.sparse import diags, lil_matrix
from scipy.linalg import solve_banded

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

import time


#---------------------------- Code for Part 1 ----------------------------#
def simulate1(n,X0=-1,Nt=2000,tf=1000,g=1.0,mu=0.0,flag_display=False):
    """
    Simulation code for part 1
    Input:
        n: number of ODEs
        X0: initial condition if n-element numpy array. If X0 is an integer, the code will set 
        the initial condition (otherwise an array is expected).
        Nt: number of time steps
        tf: final time
        g,mu: model parameters
        flag_display: will generate a contour plot displaying results if True
    Output:
    t,x: numpy arrays containing times and simulation results
    """

    def model(t, X, n, g, mu):
        """
        Defines the system of ODEs 
        
        Input:
        - t: time 
        - X: array of variables [x_0, x_1, ..., x_{n-1}]
        - g, mu: scalar model parameters
        - n: number of ODEs (size of the system)
         
        Returns:
        - dXdt: array of derivatives [dx_0/dt, dx_1/dt, ..., dx_{n-1}/dt]
        """
        dXdt = np.zeros(n)
        dXdt[0] = X[0]*(1-g*X[-2]-X[0]-g*X[1]+mu*X[-3])
        dXdt[1] = X[1]*(1-g*X[-1]-X[1]-g*X[2])
        dXdt[2:n-1] = X[2:n-1]*(1-g*X[0:n-3]-X[2:n-1]-g*X[3:n])
        dXdt[-1] = X[-1]*(1-g*X[-3]-X[-1]-g*X[0])
        return dXdt

    # Parameters
    t_span = (0, tf)  # Time span for the integration
    if type(X0)==int: #(modified from original version)
        X0 = 1/(2*g+1)+0.001*np.random.rand(n)  # Random initial conditions, modify if needed

    # Solve the system
    solution = solve_ivp(
        fun=model,
        t_span=t_span,
        y0=X0,
        args=(n, g, mu),
        method='BDF',rtol=1e-10,atol=1e-10,
        t_eval=np.linspace(t_span[0], t_span[1], Nt)  # Times to evaluate the solution
    )

    t,x = solution.t,solution.y #in original version of code was inside if-block below
    if flag_display:
        # Plot the solution
        plt.contour(t,np.arange(n),x,20)
        plt.xlabel('t')
        plt.ylabel('i')
    return t,x

def part1q1a(n,g,mu,T):
    """Part 1, question 1 (a)
    Input:
    n: number of ODEs
    g,mu: model parameters
    T: time at which perturbation energy ratio should be maximized
    
    Output:
    xbar: n-element array containing non-trivial equilibrium solution
    xtilde0: n-element array corresponding to computed initial condition
    eratio: computed maximum perturbation energy ratio
    """
    #use/modify code below as needed:
    xbar = np.zeros(n)
    xtilde0 = np.zeros(n)
    eratio = 0.0 #should be modified below

    #add code here

    # We will solve the equation Bx=1 to get the equilibrium solution
    # x is a vector representing the variables x_0 ,..., x_n-1
    # 1 is a vector of n 1s
    # B is a matrix containing 0s,1s,gammas and mus. It is defined from the RHS of the system of ODEs excluding the x_{i} prefactors
    # Excluding the prefactors ensures we have non-trivial solutions
    # Generated using chatgpt-4o

    B = np.zeros((n,n))

    for i in range(2, n - 1):
        B[i, i] = 1
        B[i, i - 2] = g
        B[i, i + 1] = g

    B[0, 0] = 1
    B[0, n - 2] = g
    B[0, 1] = g
    B[0, n - 3] = -mu

    B[1, 1] = 1
    B[1, n - 1] = g
    B[1, 2] = g

    B[n - 1, n - 1] = 1
    B[n - 1, n - 3] = g
    B[n - 1, 0] = g

    xbar= np.linalg.solve(B,(np.ones(n)))

    # We define a function that returns the jacobian for this system evaluated at a given point x
    # Generated using claude 3.5 sonnet
    def odes_jacobian(x):
        n = len(x)
        J = np.zeros((n, n))

        J[0, 0] = 1 - 2*x[0] - g*(x[n-2]+x[1]) + mu*x[n-3]
        J[0, n-2] = -g*x[0] 
        J[0, 1] = -g*x[0]
        J[0, n-3] = mu*x[0]

        J[1, 1] = 1 - 2*x[1] - g*(x[n-1]+x[2])
        J[1, n-1] = -g*x[1]
        J[1, 2] = -g*x[1]

        J[n-1, n-1] = 1 - 2*x[n-1] - g*(x[n-3]+x[0])
        J[n-1, n-3] = -g*x[n-1]
        J[n-1, 0] = -g*x[n-1]

        for i in range(2, n-1):
            J[i, i] = 1 - 2*x[i] - g*(x[i-2] + x[i+1]) 
            J[i, i-2] = -g*x[i]
            J[i, i+1] = -g*x[i]

        return J

    # Matrix A is the exponential of the Jacobian times the final time T
    J = odes_jacobian(xbar)
    A = expm(J*T)

    # Here we find the leading eigenvalue and eigenvector as used in the maximum growth problem
    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    leading_eigenvector = eigenvectors[:, np.argmax(np.abs(np.real(eigenvalues)))]
    # The maximum growth problem finds the unit vector which maximises the growth; we want a small perturbation so we scale it down
    # Trying a couple powers of 10 gives this as the closest one for part1q1b
    xtilde0 = np.real(leading_eigenvector*1e-6)
    eratio = np.max(np.abs(np.real(eigenvalues)))

    return xbar,xtilde0,eratio,J


def part1q1b():
    """Part 1, question 1(b): 
    """
    n = 19
    g = 1.2
    mu = 2.5
    T = 50

    #add code here
    xbar, xtilde0, eratio, _ = part1q1a(n,g,mu,T)
    # Call simulate1 with sufficiently large Nt for accuracy. This solves system of ODEs.
    t_sim, x_sim = simulate1(n,xbar+xtilde0,3000000,T,g,mu,False)

    # Find the value of xtilde(T) i.e. at the final timestep by subtracting the equilibrium solution
    xtildeT_sim = x_sim[:,-1] - xbar
    # Calculate the energy ratio (for any vector x we have x^T x = |x|^2)
    eratio_sim = np.linalg.norm(xtildeT_sim)**2/np.linalg.norm(xtilde0)**2
    # Calculate the difference relative to the ratio calculated by simulate1
    relative_error = np.linalg.norm(eratio_sim-eratio)/eratio_sim

    return relative_error #modify if needed

def part1q1c():
    """Part 1, question 1(c): 
    """
    n = 19
    g = 2
    mu = 0

    #add code here

    # Generated using ChatGPT-4o
    # Define range of T values in (0,50]scip
    T_values = np.linspace(0.1, 50, 1000)
    energy_ratios = []

    # Loop over T values and calculate the maximum energy ratio each time
    for T in T_values:
        xbar, xtilde0, eratio, Jac = part1q1a(n, g, mu, T)
        energy_ratios.append(eratio)

    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    popt, pcov = curve_fit(exponential_func, T_values , energy_ratios)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(T_values, energy_ratios, label='part1q1a output', color='b')
    plt.plot(T_values[::5], exponential_func(T_values[::5], *popt), '.',color='r' ,label='Fit')
    plt.xlabel('T')
    plt.ylabel('Max Energy Ratio')
    plt.title('Variation of Max Energy Ratio with T (g=2, mu=0, n=19)')
    plt.grid(True)
    plt.legend()
    plt.show()

    start_index = 950
    popt2, pcov2 = curve_fit(exponential_func, T_values[start_index:], energy_ratios[start_index:])

    # Zooming in on the area of large growth
    plt.figure(figsize=(10, 6))
    plt.plot(T_values[start_index:], energy_ratios[start_index:], label='Max Energy Ratio vs T', color='b')
    plt.plot(T_values[start_index:], exponential_func(T_values[start_index:], *popt2), 'o', color='r', label='Fit')
    plt.xlabel('T')
    plt.ylabel('Max Energy Ratio')
    plt.title('Variation of Max Energy Ratio with T (g=2, mu=0, n=19)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot residuals just as an extra check to see if the exp fit is good
    residuals = energy_ratios[start_index:] - exponential_func(T_values[start_index:], *popt)
    plt.plot(T_values[start_index:], residuals)
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.show()

    # Plot a semilogy plot and determine parameters for the linear regression on the log energy ratios
    slope, intercept, r_value, p_value, std_err = linregress(T_values, np.log(energy_ratios))
    plt.figure(figsize=(10, 6))
    plt.semilogy(T_values, energy_ratios, color='b')
    plt.xlabel('T')
    plt.ylabel('log Max Energy Ratio')
    plt.title('Semilog plot to show Variation of Max Energy Ratio with T (g=2, mu=0, n=19)')
    plt.grid(True)
    plt.show()
    print('gradient= ',slope)
    print('intercept= ',intercept)

    eigenvalues, eigenvectors = np.linalg.eig(Jac)
    print(np.max(np.real(eigenvalues)))
    # print()
    # print(eigenvectors)

    # t_sim, x_sim = simulate1(n,xbar+xtilde0,3000000,50,g,mu,False)
    # xtildes_sim = x_sim - xbar[:, np.newaxis]
    # energies = np.linalg.norm(xtildes_sim, axis=0)**2
    # eratios_sim = energies / energies[0]

    # plt.figure(figsize=(10, 6))
    # plt.plot(t_sim, eratios_sim, label='Max Energy Ratio vs T', color='b')
    # plt.plot(T_values, energy_ratios, label='Max Energy Ratio vs T', color='r')
    # plt.xlabel('T')
    # plt.ylabel('Max Energy Ratio')
    # plt.title('Variation of Max Energy Ratio with T (g=2, mu=0, n=19)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Return data for analysis if needed
    return T_values, energy_ratios

    return None #modify if needed

def part1q2():
    """Part 1, question 2: Compute correlation dimension from time series."""
    # n = 59
    # g = 1
    # mu = 0
    # Nt = 200000
    # tf = 5000
    # transient_cutoff = 0.4  # Fraction of time to exclude as transient

    # Function that gives us the relevant post-transient dynamics only
    def simulate_and_trim(n,g,mu,Nt,tf,transient_cutoff):
    
        t, x = simulate1(n=n, Nt=Nt, tf=tf, g=g, mu=mu, flag_display=True)
        
        # Define the fraction of values to cutoff from the data, removing the transient
        cutoff_index = int(transient_cutoff * len(t))
        t_trimmed = t[cutoff_index:]
        x_trimmed = x[:, cutoff_index:]

        return t_trimmed, x_trimmed
    
    # def find_dom_freq(t,x):
    #     dt = t[1] - t[0]
    #     fxx,Pxx = welch(x,fs=1/dt)
    #     plt.figure()
    #     plt.semilogy(fxx,Pxx)
    #     plt.xlabel(r'$f$')
    #     plt.ylabel(r'$P_{xx}$')
    #     plt.grid()
    #     f = fxx[Pxx==Pxx.max()][0]
    #     print("f=",f)
    #     print("dt,1/f=",dt,1/f)
    #     plt.show()

    #     return None

    # Function to generate the correlation dimension
    # Generated using ChatGPT-4o
    def find_corr_dim(t,x,lower_limit,upper_limit):
        # Compute pairwise distances
        distances = pdist(x.T)  # Transpose for (Nt - cutoff) samples
        print(f"Min distance: {np.min(distances):.2e}, Max distance: {np.max(distances):.2e}")

        # Define epsilon range for correlation sum
        epsilons = np.logspace(np.log10(np.min(distances) / 2), np.log10(np.max(distances) * 2), 50)
        correlation_sums = []

        # Find the correlation sums
        for epsilon in epsilons:
            count = np.sum(distances < epsilon)
            correlation_sums.append(count / len(distances))  # Normalize by total number of pairs
        
        correlation_sums = np.array(correlation_sums)

        # Log-log plot data
        log_eps = np.log(epsilons)
        log_corr = np.log(correlation_sums)
        
        # Look at appropriate values
        valid_indices = (correlation_sums > lower_limit) & (correlation_sums < upper_limit)
        log_eps_valid = log_eps[valid_indices]
        log_corr_valid = log_corr[valid_indices]

        # Fit a linear region
        slope, intercept, r_value, p_value, std_err = linregress(log_eps_valid, log_corr_valid)
        print(f"Estimated correlation dimension: {slope:.4f}")
        
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.loglog(epsilons, correlation_sums, 'o-', label='Correlation sum')
        plt.plot(np.exp(log_eps_valid), np.exp(slope * log_eps_valid + intercept), 'r--', label=f'Fit: slope={slope:.2f}')
        plt.xlabel('log(ε)')
        plt.ylabel('log(C(ε))')
        plt.title('Log-log plot of Correlation Sum against Epsilon')
        plt.legend()
        plt.grid()
        plt.show()

    # Function to plot Fourier coefficients (up to a given kend) for given data
    # Generated using ChatGPT-4o
    def plot_fourier_coeff(t,x,n,kend):

        plt.figure(figsize=(20,10))
        k = np.arange(0, len(t))

        for i in range(n): 
            fft_x = np.abs(np.fft.fft(x[i]))
            plt.scatter(k[1:kend],fft_x[1:kend], marker = '.')
            plt.xlabel("$k$")
            plt.ylabel(r"Fourier coefficient $|c_k|$")

        max_k = np.argmax(fft_x[1:])
        print(max_k)
        
        plt.title(r"Graph to show the $k^{th}$ Fourier coefficients for $k \geq 1$ when $n = $" + str(n))
        plt.show()

        return None

    # Function to generate a time series plot
    # Ended up not being useful - only required for it for n=9 and this required individual changes
    def general_plot_time_series(t,x,n):
        plt.figure()
        for i in range(n): 
            plt.plot(t,x[i],label=f'$x_{i}$')
        plt.xlabel(r"$t$")
        plt.ylabel("$x_{i}(t)$")
        plt.title(r"Time series plot for $n = $" + str(n))
        plt.show()

        return None
    
    # Code to generate the time series when n is 9
    def plot_time_series_n9(t,x):

        # plt.figure()

        # for i in range(n): 
        #     # plt.plot(t,x[i],label=f'$x_{i}$',linestyle=line_styles[i // 3], color=colors[i // 3])
        #     # plt.plot(t,x[i],label=f'$x_{i}$')
        #     line_style = line_styles[i // 3]
        #     marker = markers[i // 3]
        #     if i >5:
        #         plt.plot(t, x[i], label=f'$x_{i}$', linestyle='none',marker=marker,markersize=1.5,alpha=0.5)
        #     elif i in [3,4,5]:
        #         plt.plot(t, x[i], label=f'$x_{i}$', linestyle=line_style,alpha=0.5)
        #     else:
        #         plt.plot(t, x[i], label=f'$x_{i}$', linestyle=line_style)

        fig, ax = plt.subplots()

        for i in [0,1,2]:
            ax.plot(t, x[i], label=f'$x_{i}$',linestyle='-')
        tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-1.5, units='points')
        for i in [3,4,5]:
            ax.plot(t, x[i], transform=tr, label=f'$x_{i}$',linestyle='--')
        tr2 = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=1.5, units='points')
        for i in [6,7,8]:
            ax.plot(t, x[i], transform=tr2, label=f'$x_{i}$',linestyle=':')

        plt.xlabel(r"$t$")
        plt.ylabel("$x_{i}(t)$")
        plt.title(r"Time series plot for $n = $" + str(9))
        plt.legend()
        plt.show()

        peaks, _ = find_peaks(x[0])
        # Calculate time differences between peaks
        time_differences = np.diff(t[peaks])
        # Estimate the time interval
        estimated_interval = np.mean(time_differences)
        print("Estimated time interval:", estimated_interval)

        return None

    # Function to generate a plot of successive maxima
    # Generated using ChatGPT-4o
    def plot_successive_maxima(t, x):
        """
        Plot successive maxima for each variable in the system.

        Input:
            t: Array of time points.
            x: Array of system states (variables over time).
        """
        plt.figure(figsize=(10, 8))
        for i in range(x.shape[0]):
            # Find peaks for the i-th variable
            peaks, _ = find_peaks(x[i])
            maxima = x[i, peaks]

            # Plot k-th vs (k+1)-th maxima
            plt.scatter(maxima[:-1], maxima[1:])

        plt.xlabel('k-th Maxima')
        plt.ylabel('(k+1)-th Maxima')
        plt.title('Successive Maxima Plot')
        plt.grid()
        plt.show()

    # # t1, x1 = simulate_and_trim(n=9,g=1,mu=0,Nt=20000,tf=1000,transient_cutoff=0.2)
    # # xdelay = apply_time_delay(x1, 1)
    # # print(x1)
    # # print(x1.shape)
    # # print()
    # # print(xdelay)
    # # print(xdelay.shape)
    # # correlation_matrix = np.corrcoef(xdelay)
    # # print("Correlation Matrix:")
    # # print(correlation_matrix)

    # t1, x1 = simulate_and_trim(n=9,g=1,mu=0,Nt=20000,tf=1000,transient_cutoff=0.2)
    # tau = find_dom_freq(t1,x1[0,:])/5
    # print(find_dom_freq(t1,x1[0,:]))
    # correlation_matrix = np.corrcoef(x1)
    # print(correlation_matrix[0])

    # Function to plot phase plot for 3 coordinates of a simuaklted system
    # Generated using ChatGPT-4o
    def phase_plot(x,coord1,coord2,coord3):
        x_coord1 = x[:, coord1]
        x_coord2 = x[:, coord2]
        x_coord3 = x[:, coord3]

        # Create a 3D phase plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_coord1, x_coord2, x_coord3, label='Phase Plot')

        # Add labels and title
        ax.set_xlabel(f'Component {coord1}')
        ax.set_ylabel(f'Component {coord2}')
        ax.set_zlabel(f'Component {coord3}')
        ax.set_title('Phase Plot of System')
        # Show legend and plot
        ax.legend()
        plt.show()

    # t1, x1 = simulate_and_trim(n=9,g=1,mu=0,Nt=20000,tf=1000,transient_cutoff=0.2)
    # find_corr_dim(t1,x1,lower_limit=5*1e-4,upper_limit=1)
    # plot_fourier_coeff(t1,x1,9,400)
    # plot_time_series_n9(t1,x1)

    # t2a, x2a = simulate_and_trim(n=20,g=1,mu=0,Nt=20000,tf=10000,transient_cutoff=0.4)
    # find_corr_dim(t2a,x2a,lower_limit=1e-4,upper_limit=5*1e-1)
    # t2b, x2b = simulate_and_trim(n=20,g=1,mu=0,Nt=200000,tf=10000,transient_cutoff=0.4)
    # plot_successive_maxima(t2b,x2b)
    # plot_fourier_coeff(t2b,x2b,20,2000)

    # t3a, x3a = simulate_and_trim(n=59,g=1,mu=0,Nt=20000,tf=10000,transient_cutoff=0.4)
    # find_corr_dim(t3a,x3a,lower_limit=1e-6,upper_limit=1e-1)
    # t3b, x3b = simulate_and_trim(n=59,g=1,mu=0,Nt=200000,tf=10000,transient_cutoff=0.4)
    t3c, x3c = simulate_and_trim(n=59,g=1,mu=0,Nt=10000,tf=1000000,transient_cutoff=0.4)
    phase_plot(x3c,0,1,2)
    # plot_successive_maxima(t3b,x3b)
    # plot_fourier_coeff(t3b,x3b,59,2000)
    # correlation_matrix = np.corrcoef(x3b)
    # print(correlation_matrix[0])

    return None


#---------------------------- End code for Part 1 ----------------------------#


#---------------------------- Code for Part 2 ----------------------------#
def dualfd1(f):
    """
    Code implementing implicit finite difference scheme for special case m=1
    Implementation is not efficient.
    Input:
    f: n-element numpy array
    Output:
    df, d2f: computed 1st and 2nd derivatives
    """
    #parameters, grid
    n = f.size
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    
    #fd method coefficients
    #interior points:
    L1 = [7,h,16,0,7,-h]
    L2 = [-9,-h,0,8*h,9,-h]
    
    #boundary points:
    L1b = [1,0,2,-h]
    L2b = [0,h,-6,5*h]

    L1b2 = [2,h,1,0]
    L2b2 = [-6,-5*h,0,-h]

    A = np.zeros((2*n,2*n))
    #iterate filling a row of A each iteration
    for i in range(n):
        #rows 0 and N-1
        if i==0:
            #Set boundary eqn 1
            A[0,0:4] = L1b
            #Set boundary eqn 2
            A[1,0:4] = L2b
        elif i==n-1:
            A[-2,-4:] = L1b2
            A[-1,-4:] = L2b2
        else:
            #interior rows
            #set equation 1
            ind = 2*i
            A[ind,ind-2:ind+4] = L1
            #set equation 2
            A[ind+1,ind-2:ind+4] = L2

    # return A

    #set up RHS
    b = np.zeros(2*n)
    c31,c22,cb11,cb21,cb31,cb12,cb22,cb32 = 15/h,24/h,-3.5/h,4/h,-0.5/h,9/h,-12/h,3/h
    for i in range(n):
        if i==0:
            b[i] = cb11*f[0]+cb21*f[1]+cb31*f[2]
            b[i+1] = cb12*f[0]+cb22*f[1]+cb32*f[2]
        elif i==n-1:
            b[-2] =-(cb11*f[-1]+cb21*f[-2]+cb31*f[-3])
            b[-1] = -(cb12*f[-1]+cb22*f[-2]+cb32*f[-3])
        else:
            ind = 2*i
            b[ind] = c31*(f[i+1]-f[i-1])
            b[ind+1] = c22*(f[i-1]-2*f[i]+f[i+1])
    
    #return B
    out = np.linalg.solve(A,b)
    df = out[::2]
    d2f = out[1::2]
    return df,d2f


def fd2(f):
    """
    Computes the first and second derivatives with respect to x using second-order finite difference methods.
    
    Input:
    f: m x n array whose 1st and 2nd derivatives will be computed with respect to x
    
    Output:
    df, d2f: m x n arrays conaining 1st and 2nd derivatives of f with respect to x
    """

    m,n = f.shape
    h = 1/(n-1)
    df = np.zeros_like(f) 
    d2f = np.zeros_like(f)
    
    # First derivative
    # Centered differences for the interior 
    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)

    # One-sided differences at the boundaries
    df[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * h)
    df[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * h)
    
    # Second derivative 
    # Centered differences for the interior 
    d2f[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (h**2)
    
    # One-sided differences at the boundaries
    d2f[:, 0] = (2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]) / (h**2)
    d2f[:, -1] = (2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]) / (h**2)
    
    return df, d2f

def part2q1(f):
    """
    Part 2, question 1
    Input:
    f: m x n array whose 1st and 2nd derivatives will be computed with respect to x
    Output:
    df, d2f: m x n arrays conaining 1st and 2nd derivatives of f with respect to x
    computed with implicit fd scheme
    """
    #use code below if/as needed
    m,n = f.shape
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    df = np.zeros_like(f) #modify as needed
    d2f = np.zeros_like(f) #modify as needed

    #Add code here

    # Function that converts a matrix into matrix diagonal ordered form
    # No longer used since scipy.sparse can extract non-zero elements quickly - makes this redundant
    # def to_banded_matrix(a, l, u):

        # # Get the dimensions of the input matrix (see below)
        # _, n = a.shape

        # # Initialise the shape of the banded matrix from the original matrix
        # # We can see from examples in the documentation that we will have the same number of columns as the numebr of rows of the input matrix
        # # And we will have l+u+1 rows since these are the non-zero diagonals we want to store
        # ab = np.zeros((l + u + 1, n), dtype=a.dtype)

        # # Fill the banded matrix *efficiently*
        # # Iterate over all the non-zero diagonals 
        # for k in range(u, -l - 1, -1):
            # # Find the kth non-zero diagonal
            # diag = np.diag(a, k=k)
            # # Store each diagonal as the rows of the banded matrix, starting with the u^th upper diagonal and going down from there
            # row = u - k
            # # All unaffected entries remain 0 as required
            # ab[row, max(0, k):min(n, n + k)] = diag

        # return ab
    
    # Function to create the matrix A in SPARSE format (!!)
    def generate_A(n):
        # _,n = f.shape
        h = 1/(n-1)
        n = n*2

        # We fill the matrix A in the same way as implemented inefficiently in dualfd1
        # We do not use a for loop and instead set all the diagonals using numpy slicing

        # Initialise the diagonals with the correct shapes 
        maind = np.ones(n)
        ud1 = np.zeros(n - 1)
        ud2 = np.zeros(n - 2)
        ud3 = np.zeros(n - 3)
        ld1 = np.zeros(n - 1)
        ld2 = np.zeros(n - 2)
        ld3 = np.zeros(n - 3)

        # In dualfd1, using the 2 boundary equations at j=0, we set the first 4 elements of the 1st and 2nd rows
        # These elements are the coefficients of uij' uij'' uij+1' uij+1''
        # We do the exact same here but by setting these entries in the correct diagonal
        maind[0]=1
        ud1[0]=0
        ud2[0]=2
        ud3[0]=-h
        ld1[0]=0
        maind[1]=h
        ud1[1]=-6
        ud2[1]=5*h

        # We repeat this now for the 2 boundary equations at j=n-1
        maind[-1]=-h
        maind[-2]=1
        ud1[-1]=0
        ld1[-1]=0
        ld1[-2]=h
        ld2[-1]=-5*h
        ld2[-2]=2
        ld3[-1]=-6

        # We now set the remaining entries of the matrix as in dualfd1
        # In the else statement of dualfd1 we set the even rows as L1 and the odd rows as L2
        # I wasn't able to completely understand the setup of A so I generated matrices using the code in dualfd1
        # And compared coefficients to see which diagonals have which entries
        # This alows me to have vectorised operations avoiding the for loop

        # Note that we slice accordingly as to not affect the entries we set from the boundary equations
        # Generatied using ChatGPT-4o
        ud3[::2]=-h

        ud2[2::2]=7
        ud2[3::2]=-h

        ud1[2::2]=0
        ud1[3::2]=9

        maind[2:n-2:2]=16
        maind[3:n-2:2]=8*h
        
        ld1[1::2]=h
        
        ld2[0:n-4:2]=7
        ld2[1:n-3:2]=-h

        ld3[:-2:2]=-9
        ld3[1:-2:2]=0

        A = diags([ld3, ld2, ld1, maind, ud1, ud2, ud3],[-3,-2,-1,0,1,2,3])

        # inefficient
        # return A.toarray()

        # Returns a sparse form matrix that contains all the values of our septa-diagonal matrix
        # We will leverage scipy.sparse so we do not have to use .toarray() which is slow
        return A
    
    # Function to create a matrix for b
    # In dualfd1 the code only works in the specific case of m=1
    # In order to extend this to geenral m we will have to use multiple vectors b (specifically m of them)
    # To make this more efficient we initialise a matrix to store the values of b and we can slice it appropriately
    def create_matrix_b(m,n):

        # Copied from dualfd1
        c31,c22,cb11,cb21,cb31,cb12,cb22,cb32 = 15/h,24/h,-3.5/h,4/h,-0.5/h,9/h,-12/h,3/h

        b = np.zeros((m,2*n))

        # Compute boundary conditions for all rows
        # I have essentially copied the code from dualfd1 for each row of the matrix
        # Fill the entries of b using the 2 boundary equations at j=0
        b[:,0] = cb11*f[:,0]+cb21*f[:,1]+cb31*f[:,2]
        b[:,1] = cb12*f[:,0]+cb22*f[:,1]+cb32*f[:,2]

        # Fill the entries of b using the 2 boundary equations at j=n-1
        b[:,-2] = -(cb11*f[:,-1]+cb21*f[:,-2]+cb31*f[:,-3])
        b[:,-1] = -(cb12*f[:,-1]+cb22*f[:,-2]+cb32*f[:,-3])

        # FIll the rest of the values of b
        i = np.arange(1,n-1)
        b[:,2*i] = c31*(f[:,i+1]-f[:,i-1])
        b[:,2*i+1] = c22*(f[:,i-1]-2*f[:,i] + f[:,i+1])

        # here we return a (m x 2n) matrix b where each row corresponds to each vector we need
        return b

    A = generate_A(n)
    # use scipy to convert to matrix diagonal ordered form in O(1) time!
    ab = np.flipud(A.data)
    b = create_matrix_b(m,n)

    # We then apply solve banded to each row to get our numerical solution for the first and second derivatives
    for i in range(m):
        # solve banded takes parameters l and u (both 3), the banded matrix ab, and the RHS vector b
        sol = solve_banded((3, 3), ab, b[i, :])
        df[i, :] = sol[::2]
        d2f[i, :] = sol[1::2]
    
    return df,d2f

# Test function to see if part2q1 works
def test_part2q1():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x, y)

    # define the function sin(pi x)cos(pi x)
    f = np.sin(np.pi*X)*np.cos(np.pi*Y)
    df, d2f = part2q1(f)
    print("df:", df)
    print()
    print("d2f:", d2f)

    # define the true derivatives
    df_analytical = np.pi * np.cos(np.pi*X) * np.cos(np.pi*Y)
    d2f_analytical = np.pi**2 * np.sin(np.pi*X) * np.cos(np.pi*Y)

    print("Analytical df:", df_analytical)
    print("Analytical d2f:", d2f_analytical)

    return None


def part2q2():
    """
    Part 2, question 2
    Add input/output as needed

    """
    # Generated by ChatGPT-4o
    def timing_test():
        # Arrays to store timing results
        fd2_times = []
        part2q1_times = []

        # n_values = range(10,2011,10)

        # second n_values defined for comparing accuracy and time plots
        # otherwise it takes too long to plot
        n_values = range(10, 2011, 200)

        # Iterate over increasing values of n
        for n in n_values:
        # Create meshgrid and function f
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, y)
            f = np.sin(np.pi * X) * np.cos(np.pi * Y)

        # Time fd2
        start_time = time.perf_counter()
        for i in range(10):
            fd2(f)
            end_time = time.perf_counter()
        fd2_times.append((end_time - start_time)/5)

        # Time part2q1
        start_time = time.perf_counter()
        for i in range(10):
            part2q1(f)
            end_time = time.perf_counter()
        part2q1_times.append((end_time - start_time)/5)

        # start_time = time.time()
        # part2q1(f)
        # part2q1_times.append(time.time() - start_time)

        # Convert timing lists to numpy arrays
        fd2_times = np.array(fd2_times)
        part2q1_times = np.array(part2q1_times)

        # Polynomial fitting
        fd2_polyfit = np.polyfit(n_values, fd2_times, 2) # Quadratic fit
        part2q1_polyfit = np.polyfit(n_values, part2q1_times, 2)

        # Generate smooth curves for the polynomial fits
        n_fit = np.linspace(min(n_values), max(n_values), 500)
        fd2_fit = np.polyval(fd2_polyfit, n_fit)
        part2q1_fit = np.polyval(part2q1_polyfit, n_fit)

        plt.figure(figsize=(10, 6))
        plt.plot(n_values, fd2_times, 'o-', label='fd2 Timing', markersize=5)
        plt.plot(n_values, part2q1_times, 's-', label='part2q1 Timing', markersize=5)
        plt.plot(n_fit, fd2_fit, '--', label=f'fd2 Polyfit: {fd2_polyfit[0]:.2e}n^2 + {fd2_polyfit[1]:.2e}n + {fd2_polyfit[2]:.2e}')
        plt.plot(n_fit, part2q1_fit, '--', label=f'part2q1 Polyfit: {part2q1_polyfit[0]:.2e}n^2 + {part2q1_polyfit[1]:.2e}n + {part2q1_polyfit[2]:.2e}')
        
        plt.xlabel('n (Grid Size)')
        plt.ylabel('Time (seconds)')
        plt.title('Timing Comparison of fd2 and part2q1')
        plt.legend()
        plt.grid(True)
        plt.show()

        return fd2_times, part2q1_times

    # Generated by ChatGPT-4o
    def accuracy_test():
        # Define the analytical function and its derivative
        def testf(X, Y):
            return np.sin(np.pi * X) * np.cos(np.pi * Y)

        def df_analytical(X, Y):
            return np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)

        # Initialize arrays to store cumulative differences
        # n_values = range(10, 1011, 100) # Step by 100 for efficiency

        # second n_values defined for comparing accuracy and time plots
        n_values = range(10, 2011, 200)

        fd2_differences = []
        part2q1_differences = []

        for n in n_values:
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, y)

            # Compute the function and analytical derivative
            f = testf(X, Y)
            df_analytical_vals = df_analytical(X, Y)[1:-1,:]

            # Compute derivatives using the two methods
            df_fd2 , _ = fd2(f)
            df_fd2_interior=df_fd2[1:-1,:]
            df_part2q1 , _ = part2q1(f)
            df_part2q1_interior=df_part2q1[1:-1,:]

            # Compute mean differences
            fd2_diff = np.mean(np.abs(df_fd2_interior - df_analytical_vals))
            part2q1_diff = np.mean(np.abs(df_part2q1_interior - df_analytical_vals))

            # Append to results
            fd2_differences.append(fd2_diff)
            part2q1_differences.append(part2q1_diff)
        
        log_n_values = np.log(n_values)
        log_fd2_differences = np.log(fd2_differences)
        log_part2q1_differences = np.log(part2q1_differences)

        # Perform polynomial fitting (linear fit in log-log space)
        fd2_fit = np.polyfit(log_n_values, log_fd2_differences, 1)
        part2q1_fit = np.polyfit(log_n_values, log_part2q1_differences, 1)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(n_values, fd2_differences, label='fd2 Differences', marker='o', markersize=4, linestyle='-', alpha=0.7)
        plt.loglog(n_values, part2q1_differences, label='part2q1 Differences', marker='s', markersize=4, linestyle='--', alpha=0.7)

        plt.loglog(n_values, np.exp(fd2_fit[1] + fd2_fit[0] * log_n_values), 
        label=f'fd2 Fit (slope: {fd2_fit[0]:.4f})', 
        color='red', linestyle='-.', alpha=0.7)
        plt.loglog(n_values, np.exp(part2q1_fit[1] + part2q1_fit[0] * log_n_values), 
        label=f'part2q1 Fit (slope: {part2q1_fit[0]:.4f})', 
        color='green', linestyle='-.', alpha=0.7)
        plt.xlabel('Grid Size (n)')
        plt.ylabel('Average Difference')
        plt.title('Error Comparison of Numerical Derivatives')
        plt.legend()
        plt.grid(True)
        plt.show()

        return fd2_differences, part2q1_differences
 
    # code to plot the errors and times against each other
    # requires 2nd definition of n_values in timing_test() and accuracy_test()
    fd2_times, part2q1_times = timing_test()
    fd2_differences, part2q1_differences = accuracy_test()
    # length = len(fd2_differences)
    # fd2_times = fd2_times[:length]
    # part2q1_times = part2q1_times[:length]

    plt.figure(figsize=(10, 6))
    plt.loglog(fd2_differences,fd2_times,label='fd2',color='black')
    plt.loglog(part2q1_differences,part2q1_times,label='part2q1',color='blue')
    plt.xlabel('Average Difference')
    plt.ylabel('Time (seconds)')
    plt.title('Graph used to compare cost and accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return None #modify as needed


#---------------------------- End code for Part 2 ----------------------------#

if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call functions used to generate the figures included in your report.
    # n = 19
    # g = 1.2
    # mu = 2.5
    # T = 50

    # n = 5
    # g = 1.0
    # mu = 0.5
    # T = 100

    # xbar, xtilde0, eratio, _ = part1q1a(n, g, mu, T)
    # print("xbar: ", xbar)
    # print("xtilde0: ", xtilde0)
    # print("eratio: ", eratio)
    # relative_difference = part1q1b()s
    # print("energy ratio relative_difference: ", relative_difference)

    # T_values, energy_ratios = part1q1c()
    # #print(T_values, energy_ratios)

    # part1q2()

    # t1 = time.perf_counter()
    # test_part2q1()
    # t2 = time.perf_counter()
    # print(t2-t1)

    # part2q2()