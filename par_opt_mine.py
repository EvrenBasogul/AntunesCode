
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fmin
from solvol import solvol
# Function for solving the volume
from convertVDM import convertVMD
from alphaest import alphaest
from aggregationdF import aggregationdF
# Aggregation ODE function
from mysql import mylsq
from scipy.optimize import minimize



# Initial values
B = 25.8
amax = 0.315
gama = 0.28
teta_initial = [amax, B, gama]

# Experimental data (sampling instants t and measured d43)
texp = np.array([0, 0.5, 1.4, 2.3, 3.3, 4.2, 5.1, 6, 6.9, 7.8, 8.8, 9.7, 10.6, 11.5, 12.4, 13.3, 14.3])
dexp = np.array([0.38, 14.61, 31.132, 76.621, 91.602, 84.82, 80.225, 77.021, 74.752, 72.567, 70.869, 69.561, 68.673, 67.33, 66.225, 65.062, 64.155])
data = np.column_stack((texp, dexp))

# Physical constants
G = 312  # Shear rate (s^-1)
imax = 30  # Maximum size interval
dFo = 1.65  # Initial fractal dimension
dFmax = 2.37  # Maximum fractal dimension
do = 100  # Particle size in nm
x = 0.1  # Parameter for alpha estimation
y = 0.1  # Parameter for alpha estimation

# Initial number distribution, #/cm^3
a = np.array([4.60e8, 6.31e9, 1.24e9, 2.89e6, 4.24e7, 8.77e6, 5.79e5, 1.34e5, 2.32e4, 1.89e3])
sizea = len(a)
sizeb = imax - sizea
b = np.zeros(sizeb)
no = np.concatenate([a, b])  # Initial number distribution

t = 0
tmax = 15  # Maximum time
Totalvol1 = solvol(no, do)  # Total volume of solid
dcurrent = convertVMD(no, dFo, do)

# Time and number arrays to store results
time = [t]
number = [no]
VMD = [dcurrent * 1e-3]  # Volume mean diameter in microns
dF = [dFo]

# Normalization of no
nref = np.ones(imax)
for i in range(imax):
    if no[i] != 0:
        nref[i] = no[i]

nonorm = no / nref  # Normalized distribution
dFref = dFo
dFnorm = dFo / dFref  # Normalized fractal dimension

# Run the optimization with fminsearch (in Python use minimize from scipy)
result = minimize(mylsq, teta_initial, args=(nonorm, dFnorm, G, dcurrent, dFmax, do, nref, dFref, imax, x, y, data), method='Nelder-Mead', options={'xatol': 2e-16, 'maxfun': 10000, 'maxiter': 1000})

# Optimized parameters
teta_opt = result.x
fval = result.fun

# Output results
print('Optimized Parameters (theta) from optimization:')
print(teta_opt)

# Plot the evolution of theta (optional)
plt.plot(result.allvecs[:, 0], label='Theta1 (amax)')
plt.plot(result.allvecs[:, 1], label='Theta2 (B)')
plt.plot(result.allvecs[:, 2], label='Theta3 (gama)')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Theta Values')
plt.title('Theta Evolution during Optimization')
plt.show()