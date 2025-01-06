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



amax=0.315
B=25.8 
gama=0.28

texp = np.array([0, 0.5, 1.4, 2.3, 3.3, 4.2, 5.1, 6, 6.9, 7.8, 8.8, 9.7, 10.6, 11.5, 12.4, 13.3, 14.3])
dexp = np.array([0.38, 14.61, 31.132, 76.621, 91.602, 84.82, 80.225, 77.021, 74.752, 72.567,
                 70.869, 69.561, 68.673, 67.33, 66.225, 65.062, 64.155])
data = np.column_stack((texp, dexp))

# Parameters
G = 312  # s^-1
imax = 30
dFo = 1.65
dFmax = 2.37
do = 100  # nm
x, y = 0.1, 0.1

# Original number distribution, #/cm^3
a = np.array([4.60e8, 6.31e9, 1.24e9, 2.89e6, 4.24e7, 8.77e6, 5.79e5, 1.34e5, 2.32e4, 1.89e3])
b = np.zeros(imax - len(a))
no = np.concatenate([a, b])  # size(no) = 1 x imax
t = 0
tmax = 15  # min

Totalvol1 = solvol(no, do)
dcurrent = convertVMD(no, dFo, do)
time = [t]
number = np.zeros((1,30))
VMD = [dcurrent * 1e-3]  # VMD in microns
dF = [dFo]

# Normalization of no
nref = np.zeros(imax)
for i  in range(imax):
    if no[i]==0:
        nref[i]=1
    else:
        nref[i]=no[i]
nonorm = no / nref
dFref = dFo
dFnorm = dFo / dFref

#................Optimizer.....................
teta_initial=np.array([0.2947,24.9652,0.2527])
args = (nonorm, dFnorm, G, dcurrent, dFmax, do, nref, dFref, imax, x, y, data)

# Optimization using Nelder-Mead (equivalent to fminsearch)
result = minimize(mylsq, teta_initial, args=args, method='Nelder-Mead')

# Optimal parameters
teta_opt = result.x

#...............................................................................
# Estimate Alpha
Alpha = alphaest(x, y, imax, amax)
i=0
# Time stepping
while t < tmax:
    if i == 0:  # Equivalent to MATLAB's `if i==1`
      ti = t + 0.5
      tspan = np.linspace(t, ti, num=2)  # Generate two points: t and ti
    else:
      ti = t + 0.92
      tspan = np.linspace(t, ti, num=2)
    
    
    
    # Ensure Yo has the correct shape
    Yo = np.concatenate((nonorm, np.array([dFnorm]))).flatten()
    
    # Define t_eval as the same points MATLAB might use (e.g., 34 equally spaced points)
    

# Call solve_ivp with t_eval
    sol = solve_ivp(
    aggregationdF,                   # ODE function
    tspan,                        # Start and end time
    Yo,                              # Initial conditions
    method='BDF',                    # Implicit solver (like ode15s)
    args=(Alpha, B, G, dcurrent, dFmax, do, nref, dFref, gama, imax),  # Arguments
    atol=1e-4,                       # Absolute tolerance
    rtol=1e-6,                       # Relative tolerance
    dense_output=True                # Enable dense output for interpolation
    )
    tf=sol.t.T
    Yf=sol.y.T
    Yfrow = Yf.shape[0]
    Yfcol = Yf.shape[1]
    tfrow = len(tf)
    j=0
    while j<=(Yfcol-2):
        if Yf[Yfrow-1,j]<0:
            Yf[Yfrow-1,j]=0
        else:
            Yf[Yfrow-1,j]=Yf[Yfrow-1,j]
        j=j+1 
        
    nonorm = Yf[Yfrow-1,0:(Yfcol-1)]
    no=nonorm * nref
    Totalvol2 = solvol(no,do)
    dTotalvol = ((abs(Totalvol2-Totalvol1))/(Totalvol1))*100
    if dTotalvol>1:
        raise ValueError("Calculation terminated due to volume discrepancy.")
    else:
        nonorm = Yf[Yfrow-1,0:(Yfcol-1)]
        dFnorm = Yf[Yfrow-1,Yfcol-1]
        no=nonorm*nref
        t = tf[tfrow-1]
        i=i+1
        dFo = dFnorm*dFref
        dcurrent = convertVMD(no, dFo, do)
        time.append(t)
        if i>0:
            zero_row = np.zeros((1, imax))
# Append the zero row to the matrix
            number = np.vstack((number, zero_row))
            number[i,:]=no
        else:
            number[i,:]=no
        dF.append(dFo)
        VMD.append(dcurrent*1e-03)
        dFnorm=dFo/dFref
        for k in range(imax):
            if number[i,k]==0:
                nref[k]=1 
            else:
                nref[k]=number[i,k]
        nonorm=no/nref

# Plot results
plt.figure()
plt.plot(texp, dexp, 'ro', label='Experimental data')
plt.plot(time, np.array(VMD) * 1e-5, 'b-', label='Model prediction')
plt.xlabel('Time (min)')
plt.ylabel('d43 (µm)')
plt.legend()
plt.show()

# Initial guesses for optimization
#teta_initial = [0.315, 25.8, 0.28]

# Call the optimizer (fminsearch in MATLAB is replaced by fmin in Python)
#result = fmin(lambda teta: mylsq(teta, nonorm, dFnorm, G, convertVMD(no, dFo, do), dFmax, do, nref, dFref, imax, x, y, data), teta_initial)

# Optimized parameters
#teta_opt = result
#amax, B, gama = teta_opt
#print(f"Optimized parameters: amax = {amax}, B = {B}, gama = {gama}")

# Now we can use the optimized parameters and re-run the simulation with the best values.
# This process is already handled in the function above during optimization.

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(texp, dexp, 'ro', label="Observed data")
plt.plot(time, np.array(VMD) * 1e-5, 'b-', label="Calculated VMD (µm)")
plt.xlabel('Time (min)')
plt.ylabel('d43 (µm)')
plt.legend()
plt.show()
