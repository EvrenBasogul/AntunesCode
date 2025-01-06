import numpy as np
from scipy.integrate import solve_ivp
from solvol import solvol
from alphaest import alphaest
from aggregationdF import aggregationdF
from convertVDM import convertVMD


def mylsq(teta, nonorm, dFnorm, G, dcurrent, dFmax, do, nref, dFref, imax, x, y, data):
    t = 0
    tmax = 14
    y_obs = data[:, 1]  # Observed data
    y_cal = [dcurrent * 1e-3]  # VMD in microns
    amax=teta[0]
    B=teta[1]
    gama=teta[2]
    a = np.array([4.60e8, 6.31e9, 1.24e9, 2.89e6, 4.24e7, 8.77e6, 5.79e5, 1.34e5, 2.32e4, 1.89e3])
    sizea = len(a)
    sizeb = imax - sizea
    b = np.zeros(sizeb)
    no = np.concatenate((a, b))  # size(no) = (imax,)    
    # Total volume of solids as reference
    Totalvol1 = solvol(no, do)    
    # Estimate Alpha
    Alpha = alphaest(x, y, imax, amax)  # Alpha estimation function    
    i = 0
    time = [t]
    number = np.zeros((1, imax))
    dF = []
    ycal = []
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
        sol = solve_ivp(aggregationdF, tspan, Yo, method='RK45', args=(Alpha, B, G, dcurrent, dFmax, do, nref, dFref, gama, imax),  atol=1e-4,rtol=1e-6)
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
        dFnorm=Yf[Yfrow-1,Yfcol-1]
        t = tf[tfrow-1]
        i=i+1
        dFo = dFnorm*dFref
        dcurrent = convertVMD(no, dFo, do)
        time.append(t)
        if i>0:
            number = np.vstack((number, no))  # Append zero row
        dF.append(dFo)
        ycal.append(dcurrent*1e-3)
        dFref=dFo
        dFnorm=dFo/dFref
        for k in range(imax):
            if number[i,k]==0:
                nref[k]=1 
            else:
                nref[k]=number[i,k]
        nonorm=no/nref
    # Compute the expression to be minimized
    lsq = np.sum((np.array(y_obs) - np.array(y_cal)) ** 2)
    return lsq
