"""
    Fuel consumption analysis for Truck platooning
    
    This script runs an scenario of 4 trucks in platoon mood where
    particular splits are determined. 
    
    In order to use: python platoon-closed.py 
    
    Check output files in: ../output/
"""

import numpy as np

# Platoon length
N = 4

# Truck parameters
L_AVG = 18
G = 9.81
MU = 0.02
M = 44000
RHO = 1.2
A = 10
CD = 0.7

# Dynamics
K1 = G * MU
K2 = G
K3 = RHO * A * CD / (2 * M)

# Control
C1 = 0.1
C2 = 1
C3 = 0.5
U_MAX = 1.5  # Max. Acceleration
U_MIN = -1.5  # Min. Acceleration

# Time
DT = 0.1
H = 50  # samples horizon
SIMTIME = 60  # seconds
nSamples = int(SIMTIME * 1 / DT)
aDims = (nSamples, N)
aDimMPC = (H, N)

# Traffic
V_F = 25.0  # Max speed.
V_P = 20.0  # Platoon free flow
E = 25.0*0.3  # Speed drop for relaxation
C = 2400 / 3600.0
G_X = 5


def compute_parameters(g_x, c):
    """ Compute dynamically parameters based on G_X,C"""
    s_x = L_AVG + g_x
    k_x = 1 / s_x
    k_c = c / V_P
    w = c / (k_x - k_c)
    return (s_x, k_x, k_c, w)


S_X, K_X, K_C, W = compute_parameters(G_X, C)
TAU = 1/(K_X*W)  # Time shift
RTE = TAU * (V_P+W) / V_P  # Time headway

# At capacity
K = C/V_P
S_D = 1/K-L_AVG
G_T = S_D/V_P


def set_initial_condition(mS0, mV0, mDV0):
    """ Setup initial conditions of experiment"""
    mS = np.zeros(aDims)  # Spacing all trucks
    mV = np.zeros(aDims)  # Speed
    mDV = np.zeros(aDims)  # Speed diference
    mS[0, :] = mS0
    mV[0, :] = mV0
    mDV[0, :] = mDV0
    return (mS, mV, mDV)


def create_ref(dEvent, Teq):
    """Creates a reference matrix for the control"""

    def anticipation_time(T_0, T_F):
        """Computes the anticipation time according to TRB 2018"""
        T_a = E / 2 * (U_MIN-U_MAX) / (U_MIN * U_MAX) + \
            (V_P + W) / E * (T_F - T_0)
        return T_a

    def get_sigmoid(v0, vf, yld, ant):
        """ Computes a sigmoid function with rise time equivalent to anticipation time"""
        aNewTime = 8 * (aTime - (yld + ant/2)) / ant
        return v0 + (vf-v0) * 1 / (1 + np.exp(- aNewTime))

    mRef = np.ones(aDims) * Teq
    iIdTruck = dEvent['id']
    fMrgTime = dEvent['tm']
    T_0, T_X = dEvent['tg']
    _T_0, _T_X = (T_X, T_0) if T_0 > T_X else (T_0, T_X)
    fAntTime = anticipation_time(_T_0, _T_X)
    fYldTime = fMrgTime - fAntTime

    mRef[:, iIdTruck] = get_sigmoid(T_0, T_X, fYldTime, fAntTime)

    print(f'Anticipation time: {fAntTime}')
    print(f'Yielding time: {fYldTime}')

    return mRef


def initialize_mpc(mS0, mV0, mDV0):
    """ Initialize internal variables control"""
    m_S, m_V, m_DV, m_LS, m_LV, m_U = (np.zeros(aDimMPC) for i in range(6))
    m_S[0] = mS0
    m_V[0] = mV0
    m_DV[0] = mDV0
    return m_S, m_V, m_DV, m_LS, m_LV, m_U


def _cds(s):
    """ Non linear drag coefficient"""
    s[0] = L_AVG  # Accounts for leader not saving
    fCD = (1-np.exp(-2 * s / L_AVG))/2 + 0.42
    return fCD


def _cdv(v):
    """ Non linear speed coefficient"""
    return v**2


def g_cds(s):
    """ Gradient non linear drag coefficient"""
    s[0] = L_AVG  # Accounts for leader not saving
    fCD = np.exp(-2 * s / L_AVG) / (2 * L_AVG)
    return fCD


def g_cdv(v):
    """ Gradient non linear speed"""
    return 2 * v


def linear_drag(s, v, s0, v0):
    """ Computes linear term for drag"""
    KS0V0 = _cds(s0) * _cdv(v0)
    KS = g_cds(s0) * _cdv(v0)
    KV = _cds(s0) * g_cdv(v0)
    lin = KS0V0 + KS * (s-s0) + KV * (v-v0)
    return lin


def forward_evolution(X, U, D):
    """ Compute forward model evolution
        X: S, V, DV
        U: control
        D: slope
    """

    S, V, DV = X

    def cordim(x): return x.shape if len(x.shape) > 1 else (1, x.shape[0])

    U = U.reshape(cordim(U))

    DU = np.concatenate((np.zeros((U.shape[0], 1)),
                         U[:, 0:-1] - U[:, 1:]),
                        axis=1)

    run = zip(U, DU, D)

    for i, u in enumerate(run):
        u_s, du, theta = u
        if i < len(S)-1:
            DV[i+1] = DV[i] + DT * du
            S[i+1] = S[i] + DT * DV[i]
            mfac = u_s - K1 - K2 * theta\
                - K3 * linear_drag(S[i], V[i],
                                   S[0], V[0])
            V[i+1] = V[i] + DT * mfac
    return S, V, DV


def forward_evolution_alt(X, U, D):
    """ Compute forward model evolution
        X: S, V, DV
        U: control
        D: slope
    """

    S, V, DV = X

    def cordim(x): return x.shape if len(x.shape) > 1 else (1, x.shape[0])

    U = U.reshape(cordim(U))

    DU = np.concatenate((np.zeros((U.shape[0], 1)),
                         U[:, 0:-1] - U[:, 1:]),
                        axis=1)

    run = zip(U, DU, D)

    for i, u in enumerate(run):
        u_s, du, _ = u
        if i < len(S)-1:
            DV[i+1] = DV[i] + DT * du
            S[i+1] = S[i] + DT * DV[i]
            mfac = u_s
            V[i+1] = V[i] + DT * mfac
    return S, V, DV


def backward_evolution(X, Ref):
    """ Compute  bakckward costate evolution
        L: LS, LV
        X: S, V, DV
    """

    def reversedEnumerate(*args):
        """ Inverse enumeration iterator"""
        revArg = [np.flip(x, axis=0) for x in args]
        return zip(range(len(args[0])-1, -1, -1), *revArg)

    S, V, DV = X

    ls = np.zeros(aDimMPC)
    lv = np.zeros(aDimMPC)

    runinv = reversedEnumerate(S, V, DV, Ref)

    K3 = RHO * A * CD / (2 * M)

    for i, s, v, dv, tg in runinv:
        if i > 0:
            sref = v * tg + L_AVG

            lv[i-1] = lv[i] + DT * (-2 * C1 * (s-sref) * tg
                                    - C2 * dv - ls[i]
                                    - lv[i] * K3 * _cds(S[0]) * g_cdv(V[0])
                                    )

            ls[i-1] = ls[i] + DT * (2 * C1 * (s-sref)
                                    - lv[i] * K3 * g_cds(S[0]) * _cdv(v[0])
                                    )

    return ls, lv


def backward_evolution_alt(X, Ref):
    """ Compute  bakckward costate evolution
        L: LS, LV
        X: S, V, DV
    """

    def reversedEnumerate(*args):
        """ Inverse enumeration iterator"""
        revArg = [np.flip(x, axis=0) for x in args]
        return zip(range(len(args[0])-1, -1, -1), *revArg)

    S, V, DV = X

    ls = np.zeros(aDimMPC)
    lv = np.zeros(aDimMPC)

    runinv = reversedEnumerate(S, V, DV, Ref)

    _ = RHO * A * CD / (2 * M)

    for i, s, v, dv, tg in runinv:
        if i > 0:
            sref = v * tg + L_AVG
            lv[i-1] = lv[i] + DT * (-2 * C1 * (s-sref) * tg
                                    - C2 * dv - ls[i]
                                    )
            ls[i-1] = ls[i] + DT * (2 * C1 * (s-sref)
                                    )

    return ls, lv


def compute_control(mX0, mRef, mTheta):
    """ Computes a control based on mX0 and the reference mRef"""

    _m_S, _m_V, _m_DV, _m_LS, _m_LV, _ = initialize_mpc(*mX0)
    _X = (_m_S, _m_V, _m_DV)

    # Parameters

    ALPHA = 0.02
    EPS = 0.1

    # Convergence
    error = 100
    bSuccess = 2
    N = 10000  # number of iterations

    step = iter(range(N))
    n = 0
    n_prev = 0

    while (error > EPS) and (bSuccess > 0):
        try:
            next(step)

            U_star = -_m_LV / (2 * C3)

            U_star = np.clip(U_star, U_MIN, U_MAX)

            _m_S, _m_V, _m_DV = forward_evolution_alt(_X, U_star, mTheta)

            _lS, _lV = backward_evolution_alt(_X, mRef)

            _m_LS = (1 - ALPHA) * _m_LS + ALPHA * _lS
            _m_LV = (1 - ALPHA) * _m_LV + ALPHA * _lV

            error = np.linalg.norm(_m_LS - _lS) + \
                np.linalg.norm(_m_LV - _lV)

            # print(f'Error:{error}')
            # Routine for changing convergence parameter

            if error > 10e5:
                raise AssertionError('Algorithm does not converge ')
            if n >= 5000:
                ALPHA = max(ALPHA - 0.01, 0.01)
                print(f'Reaching {n} iterations: Reducing alpha: {ALPHA}')
                print(f'Error before update {error}')
                if n > 20000:
                    raise AssertionError(
                        'Maximum iterations reached by the algorithm')
                n_prev = n + n_prev
                n = 0
            if error <= EPS:
                bSuccess = 0

            n += 1

        except StopIteration:
            print('Stop by iteration')
            print('Last simulation step at iteration: {}'.format(n+n_prev))
            bSuccess = 0

    n = n + n_prev
    print(f'Total iterations:{n}')

    return U_star[0]


def closed_loop(dEvent):
    """Receives a dictionary and finds the solution in closed loop"""

    # Time
    aTime = np.arange(nSamples)*DT

    mS0 = np.ones(N) * (S_D + L_AVG)
    mV0 = np.ones(N) * V_P
    mDV0 = np.zeros(N)
    mX0 = np.array([i * (S_D + L_AVG) for i in reversed(range(4))])

    mS, mV, mDV = set_initial_condition(mS0, mV0, mDV0)
    mX = np.empty_like(mS)
    mX[0] = mX0

    mRef = create_ref(dEvent, G_T)
    mTheta = np.zeros(mRef.shape)
    mU = np.zeros(mRef.shape)

    mRefW = G_T*np.ones((H, N))
    mThetaW = np.zeros((H, N))

    for i, t in enumerate(zip(mRef, aTime)):

        if i < len(mRef)-2:

            mRefW = mRef[i:min(i+H, nSamples), :]
            mThetaW = np.empty_like(mRefW)

            print(f'Sample Time:{t[-1]}')

            aX = (mS[i], mV[i], mDV[i])

            aU = compute_control(aX, mRefW, mThetaW)

            aDU = aU[0:-1] - aU[1:]

            aDU = np.insert(aDU, 0, 0)

            mDV[i+1] = mDV[i] + DT * aDU
            mS[i+1] = mS[i] + DT * mDV[i]
            mfac = aU - K1 - K2 * mTheta[i]\
                - K3 * linear_drag(mS[i], mV[i], mS[i], mV[i])
            # mfac = aU - K1 - K2 * mTheta[i]
            mV[i+1] = mV[i] + DT * mfac

            mU[i] = aU

            mX[i+1] = mX[i] + mV[i] * DT + 0.5 * aU * DT ** 2

    mSd = mRef * V_P + L_AVG

    return mS, mV, mDV, mSd, mU, mX


if __name__ == "__main__":

    # Time
    aTime = np.arange(nSamples)*DT

    iYieldTruck = range(1, N)

    # Splits are predefined at some specific points in time (merging times)
    iIdxTimeSplit = [int(t*60*1/DT) for t in (0.5,)]
    fTimeSplit = [aTime[i] for i in iIdxTimeSplit]
    fValueTimeHwyInitial = [G_T, G_T, G_T]
    fValueTimeHwyEnd = [2*G_T, 3*G_T, 4*G_T]
    fValueTimeHwy = [(st, ed) for st, ed in zip(
        fValueTimeHwyInitial, fValueTimeHwyEnd)]

    mEvents = [{'id': i, 'tm': tm, 'tg': tg} for tm in fTimeSplit
               for i in iYieldTruck for tg in fValueTimeHwy]

    print(f'Simulating the following situations: {mEvents}')

    dirname = '../output/'

    for event in mEvents:

        print(f'Current situation:{event}')

        S, V, DV, Sd, U, X = closed_loop(event)

        sEvent = '_yield_' + str(event['id']) + '_gap_' + str(event['tg'][-1])

        filename_S = dirname + 'space' + sEvent + '.csv'
        filename_V = dirname + 'speed' + sEvent + '.csv'
        filename_Sd = dirname + 'reference' + sEvent + '.csv'
        filename_U = dirname + 'control' + sEvent + '.csv'
        filename_X = dirname + 'postition' + sEvent + '.csv'

        np.savetxt(filename_S, S, fmt='%.4f',
                   delimiter='\t', newline='\n')
        np.savetxt(filename_V, V, fmt='%.4f',
                   delimiter='\t', newline='\n')
        np.savetxt(filename_Sd, Sd,
                   fmt='%.4f', delimiter='\t', newline='\n')
        np.savetxt(filename_U, U, fmt='%.4f',
                   delimiter='\t', newline='\n')
        np.savetxt(filename_X, X, fmt='%.4f',
                   delimiter='\t', newline='\n')
