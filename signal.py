
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal


# -----------------------------------------------------------------------------
# Signal Analysis
# -----------------------------------------------------------------------------


def find_peaks(y, min_height=-np.inf, min_distance=1, threshold=0, sort_order='N', num_peaks=None):
    # data check
    if y.size < 3:
        print('The data set must contain at least 3 samples.')
        return None

    # min_height
    indx = np.flatnonzero(y[1:(y.size - 1)] > min_height) + 1
    if indx.size == 0:
        print('There are no data point greater than min_height.')
        return None

    # min_distance, threshold
    ipk = np.zeros_like(indx)
    npk = 0
    for ii in indx:
        il = max([0, ii - min_distance])
        ir = min([ii + min_distance, y.size - 1])
        if all(y[ii] - threshold >= y[il:ii]) and all(y[ii] - threshold >= y[ir:ii:-1]):
            ipk[npk] = ii
            npk = npk + 1

    if npk == 0:
        print('No peak found.')
        return None

    # sort_order
    so = sort_order[0].upper()
    if so == 'N':
        ipk = ipk[0:npk]
    else:
        pk = y[ipk[0:npk]]
        if so == 'A':
            ix = np.argsort(pk)
        elif so == 'D':
            ix = np.argsort(-pk)
        ipk = ipk[ix]

    # num_peaks
    if num_peaks:
        if npk > num_peaks:
            ipk = ipk[0:num_peaks]

    return ipk


# -----------------------------------------------------------------------------


def frequency_domain_decomposition(dm, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):
    # test run to know faxis.size
    faxis, p = signal.csd(dm[:, 0], dm[:, 0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis)

    ns = dm.shape[1]  # number of sensors
    nf = faxis.size
    p = np.zeros((nf, ns, ns))
    for i in range(ns):
        for j in range(ns):
            faxis, pij = signal.csd(dm[:, i], dm[:, j], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis)
            p[:, i, j] = abs(pij)

    spectrums = np.zeros((nf, ns))
    mode_shapes = np.zeros((nf, ns, ns))
    for k in range(nf):
        mode_shapes[k, :, :], spectrums[k, :], _ = np.linalg.svd(p[k, :, :])

    return {'x': faxis, 'y': spectrums, 'MS': mode_shapes}


def modes_from_spectrum(ss):

    ss0 = ss['y'][:, 0]
    ipk = find_peaks(ss0, min_height=np.percentile(ss0, 90))
    fn = ss['x'][ipk]
    ms = ss['MS'][ipk, :, 0].T

    return {'fn': fn, 'DR': np.zeros_like(fn), 'MS': ms}


# -----------------------------------------------------------------------------


def _stochastic_identification_args(y, nsr):

    if y.ndim is 1:
        y.shape = (y.size, 1)
    nd, ns = y.shape

    if nsr is None:
        nsr = ns

    return y, nd, ns, nsr


def blkhankel(y, nbr=10, nsr=None):
    # y: measured data, [nd x ns]
    # nbr: number of block rows, = i
    # nsr: number of reference sensors

    # nd: number of data, ns: number of sensors
    y, nd, ns, nsr = _stochastic_identification_args(y, nsr)

    ndc = nd - 2 * nbr + 1  # ndc: number of columns of block hankel matrix
    yt = y.T / np.sqrt(ndc)

    Yp = np.zeros((nbr * nsr, ndc))
    Yf = np.zeros((nbr * ns, ndc))
    for k in range(nbr):
        Yp[(k * nsr):((k + 1) * nsr), :] = yt[:nsr, k:(k + ndc)]
        Yf[(k * ns):((k + 1) * ns), :] = yt[:, (nbr + k):(nbr + k + ndc)]

    return Yp, Yf


def subspace_identification(y, nx, nbr=10, nsr=None):
    # Problem)
    # x_k+1 = A x_k + w_k
    #   y_k = C x_k + v_k
    #
    # Q = E[w_k w_k.T]
    # R = E[v_k v_k.T]
    # S = E[w_k v_k.T]
    # Exx = E[x_k x_k.T]
    # Eyy = E[y_k y_k.T], cf) Lambda_i = E[y_k+i y_k.T], Lambda_0 = E[y_k y_k.T] = Eyy
    # G = E[x_k+1 y_k.T]
    #
    # y: measured data, [nd x ns]
    # fs: sampling frequency
    # nx: number of system dofs, <= ns x nbr
    # nbr: number of block rows, = i
    # nsr: number of reference sensors
    #
    # system['fn']: natural frequencies
    # system['DR']: damping ratios
    # system['MS']: mode shapes

    # system = {'Matrix': {}}

    # nd: number of data, ns: number of sensors
    y, nd, ns, nsr = _stochastic_identification_args(y, nsr)

    if nx > ns * nbr:
        print('Warning: nx cannot be greater than ns * nbr. Set nx = ns * nbr.')
        nx = ns * nbr

    Yp0, Yf0 = blkhankel(y[:nd, :], nbr, nsr)

    H = np.vstack((Yp0, Yf0))
    F, Ut = np.linalg.qr(H.T)
    QT = F.T
    Lt = Ut.T

    nlq = np.cumsum([nsr * nbr, nsr, ns - nsr, ns * (nbr - 1)])

    Pri = Lt[nlq[0]:, :nlq[0]] @ QT[:nlq[0], :]
    U1, s1, _ = np.linalg.svd(Pri, full_matrices=False)
    U1, s1 = U1[:, :nx], s1[:nx]

    Oi = U1 @ np.diag(np.sqrt(s1))
    Xi = np.linalg.pinv(Oi) @ Pri

    Prip = Lt[nlq[2]:, :nlq[1]] @ QT[:nlq[1], :]
    Oip = Oi[:(Oi.shape[0] - ns), :]
    Xin = np.linalg.pinv(Oip) @ Prip

    Yii = Lt[nlq[0]:nlq[2], :nlq[2]] @ QT[:nlq[2], :]
    AC = np.vstack((Xin, Yii)) @ np.linalg.pinv(Xi)
    A = AC[:nx, :]
    C = AC[nx:, :]

    rhowv = np.vstack((Xin, Yii)) - AC @ Xi
    QRS = rhowv @ rhowv.T / rhowv.shape[1]
    Q = QRS[:nx, :nx]
    R = QRS[nx:, nx:]
    S = QRS[:nx, nx:]

    Exx = linalg.solve_discrete_lyapunov(A, Q)
    Eyy = C @ Exx @ C.T + R
    G = A @ Exx @ C.T + S

    # system['Matrix'] = {'A': A, 'C': C, 'Exx': Exx, 'Eyy': Eyy, 'G': G, 'Q': Q, 'R': R, 'S': S}

    # system['Mode'] = modes_from_system(system, fs)

    # system['Spectrum'] = spectrums_from_system(system, fs, df)

    return {'A': A, 'C': C, 'Exx': Exx, 'Eyy': Eyy, 'G': G, 'Q': Q, 'R': R, 'S': S}


# -------------------------------------


def modes_from_system(sm, fs):
    # x_k+1 = A x_k + w_k
    #   y_k = C x_k + v_k
    #
    # A: system matrix from function subspace_identification()
    # C: system matrix from function subspace_identification()
    # fs: sampling frequency
    #
    # fn: natural frequencies (numpy 1-dim array)
    # drn: damping ratios corresponding to fn (numpy 1-dim array)
    # Vn: mode shapes (numpy 2-dim array) corresponding to fn, column vector corresponding to 1 mode.

    # sm = system['Matrix']

    W, U = np.linalg.eig(sm['A'])
    L = np.log(W) * fs
    fn = np.abs(L) / (2 * np.pi)
    drn = -np.real(L) / np.abs(L)
    Vn = np.dot(sm['C'], U.real)

    return {'fn': fn, 'DR': drn, 'MS': Vn}


def draw_mode_shapes(sf, loc, cols=None, title=None):

    # sf = system['Mode']

    if cols is None:
        cols = range(sf['fn'].size)

    plt.figure()
    for j in cols:
        labelstr = r'$f_n$ = %.1f Hz, $\xi_n$ = %.1f%%' % (sf['fn'][j], sf['DR'][j] * 100)
        plt.plot(loc, sf['MS'][:, j], label=labelstr)
    plt.grid()
    plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)
    plt.xlabel('Location')
    plt.ylabel('Mode shape')
    plt.show()


def stabilization_diagram(systems, fs):
    # systems: list of systems, each system obtained from function subspace_identification()
    # fs: sampling frequency

    plt.figure()
    for system in systems:
        sf = system['Mode']
        nx = sf['fn'].size
        jc = sf['fn'] < fs / 2
        plt.plot(sf['fn'][jc], np.ones(sf['fn'][jc].shape) * nx, 'b.')
    plt.minorticks_on()
    plt.grid(b=True, which='both', axis='both')
    plt.title('Stabilization diagram')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('System order')
    plt.xlim([0, fs / 2])
    plt.show()


# -------------------------------------


def spectrums_from_system(sm, fs, df):

    # sm = system['Matrix']

    faxis = np.arange(0, fs / 2, df)
    z = np.exp(2 * np.pi * faxis / fs * 1j)
    nf = faxis.size
    ns = sm['C'].shape[0]

    spectrums = np.zeros((nf, ns))
    mode_shapes = np.zeros((ns, nf, ns))
    In = np.diag(np.ones(sm['A'].shape[0]))

    for j in range(nf):
        Syyi = np.real(sm['C'] @ np.linalg.inv(z[j] * In - sm['A']) @ sm['G'] + sm['Eyy'] + sm['G'].T @ np.linalg.inv(In / z[j] - sm['A'].T) @ sm['C'].T)
        mode_shapes[:, j, :], spectrums[j, :], _ = np.linalg.svd(Syyi)

    return {'x': faxis, 'y': spectrums, 'MS': mode_shapes}


def draw_spectrums(ss):

    # ss = system['Spectrum']
    ipk = []

    _, nc = ss['y'].shape

    plt.figure()
    for i in range(nc):
        plt.plot(ss['x'], ss['y'][:, i], label=str(i + 1) + '-th spectrum')
        ipk.append(find_peaks(ss['y'][:, i], min_height=np.percentile(ss['y'][:, i], 80), sort_order='N'))
        if ipk[i] is not None:
            plt.plot(ss['x'][ipk[i]], ss['y'][ipk[i], i], 'o', label=str(i + 1) + '-th peaks')
    plt.grid()
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title('Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectrum')
    plt.show()

    return ipk
