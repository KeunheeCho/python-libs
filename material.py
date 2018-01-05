import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as optimize


# -----------------------------------------------------------------------------
# Material models
# -----------------------------------------------------------------------------


class MaterialBase:

    def compute(self, strain):
        if np.isscalar(strain):
            stress, modulus = self._compute(strain)
        else:
            stress = np.zeros_like(strain)
            modulus = np.zeros_like(strain)
            for i in range(strain.size):
                stress[i], modulus[i] = self._compute(strain[i])
        return stress, modulus

    def _compute(self, e):
        pass


class BiLinear(MaterialBase):

    def __init__(self, E1, E2, sy):
        self.E1 = E1
        self.E2 = E2
        self.sy = sy
        self.ey = sy / E1
        self.ep = 0.0
        self.sp = 0.0

    def _compute(self, e):
        if np.abs(e - self.ep) < self.ey:
            s = self.E1 * (e - self.ep) + self.sp
            m = self.E1
        else:
            dr = np.sign(e - self.ep)
            dep = e - self.ep - dr * self.ey
            dsp = self.E2 * dep
            s = self.sp + dr * self.sy + dsp
            m = self.E2
            self.ep += dep
            self.sp += dsp
        return s, m


class Mattock(MaterialBase):
    """
    Menegotto-Pinto 형태의 bilinear 곡선 모델
    """

    def __init__(self, E1, A, B, C):
        """
        Mattock 모델 상수 초기화
        :param E1: 초기 기울기
        :param A: = E2 / E1
        :param B: = E1 / fo = 1 / eo, (eo, fo)는 두 점근선 간 교차점
        :param C: 두 점근선 간 교차점 부위의 곡선 형태, 클수록 뾰족해짐
        """
        self.E1 = E1
        self.A = A
        self.B = B
        self.C = C

    def _compute(self, e):
        se = np.sign(e)
        ae = np.abs(e)
        s = se * self.E1 * ae * (self.A + (1.0 - self.A) / (1.0 + (self.B * ae)**self.C)**(1.0 / self.C))
        m = self.E1 * (self.A + (1.0 - self.A) / (1.0 + (self.B * ae)**self.C)**(1.0 / self.C)) - self.E1 * ae * (1.0 - self.A) * self.B * (self.B * ae)**(self.C - 1.0) / (1.0 + (self.B * ae)**self.C)**(1.0 / self.C + 1.0)
        return s, m


class Linear(MaterialBase):

    def __init__(self, E):
        self.E = E

    def _compute(self, e):
        return self.E * e, self.E


LinElas = Linear


class ConcreteKDSAnalysis(MaterialBase):

    def __init__(self, fck, mc=2300):
        self.fck = fck
        self.mc = mc
        self.fcm, self.ecor, self.ecur, self.k, self.ectm, self.mct = KDS_concrete_parameters_analysis(fck, mc)

    def _compute(self, e):
        if (e < -self.ecur) or (e > self.ectm):
            s, m = 0.0, 0.0
        elif e < 0.0:
            r = -e / self.ecor
            s = -self.fcm * ((self.k * r - r**2) / (1.0 + (self.k - 2.0) * r))
            m = self.fcm / self.ecor * (-r * (self.k - 2.0) * (self.k - r) + (self.k - 2.0 * r) * (r * (self.k - 2.0) + 1.0)) / (r * (self.k - 2.0) + 1.0)**2
        else:  # ec > 0.0
            s = self.mct * e
            m = self.mct
        return s, m


class ConcreteKDSDesign(MaterialBase):

    def __init__(self, fck, phic=0.65):
        self.fck, self.eco, self.ecu, self.n, self.phic, self.ectd, self.mct = KDS_concrete_parameters_design(fck, phic)

    def _compute(self, e):
        if (e < -self.ecu) or (e > self.ectd):
            s, m = 0.0, 0.0
        elif e < -self.eco:
            s = -self.phic * (0.85 * self.fck)
            m = 0.0
        elif e < 0.0:
            s = -self.phic * (0.85 * self.fck) * (1.0 - (1.0 + e / self.eco)**self.n)
            m = self.phic * (0.85 * self.fck) * self.n * (1.0 + e / self.eco)**(self.n - 1.0) / self.eco
        else:  # e > 0.0
            s = self.mct * e
            m = self.mct
        return s, m


# -----------------------------------------------------------------------------


def mattock(e, E1, A, B, C):
    """
    A = E2 / E1
    B = E1(1-A) / sy
    C = about 10 (sharper as C higher)
    """
    se = np.sign(e)
    ae = np.abs(e)
    s = se * E1 * ae * (A + (1 - A) / (1 + (B * ae)**C)**(1 / C))
    d = E1 * (A + (1 - A) / (1 + (B * ae)**C)**(1 / C)) - E1 * ae * (1 - A) * B * (B * ae)**(C - 1) / (1 + (B * ae)**C)**(1 / C + 1)
    return s, d


def linear(e, E):
    if np.isscalar(e):
        return E * e, E
    else:
        return E * e, E * np.ones_like(e)


linelas = linear


def multilin(e, pe, ps):
    spl = interpolate.splrep(pe, ps, k=1)
    s = interpolate.splev(e, spl, ext=1)
    d = interpolate.splev(e, spl, der=1, ext=1)
    return s, d


# -----------------------------------------------------------------------------


def elastic_modulus_of_concrete(fck, mc=2300):
    if fck <= 40:
        df = 4
    elif fck >= 60:
        df = 6
    else:
        df = 0.1 * fck
    fcm = fck + df
    Ec = 0.077 * mc**(1.5) * fcm**(1 / 3)
    return Ec


def KDS_concrete_Df(fck):
    if fck <= 40.0:
        return 4.0
    if fck >= 60.0:
        return 6.0
    return 0.1 * fck


def KDS_concrete_fcm(fck):
    return fck + KDS_concrete_Df(fck)


def KDS_concrete_fctm(fck):
    fcm = KDS_concrete_fcm(fck)
    return 0.30 * fcm**(2 / 3)


def KDS_concrete_fctk(fck):
    fctm = KDS_concrete_fctm(fck)
    return 0.70 * fctm


def KDS_concrete_Ec(fck, mc=2300):
    fcm = KDS_concrete_fcm(fck)
    return 0.077 * mc**1.5 * fcm**(1 / 3)


# -----------------------------------------------------------------------------


def KDS_concrete_parameters_analysis(fck, mc=2300):
    table_fck = np.array([18, 21, 24, 27, 30, 35, 40, 50, 60, 70, 80, 90])
    table_ecor = np.array([1.83, 1.90, 1.97, 2.03, 2.09, 2.18, 2.26, 2.42, 2.57, 2.68, 2.79, 2.80]) / 1000
    table_ecur = np.array([3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8]) / 1000
    spl_ecor = interpolate.splrep(table_fck, table_ecor, k=1)
    spl_ecur = interpolate.splrep(table_fck, table_ecur, k=1)
    ecor = interpolate.splev(fck, spl_ecor).item(0)
    ecur = interpolate.splev(fck, spl_ecur).item(0)

    fcm = KDS_concrete_fcm(fck)
    k = 1.1 * KDS_concrete_Ec(fck, mc) * ecor / fcm
    mct = 1.1 * KDS_concrete_Ec(fck, mc)
    fctm = KDS_concrete_fctm(fck)
    ectm = fctm / mct

    return fcm, ecor, ecur, k, ectm, mct


def _KDS_concrete_analysis(ec, fcm, ecor, ecur, k, ectm, mct):
    if (ec < -ecur) or (ec > ectm):
        s = 0.0
        m = 0.0
    elif ec < 0.0:
        r = -ec / ecor
        s = -fcm * ((k * r - r**2) / (1.0 + (k - 2) * r))
        m = fcm / ecor * (-r * (k - 2.0) * (k - r) + (k - 2.0 * r) * (r * (k - 2.0) + 1.0)) / (r * (k - 2.0) + 1.0)**2
    else:  # ec > 0.0
        s = mct * ec
        m = mct
    return s, m


def KDS_concrete_analysis(ec, fcm, ecor, ecur, k, ectm, mct):
    if np.isscalar(ec):
        fc, mc = _KDS_concrete_analysis(ec, fcm, ecor, ecur, k, ectm, mct)
    else:
        fc = np.zeros_like(ec)
        mc = np.zeros_like(ec)
        for i in range(ec.size):
            fc[i], mc[i] = _KDS_concrete_analysis(ec[i], fcm, ecor, ecur, k, ectm, mct)
    return fc, mc


# -----------------------------------------------------------------------------


def KDS_concrete_parameters_design(fck, phic=0.65):
    if fck <= 40.0:
        n, eco, ecu = 2.0, 0.002, 0.0033
    else:
        n = 1.2 + 1.5 * ((100.0 - fck) / 60.0)**4
        eco = 0.002 + ((fck - 40.0) / 100000)
        ecu = 0.0033 - ((fck - 40.0) / 100000)

    mct = phic * (0.85 * fck) * n / eco
    fctd = phic * KDS_concrete_fctk(fck)
    ectd = fctd / mct

    return fck, eco, ecu, n, phic, ectd, mct


def _KDS_concrete_design(ec, fck, eco, ecu, n, phic, ectd, mct):
    if (ec < -ecu) or (ec > ectd):
        s, m = 0.0, 0.0
    elif ec < -eco:
        s = -phic * (0.85 * fck)
        m = 0.0
    elif ec < 0.0:
        s = -phic * (0.85 * fck) * (1.0 - (1.0 + ec / eco)**n)
        m = phic * (0.85 * fck) * n * (1.0 + ec / eco)**(n - 1.0) / eco
    else:  # ec > 0.0
        s = mct * ec
        m = mct
    return s, m


def KDS_concrete_design(ec, fck, eco, ecu, n, phic, ectd, mct):
    if np.isscalar(ec):
        fc, mc = _KDS_concrete_design(ec, fck, eco, ecu, n, phic, ectd, mct)
    else:
        fc = np.zeros_like(ec)
        mc = np.zeros_like(ec)
        for i in range(ec.size):
            fc[i], mc[i] = _KDS_concrete_design(ec[i], fck, eco, ecu, n, phic, ectd, mct)
    return fc, mc


# -----------------------------------------------------------------------------


def KDS_concrete_parameters(analysis_type, fck, phic=0.65, mc=2300):
    if analysis_type.upper() == 'ANALYSIS':
        params = KDS_concrete_parameters_analysis(fck, mc)
        return params
    else:
        params = KDS_concrete_parameters_design(fck, phic)
        return params


def KDS_concrete(ec, analysis_type, *params):
    if analysis_type.upper() == 'ANALYSIS':
        return KDS_concrete_analysis(ec, *params)
    else:
        return KDS_concrete_design(ec, *params)


# -----------------------------------------------------------------------------


def envelope_SCf(fck, hb, phicc=0.91, phict=0.80):
    table_fck = [120, 150, 180, ]  # MPa
    table_Ec = [40e3, 43e3, 45e3, ]  # MPa
    table_eu = [0.003, 0.0035, 0.004, ]
    table_fcrk = [5.0, 6.3, 7.6, ]  # MPa
    table_ftk = [7.0, 9.0, 11.0, ]  # MPa
    table_Gf = [37.9, 37.9, 37.9, ]  # N/mm, valid for SC180f only
    wu = 0.3  # mm
    wlim = 5.3  # mm

    ps = list()
    pe = list()

    # comp side

    spl_Ec = interpolate.splrep(table_fck, table_Ec, k=1)
    Ec = interpolate.splev(fck, spl_Ec).item(0)

    fcd = phicc * fck
    spl_eu = interpolate.splrep(table_fck, table_eu, k=1)
    eu = interpolate.splev(fck, spl_eu).item(0)

    ps.append(-0.85 * fcd)
    pe.append(-eu)

    ps.append(-0.85 * fcd)
    pe.append(-0.85 * fcd / Ec)

    ps.append(0.0)
    pe.append(0.0)

    # tens side

    spl_fcrk = interpolate.splrep(table_fck, table_fcrk, k=1)
    fcrk = interpolate.splev(fck, spl_fcrk).item(0)
    fcrd = phict * fcrk

    spl_ftk = interpolate.splrep(table_fck, table_ftk, k=1)
    ftk = interpolate.splev(fck, spl_ftk).item(0)
    ftd = phict * ftk

    spl_Gf = interpolate.splrep(table_fck, table_Gf, k=1)
    Gf = interpolate.splev(fck, spl_Gf).item(0)

    lch = Gf * Ec / ftk**2
    Leq = 0.8 * (1.0 - (1.05 + 6 * hb / lch)**(-4)) * hb

    ps.append(fcrd)
    pe.append(fcrd / Ec)

    ps.append(ftd)
    pe.append(fcrd / Ec + wu / Leq)

    ps.append(0.0)
    pe.append(fcrd / Ec + wlim / Leq)

    return pe, ps


# -----------------------------------------------------------------------------


def stress2strain(s, fn, args):
    def objfn(x):
        e = fn(x, *args) - s
        return e**2
    result = optimize.minimize_scalar(objfn, method='brent')
    return result.x
