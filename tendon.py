import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as optimize

import my.material
import my.util


# -----------------------------------------------------------------------------
# Strand, Tendon
# -----------------------------------------------------------------------------


def strand(e, cw, hw, Ap=138.7):
    """심선과 측선이 다른 재료로 이루어진 강연선의 응력을 계산합니다.

    : param e: strain vector
    : param cw: properties of core wire, {'fn': function_name, 'args': [argument list of fn], 'D': diameter, 'nu': Poisson's ratio}
    : param hw: properties of helical wire, {'fn': function_name, 'args': [argument list of fn], 'D': diameter, 'nu': Poisson's ratio, 'p': pitch_length}
    : param Ap: strand area
    : returns s: stress vector
    """

    R1 = cw['D'] / 2
    R2 = hw['D'] / 2
    r2 = R1 + R2
    A1 = np.pi * R1 ** 2
    A2 = np.pi * R2 ** 2
    p2 = hw['p']
    nu1 = cw['nu']
    nu2 = hw['nu']
    alpha2 = np.arctan(p2 / (2 * np.pi * r2))
    C1 = (r2 * np.tan(alpha2) ** 2 - nu1 * R1) / (r2 * np.tan(alpha2) ** 2 + r2 + nu2 * R2)
    fncw = getattr(my.material, cw['fn'])
    fnhw = getattr(my.material, hw['fn'])

    s = (A1 * fncw(e, *cw['args']) + 6 * A2 * fnhw(C1 * e, *hw['args']) * np.sin(alpha2)) / Ap

    return s


class Tendon:
    """
    Tendon force distribution for a given tendon profile
    """

    def __init__(self, Ep, Ap, t, x, y, z, active='R', k=2):  # active: 'L'eft / 'R'ight / 'B'oth
        """
        Initialize Tendon class
        :param Ep: Elastic modulus of tendon
        :param Ap: Area of tendon
        :param t: Parametric representation of (x, y, z)
        :param x: x-coordinates of tendon
        :param y: y-coordinates of tendon
        :param z: z-coordinates of tendon
        :param active: 'L'eft when prestressing at the minimum of t, 'R'ight when prestressing at the maximum of t, 'B'oth when presstressing at both ends
        :param k: degree of interpolation function for tendon
        """
        #
        self.Ep = Ep
        self.Ap = Ap
        #
        self.t = t
        #
        self.active = active.upper()[0]
        if self.active == 'L':
            self._t_active = t[0]
            self._t_passive = t[-1]
        elif self.active == 'R':
            self._t_active = t[-1]
            self._t_passive = t[0]
        else:
            self._t_active = (t[0], t[-1],)
            self._t_passive = (t[0] + t[-1]) / 2.0  # tPassive should be determined such that force of LHS = force of RHS
        # Profile
        self._spl_profile = interpolate.splprep([x, y, z], s=0, k=k, u=self.t)[0]
        #
        der1 = interpolate.splev(self.t, self._spl_profile, der=1)
        der2 = interpolate.splev(self.t, self._spl_profile, der=2)
        # Arc length
        self._ds = np.sqrt(der1[0] ** 2 + der1[1] ** 2 + der1[2] ** 2)
        self._spl_ds = interpolate.splrep(self.t, self._ds)
        # Angle change
        dc = np.sqrt((der2[2] * der1[1] - der2[1] * der1[2]) ** 2 + (der2[0] * der1[2] - der2[2] * der1[0]) ** 2 + (der2[1] * der1[0] - der2[0] * der1[1]) ** 2) / self._ds ** 2
        self._spl_dc = interpolate.splrep(self.t, dc)

    def profile(self, u):
        return interpolate.splev(u, self._spl_profile, ext=1)

    def _integrate_spl_scalar(self, ui, spl):
        if self.active == 'R':  # passive ---- active
            return -interpolate.splint(self._t_active, ui, spl)
        elif self.active == 'L':  # active ---- passive
            return interpolate.splint(self._t_active, ui, spl)
        else:  # active ---- active
            if ui <= self._t_passive:
                return interpolate.splint(self._t_active[0], ui, spl)
            else:
                return -interpolate.splint(self._t_active[1], ui, spl)

    def _integrate_spl_scalar_array(self, u, spl):
        if np.isscalar(u):
            return self._integrate_spl_scalar(u, spl)
        else:
            return np.array([self._integrate_spl_scalar(ui, spl) for ui in u])

    def arc_length(self, u):
        return self._integrate_spl_scalar_array(u, self._spl_ds)

    def angle_change(self, u):
        return self._integrate_spl_scalar_array(u, self._spl_dc)

    def force_jack(self, u, Pj, mu, kp):
        return Pj * np.exp(-(mu * self.angle_change(u) + kp * self.arc_length(u)))

    def _work_jack(self, u, Pj, mu, kp):
        force = self.force_jack(self.t, Pj, mu, kp)
        spl_dw = interpolate.splrep(self.t, force * self._ds)
        return self._integrate_spl_scalar_array(u, spl_dw)

    def force_set(self, u, Pj, mu, kp, slip):
        sEA = slip * self.Ep * self.Ap
        work_loss_passive = 2 * (self._work_jack(self._t_passive, Pj, mu, kp) - self.force_jack(self._t_passive, Pj, mu, kp) * self.arc_length(self._t_passive))
        if sEA < work_loss_passive:

            def objfn(ti):
                e = sEA - 2 * (self._work_jack(ti, Pj, mu, kp) - self.force_jack(ti, Pj, mu, kp) * self.arc_length(ti))
                return e ** 2

            def force_loss_right_active(tmin, tmax):
                tset = optimize.minimize_scalar(objfn, bounds=(tmin, tmax), method='bounded').x
                tas = np.hstack((tset, self.t[self.t > tset]))
                fas = 2 * (self.force_jack(tas, Pj, mu, kp) - self.force_jack(tset, Pj, mu, kp))
                return tset, tas, fas

            def force_loss_left_active(tmin, tmax):
                tset = optimize.minimize_scalar(objfn, bounds=(tmin, tmax), method='bounded').x
                tas = np.hstack((self.t[self.t < tset], tset))
                fas = 2 * (self.force_jack(tas, Pj, mu, kp) - self.force_jack(tset, Pj, mu, kp))
                return tset, tas, fas

            if self.active == 'R':  # passive ---- active
                self.set_limit, tas2, fas2 = force_loss_right_active(self._t_passive, self._t_active)
                splas = interpolate.splrep(tas2, fas2, k=1)
            elif self.active == 'L':  # active ---- passive
                self.set_limit, tas1, fas1 = force_loss_left_active(self._t_active, self._t_passive)
                splas = interpolate.splrep(tas1, fas1, k=1)
            else:  # active ---- active
                tset1, tas1, fas1 = force_loss_left_active(self._t_active[0], self._t_passive)
                tset2, tas2, fas2 = force_loss_right_active(self._t_passive, self._t_active[1])
                splas = interpolate.splrep(np.hstack((tas1, tas2)), np.hstack((fas1, fas2)), k=1)
                self.set_limit = np.array([tset1, tset2])
        else:
            force_passive = (self._work_jack(self._t_passive, Pj, mu, kp) - sEA / 2) / self.arc_length(self._t_passive)
            splas = interpolate.splrep(self.t, 2 * (self.force_jack(self.t, Pj, mu, kp) - force_passive), k=1)
            self.set_limit = None
        return self.force_jack(u, Pj, mu, kp) - interpolate.splev(u, splas, ext=1)

    def plot(self, u=None, Pj=None, mu=None, kp=None, slip=None):
        if u is None:
            u = self.t
        if Pj is None:
            nsp = 3
        else:
            nsp = 4

        plt.figure()

        plt.subplot(nsp, 1, 1)
        for i in range(len(self.t) - 1):
            td = np.linspace(self.t[i], self.t[i + 1], 11)
            xyz = self.profile(td)
            plt.plot(xyz[0], xyz[1], 'k-')
        xyz = self.profile(self.t)
        plt.plot(xyz[0], xyz[1], 'k.', label='Given')
        xyz = self.profile(u)
        plt.plot(xyz[0], xyz[1], 'ro', label='Interpolated')
        my.util.adjust_axis()
        plt.ylabel('Profile')
        plt.legend()
        plt.grid()

        plt.subplot(nsp, 1, 2)
        for i in range(len(self.t) - 1):
            td = np.linspace(self.t[i], self.t[i + 1], 11)
            plt.plot(td, self.arc_length(td), 'k-')
        plt.plot(self.t, self.arc_length(self.t), 'k.', label='at t')
        plt.plot(u, self.arc_length(u), 'ro', label='at u')
        my.util.adjust_axis()
        plt.ylabel('Arc length')
        plt.legend()
        plt.grid()

        plt.subplot(nsp, 1, 3)
        for i in range(len(self.t) - 1):
            td = np.linspace(self.t[i], self.t[i + 1], 11)
            plt.plot(td, self.angle_change(td), 'k-')
        plt.plot(self.t, self.angle_change(self.t), 'k.', label='at t')
        plt.plot(u, self.angle_change(u), 'ro', label='at u')
        my.util.adjust_axis()
        plt.ylabel('Angle change')
        plt.legend()
        plt.grid()

        if Pj is not None:
            plt.subplot(nsp, 1, 4)
            for i in range(len(self.t) - 1):
                td = np.linspace(self.t[i], self.t[i + 1], 11)
                plt.plot(td, self.force_jack(td, Pj, mu, kp), 'k-')
            plt.plot(self.t, self.force_jack(self.t, Pj, mu, kp), 'k.')
            plt.plot(u, self.force_jack(u, Pj, mu, kp), 'bo', label='Jacking')
            if slip is not None:
                for i in range(len(self.t) - 1):
                    td = np.linspace(self.t[i], self.t[i + 1], 11)
                    plt.plot(td, self.force_set(td, Pj, mu, kp, slip), 'k-')
                plt.plot(self.t, self.force_set(self.t, Pj, mu, kp, slip), 'k.')
                plt.plot(u, self.force_set(u, Pj, mu, kp, slip), 'ro', label='After set')
                if self.set_limit is not None:
                    plt.plot(self.set_limit, self.force_jack(self.set_limit, Pj, mu, kp), 'go', label='Set influence limit')
            my.util.adjust_axis()
            plt.ylabel('Force')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()


class TendonAnalysis:

    def __init__(self, tendon_strains_exp, tendons, Fj, u, Ac, Icz, Icy, Qcz=0.0, Qcy=0.0, Icyz=0.0, ismarts=0):
        self.tendon_strains_exp = tendon_strains_exp
        self.tendons = tendons
        self.Fj = Fj
        self.u = u
        self.Ac = Ac
        self.Icz = Icz
        self.Icy = Icy
        if Qcz == 0.0:
            self.Qcz = np.zeros_like(Ac)
        if Qcy == 0.0:
            self.Qcy = np.zeros_like(Ac)
        if Icyz == 0.0:
            self.Icyz = np.zeros_like(Ac)
        self.ismarts = ismarts
        #
        self.ntendons = len(tendons)
        self.nsections = u.size
        #
        self.mu = np.zeros(self.ntendons)
        self.kp = np.zeros(self.ntendons)
        self.slip = np.zeros(self.ntendons)
        self.Ec = 0.0
        #
        self.tendon_strains_opt = dict()  # self.tendon_strains_opts[kstep][i_tendon] = strain vector
        self._ijacked = list()

    def optimize(self, ksteps, ijackings, x0, xb):
        """
        ksteps: [('1p', '1s'), (None, '2s'), ('3p', '3s'), ], jacking stage is optional(=None), setting stage is mandatory
        ijackings: order of jacking tendons, [(0, ), (1, ), (2, ), ] or [(0, ), (1, 2, ), ]
        x0, xb: optimization variables, [mu, kp, slip, Ec]
        """
        smart_strains_inc_exp = dict()
        kstep = ksteps[0]
        if kstep[0] is not None:
            smart_strains_inc_exp[kstep[0]] = dict()
            smart_strains_inc_exp[kstep[0]][self.ismarts] = self.tendon_strains_exp[kstep[0]][self.ismarts]
        smart_strains_inc_exp[kstep[1]] = dict()
        smart_strains_inc_exp[kstep[1]][self.ismarts] = self.tendon_strains_exp[kstep[1]][self.ismarts]
        prev_step = kstep[1]
        for kstep in ksteps[1:]:
            if kstep[0] is not None:
                smart_strains_inc_exp[kstep[0]] = dict()
                smart_strains_inc_exp[kstep[0]][self.ismarts] = self.tendon_strains_exp[kstep[0]][self.ismarts] - self.tendon_strains_exp[prev_step][self.ismarts]
            smart_strains_inc_exp[kstep[1]] = dict()
            smart_strains_inc_exp[kstep[1]][self.ismarts] = self.tendon_strains_exp[kstep[1]][self.ismarts] - self.tendon_strains_exp[prev_step][self.ismarts]
            prev_step = kstep[1]

        def objfn(r):
            self.mu[:], self.kp[:], self.slip[:], self.Ec = my.util.r2x(r, xb)
            sum_error2 = 0.0
            self._ijacked = list()

            kstep, ijacking = ksteps[0], ijackings[0]

            # 1st jacking stage
            if kstep[0] is not None:
                smart_strains_inc_opt = self._jacking_tendon_force(self.ismarts, self.u, 'J') / (self.tendons[self.ismarts].Ep * self.tendons[self.ismarts].Ap)
                e = (smart_strains_inc_opt - smart_strains_inc_exp[kstep[0]][self.ismarts]) / smart_strains_inc_exp[kstep[0]][self.ismarts]
                sum_error2 += sum(e ** 2)

            # 1st setting stage
            smart_strains_inc_opt = self._jacking_tendon_force(self.ismarts, self.u, 'S') / (self.tendons[self.ismarts].Ep * self.tendons[self.ismarts].Ap)
            e = (smart_strains_inc_opt - smart_strains_inc_exp[kstep[1]][self.ismarts]) / smart_strains_inc_exp[kstep[1]][self.ismarts]
            sum_error2 += sum(e ** 2)

            for i in ijacking:
                self._ijacked.append(i)

            for kstep, ijacking in zip(ksteps[1:], ijackings[1:]):
                # jacking stages
                if kstep[0] is not None:
                    for j in range(self.nsections):
                        section_strain = self._solve_equilibriums_at_section(ijacking, j, self.Ec, 'J')[0]
                        smart_strains_inc_opt[j] = self.strain_at_point(self.tendons[self.ismarts].profile(self.u[j]), section_strain)
                    e = (smart_strains_inc_opt - smart_strains_inc_exp[kstep[0]][self.ismarts]) / smart_strains_inc_exp[kstep[0]][self.ismarts]
                    sum_error2 += sum(e ** 2)

                # setting stages
                for j in range(self.nsections):
                    section_strain = self._solve_equilibriums_at_section(ijacking, j, self.Ec, 'S')[0]
                    smart_strains_inc_opt[j] = self.strain_at_point(self.tendons[self.ismarts].profile(self.u[j]), section_strain)
                e = (smart_strains_inc_opt - smart_strains_inc_exp[kstep[1]][self.ismarts]) / smart_strains_inc_exp[kstep[1]][self.ismarts]
                sum_error2 += sum(e ** 2)

                for i in ijacking:
                    self._ijacked.append(i)

            return np.sqrt(sum_error2)

        result = optimize.minimize(objfn, my.util.x2r(x0, xb), method='L-BFGS-B', bounds=my.util.make_rb(xb), options={'gtol': 1e-7})
        self.mu[:], self.kp[:], self.slip[:], self.Ec = my.util.r2x(result.x, xb)

        return self.mu, self.kp, self.slip, self.Ec

    def update(self, ksteps, ijackings):
        self._ijacked = list()

        kstep, ijacking = ksteps[0], ijackings[0]
        if kstep[0] is not None:
            self._update_jacking_tendon_strain(kstep[0], ijacking, 'J')
        self._update_jacking_tendon_strain(kstep[1], ijacking, 'S')
        for i in ijacking:
            self._ijacked.append(i)
        self._lastStep = kstep[1]

        for kstep, ijacking in zip(ksteps[1:], ijackings[1:]):
            if kstep[0] is not None:
                self._update_tendon_strain(kstep[0], ijacking, self.Ec, 'J')
            self._update_tendon_strain(kstep[1], ijacking, self.Ec, 'S')
            for i in ijacking:
                self._ijacked.append(i)
            self._lastStep = kstep[1]

        return self.tendon_strains_opt

    def plot(self, var='Strain'):  # var = 'Strain' / 'Stress' / 'Force'
        for kstep in self.tendon_strains_opt:
            plt.figure()
            plt.title('Step = {}'.format(kstep))
            for i in self.tendon_strains_opt[kstep]:
                if var == 'Strain':
                    plt.plot(self.u, self.tendon_strains_exp[kstep][i], 'o-', label='Tendon #{:d}: Measured'.format(i))
                    plt.plot(self.u, self.tendon_strains_opt[kstep][i], '.-', label='Tendon #{:d}: Optimized'.format(i))
                elif var == 'Stress':
                    plt.plot(self.u, self.tendon_strains_exp[kstep][i] * self.tendons[i].Ep, 'o-', label='Tendon #{:d}: Measured'.format(i))
                    plt.plot(self.u, self.tendon_strains_opt[kstep][i] * self.tendons[i].Ep, '.-', label='Tendon #{:d}: Optimized'.format(i))
                else:  # 'Force'
                    plt.plot(self.u, self.tendon_strains_exp[kstep][i] * self.tendons[i].Ep * self.tendons[i].Ap, 'o-', label='Tendon #{:d}: Measured'.format(i))
                    plt.plot(self.u, self.tendon_strains_opt[kstep][i] * self.tendons[i].Ep * self.tendons[i].Ap, '.-', label='Tendon #{:d}: Optimized'.format(i))
            my.util.adjust_axis()
            plt.xlabel('Location')
            plt.ylabel(var)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

    @staticmethod
    def strain_at_point(xyz, section_strain):
        return section_strain[0] - xyz[1] * section_strain[1] + xyz[2] * section_strain[2]

    def _jacking_tendon_force(self, i, usections, stage):  # stage='J'acking / 'S'etting
        if stage.upper() == 'J':
            return self.tendons[i].force_jack(usections, self.Fj[i], self.mu[i], self.kp[i])
        else:
            return self.tendons[i].force_set(usections, self.Fj[i], self.mu[i], self.kp[i], self.slip[i])

    def _b_jacking_tendons_at_section(self, ijacking, usection, stage, b=None):
        if b is None:
            b = np.zeros(3)
        for i in ijacking:
            tf = self._jacking_tendon_force(i, usection, stage)
            b[0] -= tf
            b[1] -= tf * self.tendons[i].profile(usection)[1]
            b[2] -= tf * self.tendons[i].profile(usection)[2]
        return b

    def _A_jacked_tendons_at_section(self, usection, A=None):
        if A is None:
            A = np.zeros((3, 3))
        for i in self._ijacked:
            xyz = self.tendons[i].profile(usection)
            tf = self.tendons[i].Ep * self.tendons[i].Ap
            A[0, 0] += tf
            A[1, 0] += tf * xyz[1]
            A[2, 0] += tf * xyz[2]
            #
            tf1 = tf * xyz[1]
            A[0, 1] -= tf1
            A[1, 1] -= tf1 * xyz[1]
            A[2, 1] -= tf1 * xyz[2]
            #
            tf2 = tf * xyz[2]
            A[0, 2] += tf2
            A[1, 2] += tf2 * xyz[1]
            A[2, 2] += tf2 * xyz[2]
        return A

    def _A_concrete_at_section(self, jsection, Ec, A=None):
        if A is None:
            A = np.zeros((3, 3))
        A[0, 0] += Ec * self.Ac[jsection]
        A[1, 0] += Ec * self.Qcz[jsection]
        A[2, 0] += Ec * self.Qcy[jsection]
        #
        A[0, 1] -= Ec * self.Qcz[jsection]
        A[1, 1] -= Ec * self.Icz[jsection]
        A[2, 1] -= Ec * self.Icyz[jsection]
        #
        A[0, 2] += Ec * self.Qcy[jsection]
        A[1, 2] += Ec * self.Icyz[jsection]
        A[2, 2] += Ec * self.Icy[jsection]
        return A

    def _solve_equilibriums_at_section(self, ijacking, jsection, Ec, stage):
        A = np.zeros((3, 3))
        b = np.zeros((3))
        b = self._b_jacking_tendons_at_section(ijacking, self.u[jsection], stage, b)
        A = self._A_jacked_tendons_at_section(self.u[jsection], A)
        A = self._A_concrete_at_section(jsection, Ec, A)
        section_strain = np.linalg.solve(A, b)
        return section_strain, A, b

    def _update_jacking_tendon_strain(self, kstep, ijacking, stage):
        self.tendon_strains_opt[kstep] = dict()
        for i in ijacking:
            self.tendon_strains_opt[kstep][i] = self._jacking_tendon_force(i, self.u, stage) / (self.tendons[i].Ep * self.tendons[i].Ap)

    def _update_tendon_strain(self, kstep, ijacking, Ec, stage):
        self.tendon_strains_opt[kstep] = dict()
        for i in self._ijacked:
            self.tendon_strains_opt[kstep][i] = np.zeros(self.nsections)
        for j in range(self.nsections):
            section_strain = self._solve_equilibriums_at_section(ijacking, j, Ec, stage)[0]
            for i in self._ijacked:
                self.tendon_strains_opt[kstep][i][j] = self.tendon_strains_opt[self._lastStep][i][j] + self.strain_at_point(self.tendons[i].profile(self.u[j]), section_strain)
        for i in ijacking:
            self.tendon_strains_opt[kstep][i] = self._jacking_tendon_force(i, self.u, stage) / (self.tendons[i].Ep * self.tendons[i].Ap)
