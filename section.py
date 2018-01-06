import json

import matplotlib.pyplot as plt
import numpy as np

import my.material
import my.util


# -----------------------------------------------------------------------------
# Section analysis
# -----------------------------------------------------------------------------


class SectionAnalysis:

    def __init__(self, inp):
        #
        if isinstance(inp, dict):
            self.sa = inp
        else:
            with open(inp, 'r') as f:
                self.sa = json.load(f)
        #
        self._init_section_analysis()

    def run(self, step_no=None):
        self._run_section_analysis(step_no)

    def plot_section(self):
        # Force-strain
        plt.figure()
        for k2 in sorted(self.results):
            plt.plot(self.results[k2]['section']['strain'], self.results[k2]['section']['force'], '.-', label='step %s' % k2)
        my.util.adjust_axis()
        plt.grid()
        plt.legend()
        plt.xlabel('Strain')
        plt.ylabel('Force')
        plt.tight_layout()
        plt.show()
        # Moment-curvature
        plt.figure()
        for k2 in sorted(self.results):
            plt.plot(self.results[k2]['section']['curvature'], self.results[k2]['section']['moment'], '.-', label='step %s' % k2)
        my.util.adjust_axis()
        plt.grid()
        plt.legend()
        plt.xlabel('Curvature')
        plt.ylabel('Moment')
        plt.tight_layout()
        plt.show()

    def plot_cell(self, cell_no):
        plt.figure()
        for sno in sorted(self.results):
            plt.plot(self.results[sno]['cell'][cell_no]['strain'], self.results[sno]['cell'][cell_no]['stress'], '.-', label='cell %d, step %s' % (cell_no, sno))
        my.util.adjust_axis()
        plt.grid()
        plt.legend()
        plt.xlabel('Strain')
        plt.ylabel('Stress')
        plt.tight_layout()
        plt.show()

    def draw_section(self):
        sections = self.sa['Section']
        plt.figure()
        for section in sections.values():
            ys = section['Y']
            if section['Type'][:4].upper() == 'RECT':
                bs = section['B']
                for y, b in zip(ys, bs):
                    px = np.array([-b[0], b[0], b[1], -b[1], -b[0]]) / 2
                    py = np.array([y[0], y[0], y[1], y[1], y[0]])
                    plt.plot(px, py, 'k-')
            elif section['Type'][:4].upper() == 'POIN':
                plt.plot(np.zeros_like(ys), ys, 'bo')
        plt.axis('equal')
        plt.grid()
        plt.show()

    @staticmethod
    def _convert_inplist_to_list(inp_list):
        if inp_list is None:
            return list()
        lst = list()
        for li in inp_list:
            if isinstance(li, list):
                for i in range(li[0], li[1] + 1):
                    lst.append(i)
            else:
                lst.append(li)
        return lst

    def _init_section_analysis(self):
        self.results = dict()
        print('-- Processing input data')
        # keyword: Section
        self.cells = dict()
        self.set_active = set()
        sections = self.sa['Section']
        for section in sections.values():
            vmatr = self.sa['Material'][section['Material']]
            vfn = getattr(my.material, vmatr['fn'])  # globals()[vmatr['fn']]
            vargs = vmatr['args']
            ys = section['Y']
            if section['Type'][:4].upper() == 'RECT':
                bs = section['B']
                for i in range(len(ys)):
                    nd = section['Id'][i][1] - section['Id'][i][0] + 1
                    yi = np.linspace(ys[i][0], ys[i][1], num=nd + 1)
                    bi = np.linspace(bs[i][0], bs[i][1], num=nd + 1)
                    ii = np.arange(section['Id'][i][0], section['Id'][i][1] + 1)
                    dy2 = (ys[i][1] - ys[i][0]) / nd / 2
                    for j in range(nd):
                        self.cells[ii[j]] = {'Y': yi[j] + dy2, 'A': (bi[j] + bi[j + 1]) * dy2, 'M': {'fn': vfn, 'args': vargs}, 'e': 0.0, 's': 0.0}
                    if section['Active'] is True:
                        self.set_active.update(ii)
            elif section['Type'][:4].upper() == 'POIN':
                va = section['A']  # get_value(s, 'A')
                for i in range(len(ys)):
                    self.cells[section['Id'][i]] = {'Y': ys[i], 'A': va[i], 'M': {'fn': vfn, 'args': vargs}, 'e': 0.0, 's': 0.0}
                if section['Active'] is True:
                    self.set_active.update(section['Id'])
        # keyword: Output
        self.output_filename = self.sa['Output']['File']
        self.set_output = set(self._convert_inplist_to_list(self.sa['Output']['Id']))
        with open(self.output_filename, 'w') as fout:
            fout.write('Step no,Axial strain,Axial force,Curvature,Moment')
            for i in self.set_output:
                fout.write(',Strain(%d),Stress(%d)' % (i, i))
            fout.write(',Remark\n')
        # keyword: Tolerance
        self.tol = self.sa['Tolerance']

    def _run_section_analysis(self, step_no):

        def cell_strain(strain, curvature, cell):
            return strain - curvature * cell['Y'] + cell['e']

        def cell_stress(e, cell):
            return cell['M']['fn'](e, *cell['M']['args'])

        def alloc_results(num, results, set_out):
            results['section'] = dict()
            results['section']['strain'] = np.zeros(num)
            results['section']['curvature'] = np.zeros(num)
            results['section']['force'] = np.zeros(num)
            results['section']['moment'] = np.zeros(num)
            results['cell'] = dict()
            for ic in set_out:
                results['cell'][ic] = dict()
                results['cell'][ic]['strain'] = np.zeros(num)
                results['cell'][ic]['stress'] = np.zeros(num)
                results['cell'][ic]['force'] = np.zeros(num)
            return results

#        update_results(1, self.results[k2], strain, curvature, force, moment, self.cells, self.setOutput)
#        write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive, msg)

        def update_results(i, results, strain, curvature, force, moment, cells, set_out):
            results['section']['strain'][i] = strain
            results['section']['curvature'][i] = curvature
            results['section']['force'][i] = force
            results['section']['moment'][i] = moment
            for ic in set_out:
                results['cell'][ic]['strain'][i] = cells[ic]['e']
                results['cell'][ic]['stress'][i] = cells[ic]['s']
                results['cell'][ic]['force'][i] = cells[ic]['s'] * cells[ic]['A']
            return results

        def write_results(filename_out, step, strain, curvature, force, moment, cells, set_out, msg):
            with open(filename_out, 'a') as fout:
                fout.write('%s,%e,%e,%e,%e' % (step, strain, force, curvature, moment))
                for ic in set_out:
                    fout.write(',%e,%e' % (cells[ic]['e'], cells[ic]['s']))
                fout.write(',%s\n' % (msg))

#        def write_results(filename_out, step, strain, curvature, force, moment, cells, set_out, set_act, msg):
#            with open(filename_out, 'a') as fout:
#                fout.write('%s,%e,%e,%e,%e' % (step, strain, force, curvature, moment))
#                for i in set_out:
#                    if set([i]).issubset(set_act):
#                        fout.write(',%e,%e' % (cells[i]['e'], cells[i]['s']))
#                    else:
#                        fout.write(',,')
#                fout.write(',%s\n' % (msg))

        def update_cells(strain_inc, curvature_inc, cells, set_act):
            for j in set_act:
                cells[j]['e'] = cell_strain(strain_inc, curvature_inc, cells[j])
                cells[j]['s'] = cell_stress(cells[j]['e'], cells[j])[0]
            return cells

        def calc_force_moment(cells, set_act):
            force_int, moment_int = 0.0, 0.0
            for i in set_act:
                force_int += cells[i]['A'] * cells[i]['s']
                moment_int -= cells[i]['A'] * cells[i]['s'] * cells[i]['Y']
            return force_int, moment_int

        def expand_listpair(idlisti, vallisti):
            idlist = list()
            vallist = list()
            for i in range(len(idlisti)):
                if isinstance(idlisti[i], list):
                    ii = np.arange(idlisti[i][0], idlisti[i][1] + 1)
                    vi = np.linspace(vallisti[i][0], vallisti[i][1], ii.size)
                    for j in range(ii.size):
                        idlist.append(ii[j])
                        vallist.append(vi[j])
                else:
                    idlist.append(idlisti[i])
                    vallist.append(vallisti[i])
            return idlist, vallist

        def set_predefined_stress_strain(step_lv2, vid, cells):
            val = step_lv2.get('Force')
            if val is not None:
                ilist, vlist = expand_listpair(vid, val)
                for i, ii in enumerate(ilist):
                    cells[ii]['s'] = vlist[i] / cells[ii]['A']
                    cells[ii]['e'] = my.material.stress2strain(cells[ii]['s'], cells[ii]['M']['fn'], cells[ii]['M']['args'])
                return ilist, vlist
            val = step_lv2.get('Stress')
            if val is not None:
                ilist, vlist = expand_listpair(vid, val)
                for i, ii in enumerate(ilist):
                    cells[ii]['s'] = vlist[i]
                    cells[ii]['e'] = my.material.stress2strain(cells[ii]['s'], cells[ii]['M']['fn'], cells[ii]['M']['args'])
                return ilist, vlist
            val = step_lv2.get('Strain')
            if val is not None:
                ilist, vlist = expand_listpair(vid, val)
                for i, ii in enumerate(ilist):
                    cells[ii]['e'] = vlist[i]
                    cells[ii]['s'] = cells[ii]['M']['fn'](cells[ii]['e'], *cells[ii]['M']['args'])[0]
                return ilist, vlist
            ilist = self._convert_inplist_to_list(vid)
            vlist = None
            return ilist, vlist

        def obj_jac(strain_inc, curvature_inc, cells, set_act, force_ext=0.0, moment_ext=0.0):
            force_int, moment_int = 0.0, 0.0
            EA, EAy, EAy2 = 0.0, 0.0, 0.0
            for i in set_act:
                e = cell_strain(strain_inc, curvature_inc, cells[i])
                s, E = cell_stress(e, cells[i])
                A, y = cells[i]['A'], cells[i]['Y']
                force_int += s * A
                moment_int -= s * A * y
                EA += E * A
                EAy += E * A * y
                EAy2 += E * A * y**2
            obj = np.array([force_ext - force_int, moment_ext - moment_int])
            jac = np.array([[EA, -EAy], [-EAy, EAy2]])
            return obj, jac

        def objc_jacc(strain_inc, curvature_inc, cells, set_act, mcell, mval, force_ext=0.0, moment_ext=0.0):
            obj, jac = obj_jac(strain_inc, curvature_inc, cells, set_act, force_ext, moment_ext)
            objc = np.array([obj[0], obj[1], jac[0, 0] * (mval - cell_strain(strain_inc, curvature_inc, mcell))])
            jacc = np.array([[jac[0, 0], jac[0, 1], jac[0, 0]], [jac[1, 0], jac[1, 1], -jac[0, 0] * mcell['Y']], [jac[0, 0], -jac[0, 0] * mcell['Y'], 0.0]])
            return objc, jacc

        def solve_force_equil(strain_inc, curvature_inc, cells, set_act, tol, force_ext=0.0, moment_ext=0.0):
            obj, jac = obj_jac(strain_inc, curvature_inc, cells, set_act, force_ext, moment_ext)
            obj0 = obj
            while (np.abs(obj[0] / obj0[0]) > tol) and (np.abs(obj0[0]) > tol):
                strain_dinc = obj[0] / jac[0, 0]
                strain_inc += strain_dinc
                obj, jac = obj_jac(strain_inc, curvature_inc, cells, set_act, force_ext, moment_ext)
            return strain_inc

        def solve_force_moment_equil(strain_inc, curvature_inc, cells, set_act, tol, force_ext=0.0, moment_ext=0.0):
            obj, jac = obj_jac(strain_inc, curvature_inc, cells, set_act, force_ext, moment_ext)
            obj0 = obj
            while np.linalg.norm(obj) / np.linalg.norm(obj0) > tol:
                strain_dinc, curvature_dinc = np.linalg.lstsq(jac, obj)[0]
                strain_inc += strain_dinc
                curvature_inc += curvature_dinc
                obj, jac = obj_jac(strain_inc, curvature_inc, cells, set_act, force_ext, moment_ext)
            return strain_inc, curvature_inc

        def solve_force_moment_equil_with_constraint(strain_inc, curvature_inc, cells, set_act, tol, pid, mid, mval, force_ext=0.0, moment_ext=0.0):
            lambda_inc = 0.0
            objc, jacc = objc_jacc(strain_inc, curvature_inc, cells, set_act, cells[mid], mval, force_ext, moment_ext)
            objc0 = objc
            nps = len(pid)
            while np.linalg.norm(objc) / np.linalg.norm(objc0) > tol:
                strain_dinc, curvature_dinc, lambda_dinc = np.linalg.lstsq(jacc, objc)[0]
                strain_inc += strain_dinc
                curvature_inc += curvature_dinc
                lambda_inc += jacc[0, 0] * lambda_dinc / nps
                force_prestress, moment_prestress = 0.0, 0.0
                for i in pid:
                    force_prestress += lambda_inc
                    moment_prestress -= lambda_inc * cells[i]['Y']
                objc, jacc = objc_jacc(strain_inc, curvature_inc, cells, set_act, cells[mid], mval, force_ext - force_prestress, moment_ext - moment_prestress)
            return strain_inc, curvature_inc, lambda_inc

        # ----------------------------------------------------------------------

        # keyword: Step
        if step_no is None:
            steps = self.sa['Step']
        else:
            steps = dict()
            steps[str(step_no)] = self.sa['Step'][str(step_no)]
        #
        msg = ''  # 3333
        if not bool(self.results):
            strain, curvature, force, moment = 0.0, 0.0, 0.0, 0.0
        else:
            lastKey = self.results.keys()[-1]  # sorted(self.results.keys())[-1]
            section = self.results[lastKey]['section']
            strain, curvature, force, moment = section['strain'][-1], section['curvature'][-1], section['force'][-1], section['moment'][-1]
        #
        for k2 in steps:  # sorted(steps.keys()):
            print('-- Analyzing: Step %s' % k2)
            step = steps[k2]
            self.results[k2] = dict()
            #
            setPlus = set(self._convert_inplist_to_list(step.get('Activate')))
            self.set_active = self.set_active | setPlus
            #
            setMinus = set(self._convert_inplist_to_list(step.get('Deactivate')))
            self.set_active = self.set_active - setMinus
            #
            vtype = step['Type'][:4].upper()
            #
            if vtype == 'NORM':
                curvature_max = step['Curvature']['To']
                ndiv = step['Curvature']['NDivisions']
                alloc_results(ndiv + 1, self.results[k2], self.set_output)
                update_results(0, self.results[k2], strain, curvature, force, moment, self.cells, self.set_output)
#                write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive, msg)
                write_results(self.output_filename, k2, strain, curvature, force, moment, self.cells, self.set_output, msg)

                curvature_inc = (curvature_max - curvature) / ndiv
                for i in range(1, ndiv + 1):
                    strain_inc = solve_force_equil(0.0, curvature_inc, self.cells, self.set_active, self.tol, force_ext=force, moment_ext=moment)
                    update_cells(strain_inc, curvature_inc, self.cells, self.set_active)

                    force, moment = calc_force_moment(self.cells, self.set_active)
                    strain += strain_inc
                    curvature += curvature_inc
                    update_results(i, self.results[k2], strain, curvature, force, moment, self.cells, self.set_output)
#                    write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive, msg)
                    write_results(self.output_filename, k2, strain, curvature, force, moment, self.cells, self.set_output, msg)

            elif vtype == 'PRE-':
                ilist, _ = set_predefined_stress_strain(step, step['Id'], self.cells)
                setPlus = set(ilist)
                self.set_active = self.set_active | setPlus

                alloc_results(2, self.results[k2], self.set_output)
                update_results(0, self.results[k2], strain, curvature, force, moment, self.cells, self.set_output)
#                write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive, msg)
                write_results(self.output_filename, k2, strain, curvature, force, moment, self.cells, self.set_output, msg)

                strain_inc, curvature_inc = solve_force_moment_equil(0.0, 0.0, self.cells, self.set_active, self.tol, force_ext=force, moment_ext=moment)
                update_cells(strain_inc, curvature_inc, self.cells, self.set_active)

                force, moment = calc_force_moment(self.cells, self.set_active)
                strain += strain_inc
                curvature += curvature_inc
                update_results(1, self.results[k2], strain, curvature, force, moment, self.cells, self.set_output)
#                write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive, msg)
                write_results(self.output_filename, k2, strain, curvature, force, moment, self.cells, self.set_output, msg)

            elif vtype == 'POST':
                setPlus = set(self._convert_inplist_to_list(step['Id']))

                alloc_results(2, self.results[k2], self.set_output)
                update_results(0, self.results[k2], strain, curvature, force, moment, self.cells, self.set_output)
#                write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive | setPlus, msg)
                write_results(self.output_filename, k2, strain, curvature, force, moment, self.cells, self.set_output, msg)

                ilist, _ = set_predefined_stress_strain(step, step['Id'], self.cells)

                vto = step.get('To')
                if vto is None:
                    force_prestress, moment_prestress = 0.0, 0.0
                    for i in ilist:
                        force_prestress += self.cells[i]['A'] * self.cells[i]['s']
                        moment_prestress -= self.cells[i]['A'] * self.cells[i]['s'] * self.cells[i]['Y']

                    strain_inc, curvature_inc = solve_force_moment_equil(0.0, 0.0, self.cells, self.set_active, self.tol, force_ext=force - force_prestress, moment_ext=moment - moment_prestress)
                    update_cells(strain_inc, curvature_inc, self.cells, self.set_active)
                else:
                    mid = vto['Id']
                    mval = vto['Strain']

                    strain_inc, curvature_inc, lambda_inc = solve_force_moment_equil_with_constraint(0.0, 0.0, self.cells, self.set_active, self.tol, ilist, mid, mval, force_ext=force, moment_ext=moment)
                    update_cells(strain_inc, curvature_inc, self.cells, self.set_active)
                    for ii in ilist:
                        self.cells[ii]['s'] = lambda_inc / self.cells[ii]['A']
                        self.cells[ii]['e'] = my.material.stress2strain(self.cells[ii]['s'], self.cells[ii]['M']['fn'], self.cells[ii]['M']['args'])

                self.set_active = self.set_active | setPlus

                force, moment = calc_force_moment(self.cells, self.set_active)
                strain += strain_inc
                curvature += curvature_inc
                update_results(1, self.results[k2], strain, curvature, force, moment, self.cells, self.set_output)
#                write_results(self.outputFilename, k2, strain, curvature, force, moment, self.cells, self.setOutput, self.setActive, msg)
                write_results(self.output_filename, k2, strain, curvature, force, moment, self.cells, self.set_output, msg)


# -----------------------------------------------------------------------------
# Point in polygon
# -----------------------------------------------------------------------------

# routine for performing the "point in polygon" inclusion test

# Copyright 2001, softSurfer (www.softsurfer.com)
# This code may be freely used and modified for any purpose
# providing that this copyright notice is included with it.
# SoftSurfer makes no warranty for this code, and cannot be held
# liable for any real or imagined damage resulting from its use.
# Users of this code must verify correctness for their application.

# translated to Python by Maciej Kalisiak <mac@dgp.toronto.edu>

# cn_PnPoly(): crossing number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: 0 = outside, 1 = inside
# This code is patterned after [Franklin, 2000]

# def checkPointInsidePolygonByCrossing0(P, V):
#     cn = 0    # the crossing number counter

#     # repeat the first vertex at end
#     V = tuple(V[:]) + (V[0],)

#     # loop through all edges of the polygon
#     for i in range(len(V) - 1):   # edge from V[i] to V[i+1]
#         if ((V[i][1] <= P[1] and V[i + 1][1] > P[1])   # an upward crossing
#                 or (V[i][1] > P[1] and V[i + 1][1] <= P[1])):  # a downward crossing
#             # compute the actual edge-ray intersect x-coordinate
#             vt = (P[1] - V[i][1]) / float(V[i + 1][1] - V[i][1])
#             if P[0] < V[i][0] + vt * (V[i + 1][0] - V[i][0]):  # P[0] < intersect
#                 cn += 1  # a valid crossing of y=P[1] right of P[0]

#     return cn % 2   # 0 if even (out), and 1 if odd (in)

def check_point_in_polygon_by_crossing(p, vs):
    cn = 0    # the crossing number counter
    # repeat the first vertex at end
    vs = tuple(vs[:]) + (vs[0],)
    # loop through all edges of the polygon
    for v1, v2 in zip(vs[:-1], vs[1:]):   # edge from vs[i] to vs[i+1]
        if (v1[1] <= p[1] and v2[1] > p[1]) or (v1[1] > p[1] and v2[1] <= p[1]):  # an upward or downward crossing
            # compute the actual edge-ray intersect x-coordinate
            vt = (p[1] - v1[1]) / float(v2[1] - v1[1])
            if p[0] < v1[0] + vt * (v2[0] - v1[0]):  # P[0] < intersect
                cn += 1  # a valid crossing of y=P[1] right of P[0]

    return bool(cn % 2 == 1)

# ===================================================================

# wn_PnPoly(): winding number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: wn = the winding number (=0 only if P is outside V[])


# def checkPointInsidePolygonByWinding0(P, V):
#     wn = 0   # the winding number counter

#     # repeat the first vertex at end
#     V = tuple(V[:]) + (V[0],)

#     # is_left(): tests if a point is Left|On|Right of an infinite line.
#     def is_left(P0, P1, P2):
#         return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

#     # loop through all edges of the polygon
#     for i in range(len(V) - 1):     # edge from V[i] to V[i+1]
#         if V[i][1] <= P[1]:        # start y <= P[1]
#             if V[i + 1][1] > P[1]:     # an upward crossing
#                 if is_left(V[i], V[i + 1], P) > 0:  # P left of edge
#                     wn += 1           # have a valid up intersect
#         else:                      # start y > P[1] (no test needed)
#             if V[i + 1][1] <= P[1]:    # a downward crossing
#                 if is_left(V[i], V[i + 1], P) < 0:  # P right of edge
#                     wn -= 1           # have a valid down intersect
#     return wn

def check_point_in_polygon_by_winding(p, vs):
    wn = 0   # the winding number counter
    # repeat the first vertex at end
    vs = tuple(vs[:]) + (vs[0],)
    # is_left(): tests if a point is Left|On|Right of an infinite line.

    def is_left(v1, v2, p):
        return (v2[0] - v1[0]) * (p[1] - v1[1]) - (p[0] - v1[0]) * (v2[1] - v1[1])
    # loop through all edges of the polygon
    for v1, v2 in zip(vs[:-1], vs[1:]):     # edge from V[i] to V[i+1]
        if v1[1] <= p[1]:        # start y <= P[1]
            if v2[1] > p[1]:     # an upward crossing
                if is_left(v1, v2, p) > 0:  # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if v2[1] <= p[1]:    # a downward crossing
                if is_left(v1, v2, p) < 0:  # P right of edge
                    wn -= 1           # have a valid down intersect
    return bool(wn != 0)


# -----------------------------------------------------------------------------
# Section constants
# -----------------------------------------------------------------------------


def vertex_to_array(vertices, close=True):
    if close:
        vertices = tuple(vertices[:]) + (vertices[0], )
    vxy = np.array(vertices)
    return vxy[:, 0], vxy[:, 1]


def section_constants_of_polygon(vertices, dxy, axisref=(0.0, 0.0)):

    def vertex_min_max(vs, d):
        vsd = np.array([v[d] for v in vs])
        return np.min(vsd), np.max(vsd), vsd

    dA = dxy[0] * dxy[1]
    #
    vsxmin, vsxmax = vertex_min_max(vertices, 0)[0:2]
    vsymin, vsymax = vertex_min_max(vertices, 1)[0:2]
    pxs = np.arange(vsxmin, vsxmax, dxy[0]) + dxy[0] / 2
    pys = np.arange(vsymin, vsymax, dxy[1]) + dxy[1] / 2
    #
    A, Qx, Qy, Ix, Iy, Ixy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for px in pxs:
        for py in pys:
            if check_point_in_polygon_by_crossing((px, py), vertices):
                pxc = px - axisref[0]
                pyc = py - axisref[1]
                A += dA
                Qx += pyc * dA
                Qy += pxc * dA
                Ix += pyc**2 * dA
                Iy += pxc**2 * dA
                Ixy += pxc * pyc * dA
    return A, Qx, Qy, Ix, Iy, Ixy


def section_constants_of_circle(rxy, r, axisref=(0.0, 0.0)):
    lxy = (rxy[0] - axisref[0], rxy[1] - axisref[1])
    A = np.pi * r**2
    Qx = lxy[1] * A
    Qy = lxy[0] * A
    Ix = np.pi * r**4 / 4 + A * lxy[1]**2
    Iy = np.pi * r**4 / 4 + A * lxy[0]**2
    Ixy = A * lxy[0] * lxy[1]
    return A, Qx, Qy, Ix, Iy, Ixy


def section_constants(input_section, axisref=(0.0, 0.0)):
    if isinstance(input_section, dict):
        sections = input_section
    else:
        with open(input_section, 'r') as f:
            sections = json.load(f)

    A, Qx, Qy, Ix, Iy, Ixy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for key in sections:
        section = sections[key]
        if section['type'] == 'polygon':
            Ai, Qxi, Qyi, Ixi, Iyi, Ixyi = section_constants_of_polygon(section['vertex'], section['dxy'], axisref)
        elif section['type'] == 'circle':
            Ai, Qxi, Qyi, Ixi, Iyi, Ixyi = section_constants_of_circle(section['center'], section['radius'], axisref)
        if section['sign'] == '+':
            sg = 1.0
        else:
            sg = -1.0
        A += sg * Ai
        Qx += sg * Qxi
        Qy += sg * Qyi
        Ix += sg * Ixi
        Iy += sg * Iyi
        Ixy += sg * Ixyi
    return A, Qx, Qy, Ix, Iy, Ixy


def section_center(sectionFilename):
    A, Qx, Qy = section_constants(sectionFilename, axisref=(0.0, 0.0))[0:3]
    return Qy / A, Qx / A
