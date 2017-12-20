# -----------------------------------------------------------------------------
# FRP mechanics
# -----------------------------------------------------------------------------


def HalpinTsaiMethod(Ef, Em, Gf, Gm, nuf, num, vf):
    """
    [El,Et,Glt,nult,nutl] = HalpinTsaiMethod(Ef,Em,Gf,Gm,nuf,num,vf)

    input:
        Ef: Elastic Modulus of fiber
        Em: Elastic Modulus of matrix
        Gf: Shear Modulus of fiber
        Gm: Shear Modulus of matrix
        nuf: Poisson's Ratio of fiber
        num: Poisson's Ratio of matrix
        vf: Volume Ratio of fiber

    output:
        El: longitudinal Young's Modulus of ply
        Et: transverse Young's Modulus of ply
        Glt: Shear Modulus of ply
        nult: Major Poisson's Ratio of ply
        nutl: Minor Poisson's Ratio of ply
    """

    def oneProperty(ksi, Pf, Pm, vf):
        P = (Pm * (Pf + ksi * Pm + ksi * vf * (Pf - Pm))) / (Pf + ksi * Pm - vf * (Pf - Pm))
        return P

    E1 = vf * Ef + (1 - vf) * Em
    nu12 = vf * nuf + (1 - vf) * num
    if vf < 0.65:
        E2 = oneProperty(2.0, Ef, Em, vf)
        G12 = oneProperty(1.0, Ef, Em, vf)
    else:
        E2 = oneProperty(2.0 + 40 * vf**10, Ef, Em, vf)
        G12 = oneProperty(1.0 + 40 * vf**10, Ef, Em, vf)
    G23 = oneProperty(1.0 / (4 - 3 * num), Gf, Gm, vf)

    E3 = E2
    G13 = G12
    nu13 = nu12
    nu23 = vf * nuf + (1 - vf) * num * ((1 + num - nu12 * Em / E1) / (1 - num**2 + num * nu12 * Em / E1))

    return E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
