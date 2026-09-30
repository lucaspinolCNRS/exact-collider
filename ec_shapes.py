"""
ec_shapes.py -- the five bispectrum channels from the dressed legs (arXiv:2609.38167).

Shape:  S = (k1 k2 k3)^2 B_zeta / [(2 pi)^4 Delta_zeta^4],   e_j = k_j/k_t,   beta_j = 2 xi e_j.
Master formulas (Re[] projection; the 2Im of the in-in formula has been converted using (i k_t)^{-(N+1)}):
  N = 2 channels:  S^{(n)} = NN_n e1 e2 e3 Re int_0^inf dxi xi^2 e^{-xi} sum_placements prod_j K_j(2 xi e_j)
      (0)  : WW2 WW2 WW2                  NN_0    = +6 lambda_2 H^2  /(64 pi R^{3/2} Delta)
      (1||): VV WW2 WW2 + 2 perms         NN_1||  = -1 (H/Lambda_2)  /(64 pi R^{3/2} Delta)
      (2)  : WW2 VV VV  + 2 perms         NN_2    = +1 alpha         /(64 pi R^{3/2} Delta)
      (3)  : VV VV VV                     NN_3    = -6 (mu/H)        /(64 pi R^{3/2} Delta)
  N = 0 channel:   S^{(1perp)} = NN_1perp sum_c F_c Re int dxi e^{-xi} PP(2xi e_a) PP(2xi e_b) VV(2xi e_c),
      NN_1perp = +1 (H/Lambda_1)/(64 pi R^{3/2} Delta),   F_c = e_c (e_a^2 + e_b^2 - e_c^2)/(2 e_a e_b).
Reduced shapes  Shat^{(n)} = S^{(n)} Delta_zeta / c_n  with the coupling c_n in {lambda_2 H^2, H/Lambda_2, H/Lambda_1, alpha, mu/H}:
      Shat^{(n)} = sgn_n m_n /(64 pi R^{3/2}) x [geometry] x Re[bracket],   f_NL^{(n)} = (10/9) Shat^{(n)}(k,k,k).
Squeezed limit (kappa = k1/k3 -> 0):  S ~ NN Re sum_b C_b kappa^{1/2 - b nu},  A e^{i delta} = -i (C_+ + C_-^*).
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.special as sp
from ec_kernels import G_b, Scal, Hcal_b

CHANNELS = ('0', '1par', '1perp', '2', '3')
LABELS = {'0': r'$(0)$', '1par': r'$(1\parallel)$', '1perp': r'$(1\perp)$', '2': r'$(2)$', '3': r'$(3)$'}
# sign, multiplicity m_n, coupling name, weak-mixing power of lam of the clock coefficient
PREF = {'0': (+1, 6, r'$(H/\Lambda_\star)^2$', 2), '1par': (-1, 1, r'$H/\Lambda_2$', 1), '1perp': (+1, 1, r'$H/\Lambda_1$', 1),
        '2': (+1, 1, r'$\alpha$', 2), '3': (-1, 6, r'$\mu/H$', 3)}


def reduced_prefactor(channel, R):
    """sgn_n m_n / (64 pi R^{3/2}):  S^{(n)} Delta/c_n = this x geometry x Re[bracket]."""
    sgn, m, _, _ = PREF[channel]
    return sgn * m / (64 * np.pi * R ** 1.5)


# ----------------------------------------------------------------------------------------------
# 1. xi-quadrature: Gauss-Legendre panels in s = ln xi  (the soft tails make the integrand ~ xi^{1/2 +- i(...)mu}
#    at xi -> 0, which is non-polynomial: a plain Gauss-Laguerre rule would converge only algebraically)
# ----------------------------------------------------------------------------------------------
class XiRule:
    def __init__(self, N, xi_min, xi_max=55.0, width=0.25, nodes=8):
        xg, wg = leggauss(nodes)
        # panels of width `width` in s = ln xi above xi = 1e-8; twice as wide below, where only the pure
        # soft-tail power laws are left (this region matters only for the N = 0 channel)
        s_lo, s_split, s_hi = np.log(xi_min), np.log(1e-8), np.log(xi_max)
        if s_lo < s_split:
            edges = np.concatenate([np.arange(s_lo, s_split, 2 * width), np.arange(s_split, s_hi + width, width)])
        else:
            edges = np.arange(s_lo, s_hi + width, width)
        lo, hi = edges[:-1], edges[1:]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        s = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
        ws = (half[:, None] * wg[None, :]).ravel()
        self.N = N
        self.xi = np.exp(s)
        self.w = ws * self.xi ** (N + 1) * np.exp(-self.xi)     # xi^N e^{-xi} dxi = xi^{N+1} e^{-xi} ds
        self.ws = ws                                            # bare ds weights (for the complex-power moments J_b)

    def moment_weights(self, power):
        """weights for  int dxi xi^{power} e^{-xi} (...)  with a complex power (used by the clock integrals J_b)."""
        return self.ws * self.xi ** (power + 1) * np.exp(-self.xi)


# xi_min: the N = 2 integrands behave as xi^{1/2 + i(...)} at small xi, and a cut at 1e-8 leaves a relative
# truncation error of a few 1e-10 -- invisible in every shape of the paper (lambda <= 7) but exactly the size of
# the strong-mixing residue A/C_+ ~ e^{-pi lambda} at lambda ~ 12 (see STRONG-MIXING-NOTES.md); 1e-12 costs 40%
# more nodes and brings it to 1e-16.  The arbitrary-precision pipeline (hp_clock.py) cuts at 1e-18.
RULE_N2 = XiRule(2, xi_min=1e-12)
RULE_N0 = XiRule(0, xi_min=1e-18)
RULE_N2_FINE = XiRule(2, xi_min=1e-8, width=0.125, nodes=12)
RULE_N0_FINE = XiRule(0, xi_min=1e-18, width=0.125, nodes=12)


# ----------------------------------------------------------------------------------------------
# 2. Brackets and shapes for arrays of configurations
# ----------------------------------------------------------------------------------------------
def _e(k1, k2, k3):
    k1, k2, k3 = (np.atleast_1d(np.asarray(k, float)) for k in (k1, k2, k3))
    kt = k1 + k2 + k3
    return k1 / kt, k2 / kt, k3 / kt


def brackets(table, k1, k2, k3, rule2=RULE_N2, rule0=RULE_N0, channels=CHANNELS):
    """Complex brackets (before Re) of the requested channels for arrays of triangles (k1,k2,k3).
    Returns dict: '0','1par','2','3' -> (nconf,);  '1perp' -> (3, nconf) one bracket per sigma placement c."""
    e = _e(k1, k2, k3)
    out = {}
    n2 = [ch for ch in channels if ch != '1perp']
    if n2:                                              # N = 2 channels share the legs WW2, VV on the N = 2 rule
        W2 = [table.leg('W2', 2 * rule2.xi[:, None] * ej[None, :]) for ej in e] if any(ch != '3' for ch in n2) else None
        V = [table.leg('V', 2 * rule2.xi[:, None] * ej[None, :]) for ej in e] if any(ch != '0' for ch in n2) else None
        w = rule2.w[:, None]
        if '0' in n2: out['0'] = np.sum(w * W2[0] * W2[1] * W2[2], axis=0)
        if '1par' in n2: out['1par'] = np.sum(w * (V[0] * W2[1] * W2[2] + W2[0] * V[1] * W2[2] + W2[0] * W2[1] * V[2]), axis=0)
        if '2' in n2: out['2'] = np.sum(w * (W2[0] * V[1] * V[2] + V[0] * W2[1] * V[2] + V[0] * V[1] * W2[2]), axis=0)
        if '3' in n2: out['3'] = np.sum(w * V[0] * V[1] * V[2], axis=0)
    if '1perp' in channels:                             # N = 0 channel: legs PP, VV on the N = 0 rule
        P = [table.leg('P', 2 * rule0.xi[:, None] * ej[None, :]) for ej in e]
        V0 = [table.leg('V', 2 * rule0.xi[:, None] * ej[None, :]) for ej in e]
        w0 = rule0.w[:, None]
        out['1perp'] = np.array([np.sum(w0 * P[1] * P[2] * V0[0], axis=0),
                                 np.sum(w0 * P[0] * P[2] * V0[1], axis=0),
                                 np.sum(w0 * P[0] * P[1] * V0[2], axis=0)])
    return out


def F_factors(k1, k2, k3):
    """F_c = e_c (e_a^2 + e_b^2 - e_c^2)/(2 e_a e_b) for c = 1,2,3  -> (3, nconf).
    The bracket is evaluated as (e_b - e_c)(e_b + e_c) + e_a^2 with e_a the smaller of the two other legs: in a
    squeezed isosceles triangle (k1 << k2 = k3) the naive form loses e_1^2 against e_3^2 - e_2^2 in floating point."""
    e = _e(k1, k2, k3)
    out = []
    for c in range(3):
        a, b = [j for j in range(3) if j != c]
        ea, eb = np.minimum(e[a], e[b]), np.maximum(e[a], e[b])
        out.append(e[c] * ((eb - e[c]) * (eb + e[c]) + ea ** 2) / (2 * ea * eb))
    return np.array(out)


def reduced_shapes(table, k1, k2, k3, channels=CHANNELS, br=None, **kw):
    """Shat^{(n)}(k1,k2,k3) = S^{(n)} Delta_zeta / c_n for the requested channels (dict of real arrays)."""
    if br is None:
        br = brackets(table, k1, k2, k3, channels=channels, **kw)
    e1, e2, e3 = _e(k1, k2, k3)
    out = {}
    for ch in channels:
        pref = reduced_prefactor(ch, table.R)
        if ch == '1perp':
            out[ch] = pref * np.sum(F_factors(k1, k2, k3) * br['1perp'].real, axis=0)
        else:
            out[ch] = pref * e1 * e2 * e3 * br[ch].real
    return out


def fnl_equilateral(table, channels=CHANNELS, **kw):
    """f_NL^{(n)} per unit coupling = (10/9) Shat^{(n)}(k,k,k)."""
    S = reduced_shapes(table, [1.0], [1.0], [1.0], channels, **kw)
    return {ch: (10.0 / 9.0) * S[ch][0] for ch in channels}


# ----------------------------------------------------------------------------------------------
# 3. Triangle grid and cosine (Babich-Creminelli-Zaldarriaga with uniform weight, as in arXiv:2607.15251)
# ----------------------------------------------------------------------------------------------
def triangle_grid(dx=0.02, x1_min=1e-3):
    """Cell-centred grid on x1 = k1/k3 <= x2 = k2/k3 <= 1, x1 + x2 >= 1, x1 >= x1_min  (k3 = 1)."""
    x = np.arange(dx / 2, 1, dx)
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    m = (X1 <= X2) & (X1 + X2 >= 1) & (X1 >= x1_min)
    return X1[m], X2[m]


def cosine(S1, S2):
    return np.sum(S1 * S2) / np.sqrt(np.sum(S1 ** 2) * np.sum(S2 ** 2))


def S_equilateral_template(k1, k2, k3):
    e1, e2, e3 = _e(k1, k2, k3)
    return e1 * e2 * e3


# ----------------------------------------------------------------------------------------------
# 4. Squeezed limit: clock coefficients from the same tables
# ----------------------------------------------------------------------------------------------
def J_moment(table, name1, name2, power, rule):
    """J = int dxi xi^{power} e^{-xi} K1(xi) K2(xi)   (hard legs at beta = 2 xi e = xi since e_{2,3} -> 1/2)."""
    K1 = table.leg(name1, rule.xi); K2 = table.leg(name2, rule.xi)
    return np.sum(rule.moment_weights(power) * K1 * K2)


def clock_coefficients(table, channels=CHANNELS, rule2=RULE_N2, rule0=RULE_N0):
    """C_b^{(n)} (b = +1, -1) such that  S^{(n)} ~ NN_n Re sum_b C_b kappa^{1/2 - b nu}:
       (0)   : (1/2)   Wcal_b J_b[WW2,WW2]
       (1||) : (1/2lam) Wcal_b J_b[WW2,WW2] + Wcal_b J_b[VV,WW2]
       (1perp): (2/lam) Wcal_b J^0_b[PP,PP],   J^0_b = int xi^{-1/2-b nu} e^{-xi} PP^2
       (2)   : (1/lam) Wcal_b J_b[WW2,VV] + (1/2) Wcal_b J_b[VV,VV]
       (3)   : (1/2lam) Wcal_b J_b[VV,VV]
    with J2_b[K,K'] = int xi^{3/2 - b nu} e^{-xi} K K'  (at N=2; N=0 for the gradient vertex)
    and Wcal_b = sum_a e^{a pi lam/2} r_a Wcal_b^{-a}."""
    lam, nu = table.lam, table.nu
    out = {ch: {} for ch in channels}
    for b in (+1, -1):
        p2 = 1.5 - b * nu
        p0 = -0.5 - b * nu
        Wb = table.Wcal_b[b]
        JWW = J_moment(table, 'W2', 'W2', p2, rule2)
        JVW = J_moment(table, 'V', 'W2', p2, rule2)
        JVV = J_moment(table, 'V', 'V', p2, rule2)
        JPP = J_moment(table, 'P', 'P', p0, rule0)
        vals = {'0': 0.5 * Wb * JWW,
                '1par': Wb * (JWW / (2 * lam) + JVW),
                '1perp': (2 / lam) * Wb * JPP,
                '2': Wb * (JVW / lam + 0.5 * JVV),
                '3': Wb * JVV / (2 * lam)}
        for ch in channels:
            out[ch][b] = vals[ch]
    return out


def clock_amplitude(C):
    """A e^{i delta} = -i (C_+ + C_-^*) for one channel's {+1: C_+, -1: C_-}."""
    return -1j * (C[+1] + np.conj(C[-1]))


def squeezed_reduced(table, channel, kappa):
    """Reduced squeezed shape  Shat ~ pref x Re[(C_+ + C_-^*) kappa^{1/2 - nu}]  (heavy field, nu = i mu)."""
    C = clock_coefficients(table, (channel,))[channel]
    A = C[+1] + np.conj(C[-1])
    return reduced_prefactor(channel, table.R) * np.real(A * kappa ** (0.5 - table.nu))


def weak_mixing_clock(channel, lam, nu, Tb=None):
    """Closed-form lam -> 0 limits of C_b, the rows of the weak-mixing table.
    Tb = {b: T_b} with T_b = J2_b[v, v] the triple-exchange moment must be supplied for
    channel '3', the one entry not in closed form."""
    out = {}
    for b in (+1, -1):
        x = b * nu
        GS = G_b(nu, b) * Scal(x)
        if channel == '0':
            out[b] = 2 * lam ** 2 * GS * sp.gamma(2.5 - x)
        elif channel == '1par':
            out[b] = 2 * lam * GS * sp.gamma(2.5 - x)
        elif channel == '1perp':
            out[b] = 8 * lam * GS * (sp.gamma(0.5 - x) + sp.gamma(1.5 - x) + 0.25 * sp.gamma(2.5 - x))
        elif channel == '2':
            out[b] = 8 * lam ** 2 * GS * Hcal_b(nu, b)
        elif channel == '3':
            out[b] = 8 * lam ** 3 * GS * Tb[b]
        elif channel == 'LI':          # boost-invariant combination Lambda_2^{-1} = -Lambda_1^{-1}: C^{LI} = C^{(1||)} + C^{(1perp)}
            out[b] = 4 * lam * GS * sp.gamma(3.5 - x) / (0.5 - x)
    return out


def ahm_clock_combination(mu):
    """Arkani-Hamed--Maldacena anchor, in the leg grouping used here:
    (C^{LI}_+ + C^{LI*}_-)/lam  ->  -4 * 4^{i mu} Gamma(i mu) Gamma(1/2 - i mu) (5/2 - i mu)(3/2 - i mu)(1 - i sinh pi mu) sqrt(pi)/(2 cosh pi mu)
    as lam -> 0."""
    m = mu
    rhs = (4.0 ** (1j * m) * sp.gamma(1j * m) * sp.gamma(0.5 - 1j * m) * (2.5 - 1j * m) * (1.5 - 1j * m)
           * (1 - 1j * np.sinh(np.pi * m)) * np.sqrt(np.pi) / (2 * np.cosh(np.pi * m)))
    return -4 * rhs


def li_clock(C):
    """C^{LI}_b = C^{(1||)}_b + C^{(1perp)}_b from a clock_coefficients() dict."""
    return {b: C['1par'][b] + C['1perp'][b] for b in (+1, -1)}


# ----------------------------------------------------------------------------------------------
# 5. Fits of the squeezed shape
# ----------------------------------------------------------------------------------------------
def squeezed_fit(kappa, S, mu, n_clock=3, n_bg=3):
    """Least-squares fit of kappa^{-1/2} S to  sum_{m<n_clock} kappa^m [a_m cos(mu ln k) + b_m sin(mu ln k)]
    + sum_{m=1..n_bg} c_m kappa^{m-1/2}.  Returns (A = a_0 + i b_0, rms residual relative to rms of kappa^{-1/2} S)."""
    y = S / np.sqrt(kappa)
    lk = np.log(kappa)
    cols = []
    for m in range(n_clock):
        cols += [kappa ** m * np.cos(mu * lk), kappa ** m * np.sin(mu * lk)]
    for m in range(1, n_bg + 1):
        cols.append(kappa ** (m - 0.5))
    M = np.array(cols).T
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ coef
    return coef[0] + 1j * coef[1], np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(y ** 2))


def richardson_lam2(lams, vals):
    """Extrapolate f(lam) = f0 + c1 lam^2 + c2 lam^4 + ... to lam = 0 from len(lams) samples (exact polynomial fit in lam^2)."""
    lams = np.asarray(lams, float); vals = np.asarray(vals)
    M = np.vander(lams ** 2, len(lams), increasing=True)
    return np.linalg.solve(M, vals)[0]


# ----------------------------------------------------------------------------------------------
# 6. Interface to binned shape data (Philcox, github.com/oliverphilcox/Binned-Bispectrum-Shape):
#    S_theory(x, y, lam, mu_eff, channel) on the triangle x = k1/k3, y = k2/k3, |1-y| <= x <= y <= 1, normalised to 1 at (1,1)
# ----------------------------------------------------------------------------------------------
_TABLE_CACHE = {}


def get_table(lam, mu_eff, beta_grid=None, per_efold=None):
    """KernelTable for (lam, mu_eff), built once and cached in memory (0.1-0.3 s to build)."""
    from ec_kernels import KernelTable, default_beta_grid
    key = (float(lam), float(mu_eff), per_efold)
    if key not in _TABLE_CACHE:
        if beta_grid is None:
            beta_grid = default_beta_grid() if per_efold is None else default_beta_grid(per_efold=per_efold)
        _TABLE_CACHE[key] = KernelTable(lam, 1j * mu_eff, beta_grid)
    return _TABLE_CACHE[key]


def S11_equilateral(lam, mu_eff, channel, table=None):
    """Equilateral normalisation Shat^{(n)}(1,1) = S^{(n)}(k,k,k) Delta_zeta / c_n, in units of the dimensionless coupling
    c_n in {lambda_2 H^2, H/Lambda_2, H/Lambda_1, alpha, mu/H}  (f_NL = (10/9) x this)."""
    table = table or get_table(lam, mu_eff)
    return reduced_shapes(table, [1.0], [1.0], [1.0], channels=(channel,))[channel][0]


def S_theory(x, y, lam, mu_eff, channel, table=None, return_norm=False):
    """Shape S^{(channel)} at triangle coordinates x = k1/k3, y = k2/k3 (arrays; |1-y| <= x <= y <= 1 assumed; k3 = 1),
    NORMALISED TO UNITY AT x = y = 1.  channel in ('0', '1par', '1perp', '2', '3').
    With return_norm=True also returns S11 = Shat(1,1), the equilateral value in units of c_n/Delta_zeta, so that the
    physical shape is  S = (c_n/Delta_zeta) * S11 * S_theory  and  f_NL^{eq} = (10/9) (c_n/Delta_zeta) S11."""
    table = table or get_table(lam, mu_eff)
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    S = reduced_shapes(table, x, y, np.ones_like(x), channels=(channel,))[channel]
    S11 = S11_equilateral(lam, mu_eff, channel, table)
    return (S / S11, S11) if return_norm else S / S11
