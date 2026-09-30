"""
ec_kernels.py -- the master kernel of the Exact-collider project and everything that descends from it.

CONVENTIONS (those of the companion paper, arXiv:2609.38167)
  nu = nu_eff = sqrt(9/4 - m_eff^2/H^2) = i*mu_eff  (heavy field; general complex nu accepted)
  lam = rho/H,   z_a = i a lam/2,   a = +/- is the channel index.
  omega_a(u) = u^{z_a-1}(1+u)^{-z_a} 2F1(1/2-nu, 1/2+nu; 1+2z_a; -u) / Gamma(z_a),   omega_{-a} = omega_a^*
  W_n^a(beta) = int_0^inf du omega_a(u) (1+2u)^n/(1+u) e^{-beta u}   (n = 0,1,2)
  V^a(beta)   = int_0^inf du omega_a(u)  u  e^{-beta u}  = (W_2^a - W_0^a)/4     (exact algebra)
  P_a(beta)   = W_0^a + (beta/2) W_1^a
  Pfaff variable t = u/(1+u):
     omega_a(u) du = t^{z_a-1} (1-t)^{-1/2-nu} 2F1(1/2-nu, 1/2-nu+2z_a; 1+2z_a; t) dt / Gamma(z_a)
  Dressed (channel-summed) legs, with the Boltzmann-weighted boundary value e^{a pi lam/2} r_a :
     WW_n = sum_a e^{a pi lam/2} r_a W_n^{-a},   PP = WW_0 + (beta/2) WW_1,
     VV = (4/lam) sum_a e^{a pi lam/2} r_a V^{-a}
Only the a = + channel is ever integrated; a = - is its complex conjugate.

Two engines that share no quadrature code:
  * fast_kernels : float64, Gauss-Legendre panels, log-spaced in t (left) and in u (right), own 2F1.
  * gold_kernels : mpmath, tanh-sinh (mp.quad) in the variables s=-ln t and s=-ln(1-t), mp.hyp2f1.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.special as sp
from scipy.interpolate import CubicSpline
import mpmath as mp

SQRTPI = np.sqrt(np.pi)

# ----------------------------------------------------------------------------------------------
# 1. Closed forms: boundary value r_a, power-spectrum factor R, soft-tail coefficients Wcal
# ----------------------------------------------------------------------------------------------
def r_plus(lam, nu):
    """r_+ in closed form: Gamma(1/2+z)Gamma(3/4-nu/2)Gamma(3/4+nu/2)/[sqrt(pi) Gamma(3/4-nu/2+z)Gamma(3/4+nu/2+z)], z=i lam/2."""
    z = 0.5j * lam
    g = sp.gamma
    return g(0.5 + z) * g(0.75 - nu / 2) * g(0.75 + nu / 2) / (SQRTPI * g(0.75 - nu / 2 + z) * g(0.75 + nu / 2 + z))


def R_power(lam, nu):
    """R = (1/2) sum_a e^{a pi lam/2} |r_a|^2 = cosh(pi lam/2) |r_+|^2."""
    return np.cosh(np.pi * lam / 2) * abs(r_plus(lam, nu)) ** 2


def r_weighted(lam, nu):
    """The two Boltzmann-weighted boundary values (e^{pi lam/2} r_+, e^{-pi lam/2} r_-), r_- = r_+^*, which dress
    the external legs (the paper writes the factor out; there is no separate symbol)."""
    rp = r_plus(lam, nu)
    return np.exp(np.pi * lam / 2) * rp, np.exp(-np.pi * lam / 2) * np.conj(rp)


def Wcal(lam, nu, a, b):
    """Soft-tail coefficient Wcal^a_b = Gamma(1+2z_a)Gamma(2 b nu)/[Gamma(z_a)Gamma(1/2+b nu+2z_a)]."""
    z = 0.5j * a * lam
    g = sp.gamma
    return g(1 + 2 * z) * g(2 * b * nu) / (g(z) * g(0.5 + b * nu + 2 * z))


def Wcal_dressed(lam, nu, b):
    """Channel-summed tail coefficient  Wcal_b = sum_a e^{a pi lam/2} r_a Wcal_b^{-a}."""
    rp, rm = r_weighted(lam, nu)
    return rp * Wcal(lam, nu, -1, b) + rm * Wcal(lam, nu, +1, b)


# ----------------------------------------------------------------------------------------------
# 2. Fast engine: 2F1 with complex parameters on (0,1), Gauss-Legendre panels
# ----------------------------------------------------------------------------------------------
def _hyp2f1_series(a, b, c, x, nmax=400):
    """Direct power series of 2F1(a,b;c;x), complex parameters, |x| <= 1/2 (vectorised in x)."""
    x = np.asarray(x, dtype=complex)
    term = np.ones_like(x)
    total = np.ones_like(x)
    for n in range(nmax):
        term = term * ((a + n) * (b + n) / ((c + n) * (n + 1.0))) * x
        total = total + term
        if np.all(np.abs(term) <= 1e-18 * np.abs(total)):
            break
    return total


def hyp2f1_01(a, b, c, x, log1mx):
    """2F1(a,b;c;x) for real 0<x<1: series for x<=1/2, Gauss connection formula (DLMF 15.8.4) for x>1/2.
    log1mx = ln(1-x) must be supplied accurately (x may be 1-1e-14).  scipy.special.hyp2f1 does NOT accept
    complex parameters, which is why this function exists.  Requires c-a-b not an integer."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=complex)
    left = x <= 0.5
    out[left] = _hyp2f1_series(a, b, c, x[left])
    y = np.exp(log1mx[~left])                       # 1 - x, accurate
    s = c - a - b
    g = sp.gamma
    A1 = g(c) * g(s) / (g(c - a) * g(c - b))
    A2 = g(c) * g(-s) / (g(a) * g(b))
    out[~left] = (A1 * _hyp2f1_series(a, b, a + b - c + 1, y)
                  + A2 * np.exp(s * log1mx[~left]) * _hyp2f1_series(c - a, c - b, s + 1, y))
    return out


class FastNodes:
    """Beta-independent quadrature nodes for the Pfaff integral, and the beta-independent factors at them.
    Left region  t in [t_min, 1/2]: panels uniform in ln t (this is where the u->0 continuation lives).
    Right region u in [1, u_max]  : panels uniform in ln u (this is where the clock tail u^{-3/2+-nu} lives).
    """
    def __init__(self, lam, nu, beta_min=1e-13, t_min=1e-18, nodes_per_panel=12, width_left=0.5, width_right=0.5,
                 u_cut=50.0):
        self.lam, self.nu = lam, nu
        z = 0.5j * lam
        self.z = z
        xg, wg = leggauss(nodes_per_panel)
        # ---- left: s = ln t, from ln t_min to ln(1/2)
        s_edges = np.arange(np.log(0.5), np.log(t_min) - 1e-12, -width_left)[::-1]
        s_edges[0] = np.log(t_min)                  # pin the lowest edge at ln t_min (first panel may be wider)
        sL, wL = _panel_nodes(s_edges, xg, wg)
        tL = np.exp(sL); uL = tL / (1.0 - tL)
        log_tL = sL; log1mtL = np.log1p(-tL)
        AL = wL * np.exp(z * sL)                    # dt = t ds  ->  w t^{z-1} t = w t^z
        # ---- right: s = ln u, from 0 to ln u_max
        u_max = u_cut / beta_min
        s_edges = np.arange(0.0, np.log(u_max) + width_right, width_right)
        sR, wR = _panel_nodes(s_edges, xg, wg)
        uR = np.exp(sR); tR = uR / (1.0 + uR)
        log1mtR = -np.log1p(uR); log_tR = sR + log1mtR
        AR = wR * uR / (1.0 + uR) ** 2 * np.exp((z - 1.0) * log_tR)   # dt = u/(1+u)^2 ds
        # ---- assemble
        self.nL = len(tL)
        self.t = np.concatenate([tL, tR]); self.u = np.concatenate([uL, uR])
        self.log_t = np.concatenate([log_tL, log_tR]); self.log1mt = np.concatenate([log1mtL, log1mtR])
        self.A = np.concatenate([AL, AR]) / sp.gamma(z)          # measure factor  w t^{z-1} dt/ds / Gamma(z)
        # beta-independent regular factor  B = (1-t)^{-1/2-nu} 2F1(1/2-nu, 1/2-nu+2z; 1+2z; t)
        a, b, c = 0.5 - nu, 0.5 - nu + 2 * z, 1 + 2 * z
        F = hyp2f1_01(a, b, c, self.t, self.log1mt)
        self.B = np.exp((-0.5 - nu) * self.log1mt) * F
        # rational weights  chi_n = (1+2u)^n/(1+u) = (1+t)^n (1-t)^{1-n};  chi_V = u
        omt = np.exp(self.log1mt)
        self.chi = [(1 + self.t) ** n * omt ** (1 - n) for n in range(3)]
        # analytic piece of the u->0 continuation (M=1 Taylor term of the regular factor, which is 1 at t=0):
        #   int_0^{1/2} t^{z-1} dt = (1/2)^z / z ;  the subtracted "1" is applied on the left nodes only
        self.const = 0.5 ** z / z / sp.gamma(z) - np.sum(self.A[:self.nL])
        self.left_mask = np.arange(len(self.t)) < self.nL

    def kernels(self, beta):
        """W_0, W_1, W_2, V of the a=+ channel at an array of beta >= 0 (beta = 0 allowed for W_0 and W_1)."""
        beta = np.atleast_1d(np.asarray(beta, dtype=float))
        E = np.exp(-np.outer(self.u, beta))                 # (nodes, beta), real
        AB = self.A * self.B
        weights = np.array([AB * self.chi[0], AB * self.chi[1], AB * self.chi[2], AB * self.u])   # (4, nodes)
        M = np.vstack([weights.real, weights.imag]) @ E     # one real matrix product (8, beta)
        K = M[:4] + 1j * M[4:]
        return {'W0': K[0] + self.const, 'W1': K[1] + self.const, 'W2': K[2] + self.const, 'V': K[3]}


def _panel_nodes(edges, xg, wg):
    lo, hi = edges[:-1], edges[1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    nodes = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
    weights = (half[:, None] * wg[None, :]).ravel()
    return nodes, weights


def fast_kernels(lam, nu, beta, **kw):
    """Convenience wrapper: build the nodes for (lam, nu) and return the four a=+ kernels at beta."""
    return FastNodes(lam, nu, **kw).kernels(beta)


# ----------------------------------------------------------------------------------------------
# 3. Gold engine (mpmath).  Shares no quadrature code with the fast engine.
# ----------------------------------------------------------------------------------------------
def gold_kernels(beta, lam, nu, dps=20, which=('W0', 'W1', 'W2', 'V'), s_max_left=48.0, verbose=False):
    """W_n^+(beta), V^+(beta) by mp.quad (tanh-sinh) on the Pfaff form.
    Left  : t = e^{-s}, s in [ln 2, s_max]:  int t^{z-1}[phi(t)-phi(0)] dt = int e^{-z s} [phi(e^{-s}) - phi(0)] ds,
            plus the analytic continuation term phi(0) (1/2)^z / z.
    Right : 1-t = e^{-s}, s in [ln 2, s_r]:   dt = e^{-s} ds,  e^{-beta u} = exp(-beta (e^s - 1)).
    Returns a dict of mpc values (and the mp.quad error estimates in key 'err')."""
    mp.mp.dps = dps
    beta = mp.mpf(beta); lam = mp.mpf(lam); nu = mp.mpmathify(nu)
    z = mp.mpc(0, lam) / 2
    a, b, c = mp.mpf(1) / 2 - nu, mp.mpf(1) / 2 - nu + 2 * z, 1 + 2 * z
    memo = {}                                   # the four kernels share the same 2F1 values at the same nodes

    def F(t):
        v = memo.get(t)
        if v is None:
            v = mp.hyp2f1(a, b, c, t); memo[t] = v
        return v
    chi = {'W0': lambda t: (1 - t), 'W1': lambda t: (1 + t), 'W2': lambda t: (1 + t) ** 2 / (1 - t),
           'V': lambda t: t / (1 - t)}
    phi0 = {'W0': 1, 'W1': 1, 'W2': 1, 'V': 0}
    out, err = {}, {}
    # right-hand cut: exp(-beta(e^s-1)) e^{-s/2} ~ 1e-25 ; for beta=0 the (1-t)^{1/2-nu} decay only -> cap at s = 60
    s_r = float(mp.log(60 / beta + 1)) if beta > 0 else 60.0
    pts_left = [mp.log(2)] + [mp.mpf(k) for k in range(1, int(s_max_left) + 1)]
    pts_right = [mp.log(2)] + [mp.log(2) + mp.mpf(k) / 2 for k in range(1, int(2 * (s_r - float(mp.log(2)))) + 2)]
    for key in which:
        ch, p0 = chi[key], phi0[key]

        def f_left(s, ch=ch, p0=p0):
            t = mp.exp(-s)
            g = mp.exp((-mp.mpf(1) / 2 - nu) * mp.log1p(-t)) * F(t) * ch(t) * mp.exp(-beta * t / (1 - t))
            return mp.exp(-z * s) * (g - p0)

        def f_right(s, ch=ch):
            omt = mp.exp(-s); t = 1 - omt
            return (mp.exp((z - 1) * mp.log(t)) * mp.exp((-mp.mpf(1) / 2 - nu) * (-s)) * F(t) * ch(t)
                    * mp.exp(-beta * (mp.exp(s) - 1)) * omt)

        IL, eL = mp.quad(f_left, pts_left, error=True)
        IR, eR = mp.quad(f_right, pts_right, error=True)
        val = (IL + IR + p0 * mp.mpf(1) / 2 ** z / z) / mp.gamma(z)
        out[key] = val; err[key] = (eL + eR) / abs(mp.gamma(z))
        if verbose:
            print(key, val, 'quad error estimate', mp.nstr(err[key], 3))
    out['err'] = err
    return out


def gold_W0_direct_u(beta, lam, nu, dps=20):
    """W_0^+(beta) directly in the u variable (NO Pfaff transformation): an independent check of the Pfaff-transformed evaluation.
    u = e^{-s} on (0,1) with the same M=1 continuation, u = e^{s} on (1, inf)."""
    mp.mp.dps = dps
    beta = mp.mpf(beta); lam = mp.mpf(lam); nu = mp.mpmathify(nu)
    z = mp.mpc(0, lam) / 2
    om_reg = lambda u: (1 + u) ** (-z) * mp.hyp2f1(mp.mpf(1) / 2 - nu, mp.mpf(1) / 2 + nu, 1 + 2 * z, -u) / (1 + u) * mp.exp(-beta * u)
    f_small = lambda s: mp.exp(-z * s) * (om_reg(mp.exp(-s)) - 1)          # u = e^{-s}, du = -u ds
    f_large = lambda s: mp.exp(z * s) * om_reg(mp.exp(s))                  # u = e^{s},  du = u ds
    s_r = float(mp.log(60 / beta + 1)) if beta > 0 else 60.0
    I1 = mp.quad(f_small, [mp.mpf(k) for k in range(0, 49)])
    I2 = mp.quad(f_large, [mp.mpf(k) / 2 for k in range(0, int(2 * s_r) + 2)])
    return (I1 + I2 + 1 / z) / mp.gamma(z)


# ----------------------------------------------------------------------------------------------
# 4. Weak-mixing closed forms:  V_1, V_2, varrho_1, S(x), G_b, H_b, J2[V_1,V_1]
#    V^a(beta) = z_a V_1(beta) + z_a^2 V_2(beta) + O(z_a^3),  z_a = i a lam/2
# ----------------------------------------------------------------------------------------------
def varrho1(nu):
    """varrho_1 = pi/2 + (i/2)[psi(1/2) - psi(3/4-nu/2) - psi(3/4+nu/2)];  e^{a pi lam/2} r_a = 1 + a lam varrho_1 + ..."""
    d = sp.digamma
    return np.pi / 2 + 0.5j * (d(0.5) - d(0.75 - nu / 2) - d(0.75 + nu / 2))


def Scal(x):
    """The scalar S(x) = psi(1/2+x) + gamma/2 - i varrho_1 = -(pi/2) cot(pi/4 + pi x/2) - i pi/2."""
    return -(np.pi / 2) / np.tan(np.pi / 4 + np.pi * x / 2) - 0.5j * np.pi


def G_b(nu, b):
    """G_b = Gamma(2 b nu)/Gamma(1/2 + b nu): tail coefficient of the free rotated mode V_1."""
    return sp.gamma(2 * b * nu) / sp.gamma(0.5 + b * nu)


def Hcal_b(nu, b):
    """Hcal_b = J2_b[v, 1] = int xi^{3/2-b nu} e^{-xi} v(xi),  v = -(1/2) V_2 - i varrho_1 V_1,  in closed form
    written with A = 1/2 - b nu."""
    A = 0.5 - b * nu
    return ((2 + A) * sp.gamma(A) / (2 * (1 + A))
            + 2 * sp.gamma(2 * A) / ((1 + A) * sp.gamma(A)) * (Scal(b * nu) - np.pi * np.tan(np.pi * b * nu)))


def J2_V1V1_b(nu, b):
    """J2_b[V_1, V_1] = int xi^{3/2-b nu} e^{-xi} V_1(xi)^2 = 2^{2s-5}/pi Gamma((s-1)/2)^2 Gamma((s-1)/2+nu)Gamma((s-1)/2-nu)/Gamma(s-1),
    s = 5/2 - b nu."""
    s = 2.5 - b * nu
    h = (s - 1) / 2
    return 2.0 ** (2 * s - 5) / np.pi * sp.gamma(h) ** 2 * sp.gamma(h + nu) * sp.gamma(h - nu) / sp.gamma(s - 1)


def V1_closed(beta, nu, dps=30):
    """V_1(beta) = e^{beta/2} K_nu(beta/2)/sqrt(pi beta), the free rotated mode, coefficient of z_a in the sigma kernel
    computed with mpmath (complex order allowed)."""
    mp.mp.dps = dps
    beta = mp.mpf(beta); nu = mp.mpmathify(nu)
    return complex(mp.exp(beta / 2) * mp.besselk(nu, beta / 2) / mp.sqrt(mp.pi * beta))


def V2_closed(beta, nu, dps=None):
    """V_2(beta), the coefficient of z_a^2 in the sigma kernel:
         V_2 = -2 sum_b Psi_b V_1^{(b)} + 4 2F2(1,1;3/2-nu,3/2+nu;beta)/(1-4 nu^2),
    Psi_b = psi(1/2 + b nu) + gamma/2,  V_1^{(+)} = (pi/2) e^{beta/2} I_{-nu}(beta/2)/(sin(pi nu) sqrt(pi beta)),
    V_1^{(-)} = -(pi/2) e^{beta/2} I_{nu}(beta/2)/(sin(pi nu) sqrt(pi beta)).  Guard digits ~ beta/ln10 (the two
    pieces grow like e^beta while V_2 decays)."""
    if dps is None:
        dps = int(25 + 1.2 * float(beta) / np.log(10))
    mp.mp.dps = dps
    beta = mp.mpf(beta); nu = mp.mpmathify(nu)
    pref = mp.exp(beta / 2) / mp.sqrt(mp.pi * beta) * (mp.pi / 2) / mp.sin(mp.pi * nu)
    V0p = pref * mp.besseli(-nu, beta / 2)
    V0m = -pref * mp.besseli(nu, beta / 2)
    Psip = mp.digamma(mp.mpf(1) / 2 + nu) + mp.euler / 2
    Psim = mp.digamma(mp.mpf(1) / 2 - nu) + mp.euler / 2
    F22 = mp.hyp2f2(1, 1, mp.mpf(3) / 2 - nu, mp.mpf(3) / 2 + nu, beta)
    return complex(-2 * (Psip * V0p + Psim * V0m) + 4 * F22 / (1 - 4 * nu ** 2))


def sigma_leg_weak(beta, nu):
    """Weak-mixing dressed sigma leg per unit lam:  VV(beta)/lam -> 4 v(beta),  v = -(1/2) V_2 - i varrho_1 V_1."""
    return 4 * (-0.5 * V2_closed(beta, nu) - 1j * varrho1(nu) * V1_closed(beta, nu))


# ----------------------------------------------------------------------------------------------
# 5. Tables of dressed legs with cubic interpolation in ln beta
# ----------------------------------------------------------------------------------------------
class KernelTable:
    """All kernels of one (lam, nu) point on a log-spaced beta grid, the dressed legs WW_0, WW_1, WW_2, VV,
    cubic splines in ln beta, and the matched soft-tail extrapolation below beta_min."""

    def __init__(self, lam, nu, beta_grid, kernels=None, nodes_kw=None):
        self.lam, self.nu, self.beta = lam, nu, np.asarray(beta_grid, float)
        if kernels is None:
            kernels = FastNodes(lam, nu, beta_min=self.beta[0], **(nodes_kw or {})).kernels(self.beta)
        self.k = kernels                                        # a = + channel: W0, W1, W2, V
        rp, rm = r_weighted(lam, nu)                            # e^{pi lam/2} r_+, e^{-pi lam/2} r_-
        self.rw_p, self.rw_m = rp, rm
        self.R = R_power(lam, nu)
        # dressed legs  WW_n = e^{pi lam/2} r_+ W_n^- + e^{-pi lam/2} r_- W_n^+,  W_n^- = conj(W_n^+)
        dress = lambda K: rp * np.conj(K) + rm * K
        self.WW = {n: dress(self.k['W%d' % n]) for n in range(3)}
        self.VV = (4.0 / lam) * dress(self.k['V'])
        self.PP = self.WW[0] + 0.5 * self.beta * self.WW[1]
        self.lb = np.log(self.beta)
        self._spl = {name: CubicSpline(self.lb, arr) for name, arr in
                     (('W0', self.WW[0]), ('W1', self.WW[1]), ('W2', self.WW[2]), ('V', self.VV), ('P', self.PP))}
        # soft tails: WW_2 ~ 4 sum_b Wcal_b beta^{-1/2-b nu},  VV ~ (4/lam) sum_b Wcal_b beta^{-1/2-b nu}
        self.Wcal_b = {b: Wcal_dressed(lam, nu, b) for b in (+1, -1)}
        bmin = self.beta[0]
        self.c_W2 = self.WW[2][0] - self.tail_W2(bmin)          # matched analytic constant below beta_min
        self.c_V = self.VV[0] - self.tail_V(bmin)

    def tail_W2(self, beta):
        beta = np.asarray(beta, float)
        return 4 * sum(self.Wcal_b[b] * beta ** (-0.5 - b * self.nu) for b in (+1, -1))

    def tail_V(self, beta):
        return self.tail_W2(beta) / self.lam

    def leg(self, name, beta):
        """Dressed leg 'W2', 'V', 'P', 'W0', 'W1' at arbitrary beta > 0 (spline inside the table, matched tail below)."""
        beta = np.asarray(beta, float)
        out = np.empty(beta.shape, dtype=complex)
        inside = beta >= self.beta[0]
        if np.any(beta > self.beta[-1] * (1 + 1e-12)):
            raise ValueError('beta beyond the table: max %g requested %g' % (self.beta[-1], beta.max()))
        out[inside] = self._spl[name](np.log(beta[inside]))
        if np.any(~inside):
            bl = beta[~inside]
            if name == 'W2':
                out[~inside] = self.tail_W2(bl) + self.c_W2
            elif name == 'V':
                out[~inside] = self.tail_V(bl) + self.c_V
            else:                                       # W0, W1, P are finite at beta = 0 (heavy field)
                out[~inside] = self._spl[name](self.lb[0])
        return out


def default_beta_grid(beta_min=1e-13, beta_max=60.0, per_efold=100):
    n = int(np.ceil(np.log(beta_max / beta_min) * per_efold)) + 1
    return np.exp(np.linspace(np.log(beta_min), np.log(beta_max), n))


def build_tables(lams, mus, beta_grid, cache=None, verbose=True, nodes_kw=None):
    """Tabulate the a=+ kernels for a grid of (lam, mu_eff), nu = i mu_eff; cache to an .npz; return dict of KernelTables."""
    import os, time
    lams = np.atleast_1d(lams); mus = np.atleast_1d(mus)
    data, timings = {}, {}
    if cache is not None and os.path.exists(cache):
        z = np.load(cache)
        if np.allclose(z['beta'], beta_grid) and np.allclose(z['lams'], lams) and np.allclose(z['mus'], mus):
            for i, lam in enumerate(lams):
                for j, mu in enumerate(mus):
                    data[(i, j)] = {key: z['%s_%d_%d' % (key, i, j)] for key in ('W0', 'W1', 'W2', 'V')}
            if verbose:
                print('loaded kernel tables from', cache)
    if not data:
        store = {'beta': beta_grid, 'lams': lams, 'mus': mus}
        for i, lam in enumerate(lams):
            for j, mu in enumerate(mus):
                t0 = time.time()
                k = FastNodes(lam, 1j * mu, beta_min=beta_grid[0], **(nodes_kw or {})).kernels(beta_grid)
                timings[(i, j)] = time.time() - t0
                data[(i, j)] = k
                for key in k:
                    store['%s_%d_%d' % (key, i, j)] = k[key]
        if cache is not None:
            np.savez_compressed(cache, **store)
        if verbose:
            tt = np.array(list(timings.values()))
            print('tabulated %d (lam, mu) points: %.2f s per point (min %.2f, max %.2f), total %.1f s'
                  % (len(tt), tt.mean(), tt.min(), tt.max(), tt.sum()))
    tables = {}
    for i, lam in enumerate(lams):
        for j, mu in enumerate(mus):
            tables[(float(lam), float(mu))] = KernelTable(lam, 1j * mu, beta_grid, kernels=data[(i, j)])
    return tables, timings
