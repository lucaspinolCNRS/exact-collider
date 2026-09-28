# Data

The raw tables behind the companion paper, as plain CSV with a commented header (lines starting
with `#`). Read them with `pandas.read_csv(f, comment='#')`, or with
`numpy.genfromtxt(f, delimiter=',', names=True, comments='#', dtype=None, encoding=None)`.
Channel keys, couplings and conventions are those of the top-level `README.md`.

| file | content |
|---|---|
| `plane_amplitudes_cosines.csv` | The (λ, μ<sub>eff</sub>) plane on a 200 × 200 grid, λ = 0.05–6.02 and μ<sub>eff</sub> = 0.50–6.47 in steps of 0.03. For each channel: `fnlpc_<ch>`, the equilateral amplitude per unit coupling f<sub>NL</sub><sup>(n)</sup>/c<sub>n</sub>; `cos_<ch>`, the cosine with the equilateral template on the dx = 0.01 triangle grid; `reproj_<ch>`, the mean of \|Re B\|/\|B\| over the brackets at the equilateral point, the fraction of the complex bracket that survives the real projection. |
| `planck_scan_coarse.csv` | The comparison with Planck PR4, coarse pass: λ = 0.1–6.0 and μ<sub>eff</sub> = 1.0–4.0 in steps of 0.1, six channels. `fnl_hat` and `sigma_fnl` for the template normalised to 1 at the equilateral point, `snr`, and `S11_reduced` = S<sup>(n)</sup>(k,k,k) Δ<sub>ζ</sub>/c<sub>n</sub>. |
| `planck_scan_refined.csv` | The two refinement passes, on a 0.03 lattice where the coarse pass exceeded 1σ and a 0.01 lattice above 2σ; same columns, plus the step of the pass. |
| `best_fits.csv` | The largest excursion of each channel, at the exact lattice point of the scan: m<sub>eff</sub>/H, m²/H², `fnl_hat` ± `sigma_fnl`, `snr`, f<sub>NL</sub>/c<sub>n</sub> at that point and the coupling it implies. |
| `bestfit_templates_planck_bins.csv` | The six best-fitting templates on the 171 bins of the Planck release, normalised to 1 at the equilateral point and tilted as the release requires. With the release's data and covariance they return the numbers of `best_fits.csv` (see below). |
| `bestfit_shapes_triangles.csv` | The same six shapes over the dx = 0.01 triangle grid, untilted, normalised to 1 at the equilateral point. |
| `planck_binned_shape.bin` | Not ours: the Planck PR4 binned shape of O. H. E. Philcox (arXiv:2603.17004, MIT licence, see `../THIRD_PARTY.md`), repacked for the explorer. Little-endian: the 4 bytes `ECBS`, uint32 version = 1, uint32 n = 171, 4 bytes of padding, then float64 arrays x, y and the measured shape (n values each), and the n × n inverse covariance, row by row. |

In the two scan tables the rows run over channels, then λ, then μ<sub>eff</sub>. The signal-to-noise
values are the largest excursions of a scan over ~10⁵ correlated templates, not significances.

The best fit of the no-exchange channel, from the tables alone:

```python
import numpy as np, pandas as pd
n = 171
raw = np.fromfile('planck_binned_shape.bin', dtype='<f8', offset=16)
x, y, d, icov = raw[:n], raw[n:2*n], raw[2*n:3*n], raw[3*n:].reshape(n, n)
t = pd.read_csv('bestfit_templates_planck_bins.csv', comment='#')['S_0_over_S11'].values
F = t @ icov @ t
print((t @ icov @ d) / F, F ** -0.5)          # fnl_hat = -121.14, sigma_fnl = 43.89
```

**Precision.** Everything here was computed with the double-precision modules of this repository,
`ec_kernels.py` and `ec_shapes.py` (leg kernels on a logarithmic grid β ∈ [10⁻¹³, 60] with 100 nodes
per e-fold), and is reliable up to λ ≈ 6.

**Coordinates of the maxima.** The refined lattices sit at odd multiples of 0.005, so the maxima of
the scan are at points such as (λ, μ<sub>eff</sub>) = (2.305, 2.545) for no exchange. `best_fits.csv`
gives them to three decimals: evaluating a template at the two-decimal rounding of such a point, on a
narrow ridge, shifts `fnl_hat` by several per cent.
