# Exact bispectra in strongly mixed multifield inflation

Tree-level primordial bispectra of the effective field theory of inflationary fluctuations extended by
a massive isocurvature scalar σ, with the curvature–isocurvature mixing ρ π̇<sub>c</sub>σ resummed to
all orders in λ = ρ/H. This repository accompanies

* L. Pinol, *Exact bispectra in strongly mixed multifield inflation*, [arXiv:2609.38167](https://arxiv.org/abs/2609.38167)
* L. Pinol, *New exact bispectrum shapes in multifield inflation*, [arXiv:2607.15251](https://arxiv.org/abs/2607.15251)

| file | what it is |
|---|---|
| `index.html` | the explorer, self-contained |
| `ec_kernels.py`, `ec_shapes.py` | two Python modules that compute the shapes |
| `example.ipynb` | a short notebook that uses them |
| `data/` | the raw tables, in CSV, see `data/README.md` |

## The explorer

**<https://lucaspinolcnrs.github.io/exact-collider/>** computes the six shapes in your browser at any
point with λ < 6 and 0.5 < μ<sub>eff</sub> < 6 — the weight, the leg kernels and the
Schwinger-parameter integral are evaluated live, not interpolated — draws them over all triangles and
on isosceles configurations, compares a single operator with the Planck PR4 binned bispectrum, and
exports what it draws as CSV. It is a single file, `index.html`, which also runs offline: download it with the link at the bottom
of the explorer, or with the download button of [`index.html`](https://github.com/lucaspinolCNRS/exact-collider/blob/main/index.html),
and open it in a browser.

Its controls start behind a small game. To skip it, give the gatekeeper the word `lucas.pinol`, or
open **<https://lucaspinolcnrs.github.io/exact-collider/?open>**.

## Python

`ec_kernels.py` tabulates the dressed leg kernels at a point (λ, μ<sub>eff</sub>), and `ec_shapes.py`
turns them into the shapes of the five computed channels. They need only `numpy`, `scipy` and
`mpmath`:

```python
from ec_shapes import get_table, reduced_shapes
T = get_table(2.0, 2.0)                      # lambda, mu_eff
S = reduced_shapes(T, [1.0], [1.0], [1.0])   # channels '0', '1par', '1perp', '2', '3': S Delta_zeta / c_n at k1 = k2 = k3
```

`example.ipynb` draws one shape over all triangles, the isosceles cut down to the squeezed limit and
the equilateral amplitudes along λ at fixed μ<sub>eff</sub>, and checks the weak-mixing limit of the
no-exchange channel. The explorer's JavaScript engine and these modules implement the same algorithm
independently, and agree to better than 10⁻⁵ over λ ≤ 6.

## Data

`data/` holds the raw tables: the equilateral amplitudes and shape cosines of the six channels over
the (λ, μ<sub>eff</sub>) plane, the full comparison with Planck PR4 (every point of the scan and each
channel's best fit), and the best-fitting templates on the Planck bins and over all triangles.

## Conventions

* S(k<sub>1</sub>,k<sub>2</sub>,k<sub>3</sub>) = (k<sub>1</sub>k<sub>2</sub>k<sub>3</sub>)² B<sub>ζ</sub> / [(2π)⁴ Δ<sub>ζ</sub>⁴] and
  f<sub>NL</sub> = (10/9) S(k,k,k), with Δ<sub>ζ</sub>² = 2.100549×10⁻⁹.
* m<sub>eff</sub>² = m² + ρ², and μ<sub>eff</sub> = √(m<sub>eff</sub>²/H² − 9/4) = √(λ² + m²/H² − 9/4) is the collider frequency.
* The six channels, keyed as in every file here, with their dimensionless couplings c<sub>n</sub>:

| key | vertex in ℒ⁽³⁾/a³ | c<sub>n</sub> |
|---|---|---|
| `0` | −λ<sub>2</sub> π̇<sub>c</sub>³ | λ<sub>2</sub>H² |
| `1par` | −π̇<sub>c</sub>²σ / 2Λ<sub>2</sub> | H/Λ<sub>2</sub> |
| `1perp` | −(∂<sub>i</sub>π<sub>c</sub>)²σ / 2Λ<sub>1</sub>a² | H/Λ<sub>1</sub>, fixed by the non-linearly realised boosts: H/Λ<sub>1</sub> = 2πΔ<sub>ζ</sub>λ/√R |
| `LI` | −(∂<sub>μ</sub>π<sub>c</sub>)²σ / 2Λ<sub>1</sub> | H/Λ<sub>1</sub>, at the boost-invariant point H/Λ<sub>2</sub> = −H/Λ<sub>1</sub> |
| `2` | −(α/2) π̇<sub>c</sub>σ² | α |
| `3` | −μ σ³ | μ/H — μ is the σ³ coupling, μ<sub>eff</sub> the collider frequency |

* Planck PR4: templates normalised to 1 at the equilateral point and evaluated at
  k<sub>i</sub><sup>(4−n<sub>s</sub>)/3</sup> with n<sub>s</sub> = 0.9660499, as the release requires.
* Double precision is reliable up to λ ≈ 6.

## Citing

Please cite the two papers above. Each release of this repository is also archived on Zenodo with its
own DOI.
<!-- after the first release, paste the Zenodo badge here -->

## Licence

Code: MIT, see `LICENSE`. Data: CC0 1.0, see `LICENSE-DATA.md`. The Planck PR4 binned shape inlined in
the explorer and stored in `data/` is O. H. E. Philcox's release, under its own MIT licence, see
`THIRD_PARTY.md`.
