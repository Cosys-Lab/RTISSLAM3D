# Spec: Heat-Kernel CFAR on the Sphere (RTISSLAM 3D)

This is the detector RTISSLAM 3D uses to turn a beamformed energy volume into 3D detections. It generalises the range-wise CFAR (RW-CFAR) of 2D RTISSLAM:

- **RW-CFAR:** the clutter level is the mean over *all* directions at a range.
- **Heat-kernel CFAR:** the clutter level is a *local* weighted mean over neighbouring directions. "Local" is defined by the heat kernel of a graph Laplacian on the sampled hemisphere.
- **Link:** RW-CFAR is the infinite-time limit of the heat-kernel CFAR.

**Code:**
- `rtis3d/laplacian.py` (`HeatKernel`, `sphere_cfar`)
- `rtis3d/detections.py` (`Detector`)

## 1. Input

- **Energy volume** $P(r,\psi)$, of size $[R \times M]$:
  - $R$ = 475 range cells of 2 cm, up to 9.7 m;
  - $M$ = number of sampled directions;
  - produced by DMAS5 + CF beamforming, as the envelope zero-phase low-passed at 5 kHz.
- **Directions** $\vec u_m$: a Fibonacci (golden-angle) lattice on the frontal hemisphere ($x > 0$) at about 3° spacing (`rtis3d/sphere.py`). Each direction covers roughly the same solid angle. The frame is x forward, y left, z up.
- **Units:** the volume is stored in dB and converted to linear power, $P = 10^{\,\mathrm{dB}/10}$. All CFAR operations work on linear power.

## 2. Graph on the directions

**k-NN graph:** each direction is connected to its $k = 10$ nearest neighbours under a beam-matched angular distance:

$$d_{az} = \mathrm{wrap}(az_n - az_m)\cdot\cos\!\Big(\tfrac{el_m + el_n}{2}\Big), \qquad d_{el} = \frac{el_n - el_m}{s_{el}}, \qquad d^2 = d_{az}^2 + d_{el}^2$$

- $s_{el}$ = 1.7, because the eRTIS main lobe is about 1.7× wider in elevation than in azimuth.
- The scaling makes the smoothing follow the point-spread function instead of being isotropic.

**Weights:**

$$W_{mn} = \exp\!\big(-d_{mn}^2 / h^2\big)$$

- $h^2$ is the median squared k-NN distance over all nodes.
- The matrix is symmetrised with $W \leftarrow \max(W, W^\top)$.
- The degree is $d_m = \sum_n W_{mn}$, and $D = \mathrm{diag}(d)$.

## 3. Heat kernel

**Random-walk Laplacian and heat kernel:**

$$L = I - D^{-1}W, \qquad H_t = e^{-tL}$$

**Computation:**
- Use the eigendecomposition of the symmetric normalised Laplacian:
$$I - D^{-1/2} W D^{-1/2} = V\Lambda V^\top$$
  with $\Lambda$ clipped at $\geq 0$.
- The kernel is then
$$H_t = D^{-1/2}\, V e^{-t\Lambda} V^\top\, D^{1/2}$$
- The dense $[M\times M]$ matrix is computed once per direction set.

**Properties:**
- $H_t$ is **row-stochastic**: each row is a weighted local mean that keeps constants unchanged. It therefore needs no special treatment at the edge of the hemisphere.
- It is the graph analogue of the Laplace–Beltrami heat kernel $e^{-t\Delta_{S^2}}$. In spherical-harmonic form, that kernel multiplies degree $\ell$ by $e^{-t\,\ell(\ell+1)}$.

**Choosing $t$:**
- $t$ is set by bisection: 60 geometric steps over $[10^{-3}, 10^{5}]$.
- The target is that the **RMS angular radius** of the kernel row at broadside (the direction with the largest $x$) equals `sigma_deg`:
$$\sigma(t) = \sqrt{\frac{\sum_n H_t[i_0,n]\,\theta_{i_0 n}^2}{\sum_n H_t[i_0,n]}}$$
  where $\theta$ is the great-circle angle.

## 4. CFAR

**Background:** the local mean over directions, then a Gaussian along range:

$$B(r,\psi_m) = \Big(g_{\sigma_r} *_r \sum_n H_t[m,n]\,P(r,\psi_n)\Big)$$

- $g_{\sigma_r}$ is a normalised Gaussian with $\sigma_r$ = 3 range cells (6 cm), truncated at $\pm\lceil 3\sigma_r\rceil$.
- Edges use replicate padding.

**Output** (the CFAR excess):

$$P^{+}(r,\psi) = \max\big(P(r,\psi) - g\cdot B(r,\psi),\ 0\big)$$

**Implementation:** torch on the GPU when available. The smoothing over directions is `P @ K.T`, and the range smoothing is a `conv1d`.

## 5. Detections

1. **Range gate:** set $P^{+}$ to 0 outside $r \in [0.5, 8]$ m.
2. **Peak-relative threshold:** a cell is a detection if
$$P^{+}(r,\psi) > \max_{r',\psi'} P^{+}(r',\psi')\cdot 10^{-25/10}$$
   i.e. within 25 dB of the strongest output in that capture.
3. **Output:** each detection has
   - position $\vec p_j = r_j\,\vec u(\psi_j)$ in the sensor frame, with $r_j$ the cell centre;
   - weight $w_j = P^{+}_j$, the CFAR excess in linear units.
4. **Ground-multipath mirror removal** (after the CFAR):
   - In point-target simulations, the mirrors are only 4–9 dB weaker than the direct echoes, so the CFAR cannot reject them by level.
   - Fit a robust plane $\vec n^\top\vec x = d$ to the low near-field detections: $r < 3$ m, $z < -0.1$ m, and $z > -(h_{prior}+0.6)$ with $h_{prior}$ = 0.32 m. At least 15 points are needed.
   - The fit is six iterations of iteratively reweighted least squares with a Huber weight ($k$ = 0.08 m), using $\sqrt{w_j}$ as base weights. The normal is the smallest eigenvector of the weighted covariance.
   - The plane is accepted if its normal is within 30° of vertical.
   - Detections more than 0.15 m below the plane are dropped.
   - The plane also gives the sensor height above the local ground ($-d$) and its tilt.

## 6. Parameters (pipeline defaults)

| Symbol / name | Value | Meaning |
|---|---|---|
| `sigma_deg` | 12° | RMS angular radius of the kernel at broadside (sets $t$) |
| `el_scale` $s_{el}$ | 1.7 | elevation stretch of the metric (beam anisotropy) |
| `k` | 10 | graph neighbours |
| `range_sigma` $\sigma_r$ | 3 cells (6 cm) | Gaussian smoothing of the background along range |
| `gain` $g$ | 5 | CFAR gain (default argument of `sphere_cfar` is 2; the `Detector` uses 5) |
| `peak_db` | 25 dB | detections within this of the capture's strongest output |
| `r_min`, `r_max` | 0.5 m, 8 m | range gate |
| `below` | 0.15 m | mirror rejection distance under the ground plane |
| `h_prior` | 0.32 m | prior sensor height |

## 7. Limits and relation to RW-CFAR

**Large $t$** ($t\to\infty$, `sigma_deg` → ∞):
- $H_t$ converges to the stationary distribution of the random walk: every row becomes $d_n/\sum d$.
- So $B(r,\cdot)$ becomes a degree-weighted mean over all directions at range $r$, which is the **RW-CFAR** of 2D RTISSLAM:
$$P^{+} = \max\big(P - \alpha\,(g_\sigma * \tilde P)(r),\, 0\big), \qquad \tilde P(r) = \tfrac{1}{M}\textstyle\sum_m P(r,\psi_m)$$
  (2D RTISSLAM, range-wise CFAR; final 2D setting $\alpha$ = 2, smoothed.)
- Strictly, the limit is degree-weighted rather than uniform. On the Fibonacci lattice the k-NN degrees are nearly uniform, so the difference is small.

**Small $t$:** a local cell-averaging CFAR on the sphere.

**Why local in 3D:**
- Over a full hemisphere, the mean over all directions is a poor clutter model.
- At short range the ground fills the lower half and inflates the per-range mean, which suppresses real targets at that range.
- A few trunks barely move the mean, so their sidelobes survive.
- The local mean follows the clutter where it actually is.

## 8. Evaluation hook

`scripts/overlay3d.py` scores detections against the 3D lidar for each capture, without SLAM.

- **Variants:** RW-CFAR (plain mean, lines ~103–110) and heat-kernel CFAR at 6°, 12° and 25°.
- **Gains:** 2, 3 and 5 for each variant.
- **Metrics:**
  - precision: the fraction of detections within a tolerance of a lidar point;
  - the fraction of detections below the local lidar ground (mirrors);
  - detections per capture.

No result JSON is stored in the repository. Re-run it to regenerate the numbers behind the 12° / gain 5 choice:

```bash
python scripts/overlay3d.py <vol dir> <dataset.h5> --offset 409980.68 --n 200
```

## 9. Usage

```python
from rtis3d.detections import Detector
det = Detector(U, ranges, sigma_deg=12, gain=5, peak_db=25)   # builds the heat kernel once
pts, w, info = det(volume_db)    # pts [n,3] sensor frame, w [n] CFAR excess, info: n_raw, plane, height, n_mirror
```
