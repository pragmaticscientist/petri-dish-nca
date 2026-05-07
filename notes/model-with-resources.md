# Petri Dish NCA with Resources

## Overview

This document describes the resource mechanic added to the adversarial Neural Cellular Automata (NCA) simulation. Resources create fitness niches: species whose internal state aligns with a locally abundant resource type gain a territory advantage, making *being different from your competitor* advantageous rather than just *being stronger*.

## Cell Tensor Layout

Each cell carries $C = C_s + C_h + N + 1$ channels:

$$
\underbrace{a_0, a_1, \ldots, a_N}_{\text{alive (sun + NCAs)}} \;\Big|\; \underbrace{s_0, \ldots, s_{C_s-1}}_{\text{state (attack | defense)}} \;\Big|\; \underbrace{h_0, \ldots, h_{C_h-1}}_{\text{hidden}}
$$

The first half of the state channels $\mathbf{v}^{\mathrm{att}} \in \mathbb{R}^{C_s/2}$ are the **attack vector**; the second half $\mathbf{v}^{\mathrm{def}} \in \mathbb{R}^{C_s/2}$ are the **defense vector**.

## Resource Tensor

Resources live in a separate tensor $\mathbf{R} \in \mathbb{R}^{B \times K \times H \times W}$ (not part of the cell tensor), where $K$ is the number of resource types. Each resource type $k$ has a fixed **fingerprint vector** $\mathbf{f}_k \in \mathbb{R}^{C_s/2}$, chosen to be orthonormal:

$$
\mathbf{F} \in \mathbb{R}^{K \times C_s/2}, \qquad \mathbf{F}\mathbf{F}^\top = \mathbf{I}_K
$$

These fingerprints define which region of attack-vector space is rewarded by each resource type.

## Resource Dynamics

At the start of each epoch the resource amounts undergo **logistic regrowth** toward a carrying capacity $\kappa$:

$$
R_{k,x,y} \;\leftarrow\; R_{k,x,y} + \alpha\,(\kappa - R_{k,x,y})
$$

with optional **spatial diffusion** (average-pooling over a $3\times3$ neighborhood):

$$
R_{k,x,y} \;\leftarrow\; (1-\delta)\,R_{k,x,y} + \delta\,\overline{R}_{k,x,y}
$$

After each epoch, living NCAs **consume** resources proportional to their cosine affinity and local aliveness:

$$
R_{k,x,y} \;\leftarrow\; R_{k,x,y} - \beta\;\max\!\bigl(0,\, \cos(\mathbf{v}^{\mathrm{att}}_{x,y},\, \mathbf{f}_k)\bigr)\cdot A_{x,y}
$$

where $A_{x,y} = \sum_{m=1}^{N} a_{m,x,y}$ is total NCA aliveness (sun excluded) and $\beta$ is the consumption rate.

## Competition with Resource Bonus

At each simulation step, every entity $m$ proposes an update vector from which attack and defense channels are extracted. The **territory strength** of entity $m$ at location $(x,y)$ from perspective $n$ aggregates pairwise cosine similarities against all opponents:

$$
\sigma_{n,m,x,y} = \sum_{m' \neq m} \cos(\mathbf{v}^{\mathrm{att}}_{m},\, \mathbf{v}^{\mathrm{def}}_{m'}) - \cos(\mathbf{v}^{\mathrm{att}}_{m'},\, \mathbf{v}^{\mathrm{def}}_{m})
$$

The **resource bonus** adds a niche-fitness term:

$$
\rho_{n,m,x,y} = \sum_{k=1}^{K} R_{k,x,y} \cdot \cos\!\left(\mathbf{v}^{\mathrm{att}}_{n,m,x,y},\; \mathbf{f}_k\right)
$$

The final strength used for softmax territory assignment is:

$$
\tilde{\sigma}_{n,m,x,y} = \sigma_{n,m,x,y} + \lambda\,\rho_{n,m,x,y}
$$

where $\lambda$ is `resource_strength_weight`. Aliveness is then assigned as:

$$
a_{m,x,y} = \operatorname{softmax}_m\!\left(\tilde{\sigma}_{n,m,x,y} \,/\, \tau\right)
$$

with temperature $\tau$.

## Why This Produces Character Displacement

With two orthogonal resource fingerprints $\mathbf{f}_0 \perp \mathbf{f}_1$, gradient descent pushes each NCA's attack vector toward whichever fingerprint yields the largest local bonus. When two NCAs co-occur in the same region (**sympatry**), they both compete for the same resources: the NCA that consumes resource $k$ locally depletes it, reducing the bonus for its competitor. The gradient then pushes the competitor toward the *other* resource fingerprint. This is **negative frequency-dependent selection**: common trait values deplete their local niche and rare ones gain advantage.

**Character displacement experiment:**
- *Allopatric*: each NCA trained alone; record its mean attack vector.
- *Sympatric*: both NCAs in the same world; record their mean attack vectors.

If $\|\bar{\mathbf{v}}^0_{\text{symp}} - \bar{\mathbf{v}}^1_{\text{symp}}\| > \|\bar{\mathbf{v}}^0_{\text{allo}} - \bar{\mathbf{v}}^1_{\text{allo}}\|$ reliably across seeds, that is character displacement.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_resources` | `0` | Number of resource types (0 = disabled) |
| `resource_carrying_capacity` | `1.0` | Maximum resource amount $\kappa$ |
| `resource_regrowth_rate` | `0.1` | Regrowth rate $\alpha$ per epoch |
| `resource_diffusion` | `0.0` | Spatial diffusion coefficient $\delta$ |
| `resource_consumption_rate` | `0.1` | Consumption rate $\beta$ per epoch |
| `resource_strength_weight` | `1.0` | Resource bonus weight $\lambda$ |

## Example

```bash
uv run python src/train.py --config configs/example.json
```

with `configs/resources_2nca.json`:
```json
{
    "n_ncas": 2,
    "n_resources": 2,
    "resource_strength_weight": 2.0,
    "resource_regrowth_rate": 0.05,
    "resource_consumption_rate": 0.2,
    "cell_state_dim": 8,
    "grid_size": [20, 20],
    "epochs": 5000
}
```
