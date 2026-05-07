# Speciation in Petri Dish NCA — Brainstorm

Notes from a design discussion on how to model speciation in this codebase. Two
options were considered. Option (a) is the cheap, interpretive route; option
(b) is the honest one and requires real refactoring.

## Option (a): single NCA, emergent type channels (polymorphism)

Set `n_ncas = 1`. Because there is only one network — one "genome" — every
living cell belongs to the same species by construction. That frees the
per-cell state vector to carry a different meaning: a **trait / type embedding**.

- Reinterpret some or all of `cell_state_dim` channels as a type vector field
  over the grid. With e.g. `cell_state_dim = 16`, the last 8 channels can be
  treated as an 8-dim trait embedding per cell.
- "Speciation" is then **clustering in trait space**: take the type vectors of
  all living cells, project (PCA / t-SNE / k-means), and see whether they form
  one mode or several stable modes.
- Two stable basins of attraction = polymorphism. One mode = no differentiation.

### What this actually models

Not strict speciation — there is still one network. It models **cell-type
differentiation** (one genome producing multiple stable phenotypes, like
neurons vs. skin cells). To get true speciation under this framing you'd need
a reproductive-isolation rule: cells only copy / influence type vectors from
same-cluster neighbors.

### What makes the bifurcation an attractor

A single network won't spontaneously split its trait field into two modes
unless something rewards being different from your neighbors. The natural
mechanism is **frequency-dependent fitness via depletable resources**: add
resource channels that get consumed by cells based on their trait vector, so
common trait values deplete their resource locally and rare ones don't. That's
what turns "two clusters" into a stable equilibrium rather than noise.

### Why this only makes sense with `n_ncas = 1`

With multiple NCAs, the cell's identity is already pinned by the alive
channels — the state vector is just working memory for that NCA's update
rule. There's no room to reinterpret it as species identity because species
identity is already encoded elsewhere. With one NCA there's no competing
identity signal, so the state vector is free to carry the trait
interpretation.

---

## Option (b): lineage forking (real speciation)

The honest version. Breaks a load-bearing assumption: `MergedCAModel` uses
`groups=n_ncas` in a single `Conv2d`, and `n_ncas` is fixed at init.

Two ways to handle this:

1. Reallocate the conv with a new group on the fly — ugly, breaks optimizer
   state. Don't.
2. **Pre-allocate `n_ncas_max` slots**, most initially inactive, and "activate"
   a slot by copying a parent's weights with noise when a fork condition
   fires. This is the tractable path.

### Design sketch

**Slot model.** Keep `MergedCAModel` with `groups=n_ncas_max` (e.g. 8). Add a
boolean `active: BoolTensor[n_ncas_max]` on `CASunGroup`. Inactive slots:

- Have weights present in the conv but never receive gradient (mask loss /
  zero grads after backward).
- Their alive channel is forced to 0 in the grid; `_init_pool` doesn't seed
  them.
- `n_ncas` in the rest of the code becomes "active count" — most call sites
  that index `1 : n_ncas+1` need to use the `active` mask instead. This is
  the invasive part.

**Fork trigger.** Cheapest viable signal: per-NCA, per-region trait variance.
Compute variance of the state vector across living cells of NCA `i` within
local windows; if it's bimodal (or just exceeds a threshold for `K`
consecutive epochs), fork. A simpler proxy: split a slot when its territory
has been growing steadily for `K` epochs *and* a free slot exists —
"successful lineages get the chance to diversify." Don't overthink the
trigger first pass; see if the mechanism produces anything interesting before
tuning the policy.

**Activation step.** When slot `j` (free) forks from slot `i`:

- Copy `i`'s conv weights for group `i` into group `j`'s slice, add Gaussian
  noise (`σ` small — start ~1e-2 of weight std).
- Set `active[j] = True`.
- Seed slot `j`: pick a subset of slot `i`'s living cells (e.g. half a
  contiguous region) and reassign their alive channel from `i` to `j`. This is
  the "geographic" speciation event.
- **Optimizer state**: Adam/RMSProp momentum buffers for group `j` are stale.
  Either reset those buffer slices to zero on activation, or copy from group
  `i` alongside the weights. Resetting is safer.

**Extinction.** When slot `i`'s total alive mass falls below threshold for
`K` epochs: set `active[i] = False`, zero its alive channel everywhere in the
pool, zero its optimizer buffers so a future fork into this slot starts
clean. Either leave the weights (lets a future fork reuse niche memory) or
zero them (strict isolation). Pick one and commit.

### Where this lands in the code

- `model.py:CASunGroup` — add `active`, `n_ncas_max`, fork/extinct methods,
  gradient masking in `update_models`.
- `world.py` — new `LineageFeature(Feature)` whose `after_step` evaluates
  triggers and calls `group.fork(i, j)` / `group.extinct(i)`. Keeps the
  policy out of the model class.
- `config.py` — add `n_ncas_max`, `fork_trigger_*`, `fork_noise_std`,
  `extinction_threshold`, `extinction_patience`.
- `world.py:_init_pool` — only seed cells for initially active slots
  (probably just slot 0 or a small `n_ncas_initial`).

### The honest cost

Every place in the codebase that does `range(n_ncas)` or slices
`[1:n_ncas+1]` needs to become "iterate active slots." That's the refactor
tax. Worth grepping `model.py` and `world.py` for those index sites first to
size the refactor — if <20 sites, fine; if 50, build a thin `ActiveSlots`
helper to centralize it.

---

## Resources

Resources are the mechanism that gives the rest of these ideas teeth. Without
them, "competition" in the codebase is already implemented (attack vs.
defense cosine similarity for territory), but there's nothing that makes
*being different* advantageous — only *being stronger*. Resources fix that.

### Minimal model

Add `n_resources` extra channels to the cell tensor — call them resource
channels, sitting alongside state/hidden but treated specially:

- They are **not** updated by the NCA's conv output. They follow their own
  dynamics (regrowth + consumption).
- **Regrowth**: each step, each resource channel relaxes toward a per-channel
  carrying capacity (e.g. `r += α * (K - r)`), optionally with spatial
  diffusion so depleted patches refill from neighbors.
- **Consumption**: each living cell consumes resources at a rate that depends
  on its trait vector (option a) or its NCA identity (option b). The simplest
  coupling is a learned or fixed `consumption_weights[n_traits, n_resources]`
  matrix; a cell's consumption of resource `k` is
  `softmax(trait)·consumption_weights[:, k]` times its aliveness.
- **Fitness coupling**: a cell's growth / survival is gated by the resources
  available locally. Cheapest hookup: scale the cell's effective alive
  contribution by `min_k(local_resource_k / demand_k)` (Liebig's law of the
  minimum), so a cell starves if any required resource is locally depleted.

### Why this matters for option (a)

A single NCA with no resources will not spontaneously bifurcate its trait
field — there's no gradient pulling it toward "be different from your
neighbor." With resources, common trait values locally deplete their
preferred resource, so rare trait values have a fitness advantage. That's
**negative frequency-dependent selection**, and it's what turns two clusters
in trait space into a stable attractor instead of noise.

### Why this matters for option (b)

Without resources, the only thing distinguishing lineages after a fork is
combat strength. Selection collapses back to "the strongest variant wins" and
forks get extinguished quickly. Resources create **niches** — a lineage that
specializes on resource A can coexist with one that specializes on B even if
B is combat-stronger overall, because B starves where A doesn't. That's the
condition for forks to persist long enough to look like real speciation.

### Where this lands in the code

- `config.py` — `n_resources`, `resource_carrying_capacity`,
  `resource_regrowth_rate`, `resource_diffusion`, `consumption_mode`
  (`fixed` | `learned`).
- World tensor — extend `cell_dim` or carry resources as a separate
  `world.resources` tensor of shape `[batch, n_resources, H, W]`. Separate
  tensor is cleaner: the NCA's grouped conv shouldn't touch resource
  channels, and you avoid masking out indices in every model call.
- `world.py` — new `ResourceFeature(Feature)` with `before_step` (regrowth +
  diffusion) and `after_step` (consumption + fitness gating). Keeps it
  parallel to existing feature hooks.
- `model.py` — the alive update needs to read local resource availability.
  Either pass `resources` as an extra arg into `CASunGroup.forward`, or
  concat resource channels to the model's input view (read-only — gradients
  flow through the network's reaction to them, not into them).

### Open question: are consumption weights learned or fixed?

Fixed (random per slot, frozen at init) is simpler and keeps the experiment
about whether the NCA *uses* the niches. Learned (consumption weights as
parameters trained jointly) is more powerful but introduces an extra reason
for collapse — the network can just learn to consume everything. Start
fixed.

---

## Character displacement

Character displacement is the prediction that **two species' traits diverge
more in sympatry than in allopatry** — when they're forced to coexist,
selection pushes them apart to reduce competition. It's a clean, falsifiable
test of whether the system is doing what we think it's doing.

### As an experiment, not a mechanism

Character displacement isn't something you implement; it's something you
**measure**, given resources are in place. The setup:

1. Train two configurations side by side:
   - **Allopatric**: two NCAs (option b) or two seeded trait clusters
     (option a) initialized in **separate worlds** — they never see each
     other. Run to convergence; record their trait vectors / consumption
     profiles.
   - **Sympatric**: same two lineages initialized in the **same world**,
     sharing the grid and resource pool. Run to convergence; record again.
2. Compare the trait-space distance between the two lineages across
   conditions. If `dist_sympatric > dist_allopatric` reliably, you have
   character displacement.

### What you need to make this measurable

- A **trait readout** that's stable across runs. For option (a), the mean
  state-vector cluster centroid per region. For option (b), per-slot
  consumption profile (the resource each lineage actually uses) is the
  cleanest metric — it's low-dimensional and biologically interpretable.
- **Multiple seeds per condition.** Single runs are noise. Want at least
  10–20 seeds per (allopatric, sympatric) cell to claim a real effect.
- **Allopatric runs that match sympatric except for the encounter.** Same
  resource pool composition, same grid size, same NCA capacity — only the
  presence of the competitor changes. Otherwise you're measuring the wrong
  thing.

### Where this lands in the code

- No new `Feature` needed for the mechanism; resources already do the work.
- New script `src/experiments/character_displacement.py` — orchestrates the
  paired runs, dumps trait readouts to disk, computes the displacement
  metric.
- `viz.py` — a 2D scatter of trait centroids per condition is the obvious
  plot; pairs of points (one allopatric, one sympatric per lineage) with
  arrows connecting them shows the displacement directly.

### Why this is the most interesting experiment

Speciation is a structural claim ("the system can produce two stable
lineages"). Character displacement is a **dynamical** claim ("the lineages
adapt *to each other's presence*"). The latter is harder to fake — you can
get two clusters by accident, but you can't get coordinated divergence under
sympatry without genuine eco-evolutionary feedback. If the model shows
character displacement, that's a strong signal it's doing something real.

---

## Learning algorithm: within-lifetime gradients vs. between-lifetime selection

### What the codebase does today

The current setup is within-lifetime BPTT:

- Each epoch, `World.step` runs `steps_before_update` no-grad burn-in steps,
  then `steps_per_update` steps with gradients enabled.
- `CASunGroup.update_models` computes a per-NCA loss
  `-asinh(batch_alive)` — each NCA wants to maximize its own territory mass —
  and backprops through the unrolled simulation
  ([model.py:545-551](src/model.py#L545-L551)).
- Gradient norm is clipped to 1.0; weights for all NCAs live in one grouped
  Conv2d, so each group's grads stay isolated.
- There's already a *tiny* population component:
  `UpdatePoolWithNondeadFeature` only writes back pool entries where no NCA
  died. That's selection at the run level, layered on top of gradients.

### The choice

**(1) Pure within-lifetime gradients (BPTT, current).**

- *Pros:* every cell at every step gives gradient signal — sample-efficient.
  Already implemented, already works.
- *Cons:* memory cost grows linearly in `steps_per_update` (saved activations
  × grid size). Loss is a *proxy* — territory at step t, not whole-run
  viability. Worst of all, **BPTT can't see discrete events**: a fork, an
  extinction, a resource-triggered death are non-differentiable, so anything
  involving lineage births/deaths is invisible to the gradient.

**(2) Pure between-lifetime selection (ES / evolutionary / population-based).**

- *Pros:* gradient-free. Handles discrete events natively — fork, extinct,
  spawn, all fine. Fitness can be anything you can compute at end-of-run
  (whole-run territory, terminal trait diversity, character displacement
  score). Parallelizes embarrassingly: each population member is an
  independent rollout.
- *Cons:* dramatically less sample-efficient. ES on a conv network with
  ~10⁵+ parameters typically needs population sizes in the hundreds–thousands
  and many generations to match what BPTT finds in minutes. Practical only
  with significant parallel compute, and tends to be the right call only
  when gradients are unavailable or actively misleading.

**(3) Hybrid (BPTT within lifetime, selection at the lineage level).**

This is Population Based Training territory and is almost certainly the right
fit for this project:

- The **update rule** (conv weights) is trained by BPTT during a lineage's
  lifetime — keep the current efficient gradient signal.
- The **lineage-level events** (fork, extinct, parameter resets, weight
  perturbation) are handled by selection — they were never in the gradient
  graph anyway.
- The pool's existing "kill dead runs" rule is already a primitive form of
  this; we'd be making it more explicit and more powerful.

### Recommendation by direction

- **Option (a) polymorphism + resources:** stay with BPTT. The whole story
  (state-vector field, resource consumption, frequency-dependent fitness) is
  differentiable end-to-end. No reason to give up the gradient.
- **Option (b) lineage forking:** go hybrid. BPTT trains each active slot's
  weights; the `LineageFeature` triggers forks/extinctions outside the
  gradient graph; new slots are seeded by copy-with-noise from a parent.
  This is the design already sketched in the option (b) section above —
  worth recognizing it explicitly as a hybrid PBT-style algorithm.
- **Pure ES** is overkill unless we discover the BPTT loss is *actively
  pushing against* speciation (e.g. it strongly prefers a single dominant
  lineage). If that happens, it'd be a signal to switch — but cross that
  bridge later.

### Loose ends

- The current loss optimizes *current-step* territory. With resources and
  long horizons, what we actually want is *long-horizon* viability. Two
  cheap fixes without abandoning BPTT: (a) increase `steps_per_update` and
  weight later steps more in the loss; (b) periodically score lineages on
  whole-run fitness (terminal alive mass, resource-niche occupancy) and use
  that score as a selection signal at the pool / lineage level — gradients
  stay short, selection captures the long horizon.
- If we go hybrid, the optimizer-state issue from option (b) becomes more
  pointed: forks need clean optimizer buffers to avoid inheriting stale
  Adam/RMSProp momentum from an inactive slot. Already noted there.

---

## Open directions (not yet picked)

- Allopatric seed code in `src/world.py:_init_pool` (geographic separation at
  init).
- Resource channels (prerequisite for option (a) bifurcation to be stable).
- Option (a) polymorphism experiment: `n_ncas=1` + expanded `cell_state_dim`
  + resources.
- Option (b) weight-set forking as described above.
