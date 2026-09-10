# hocloop-proxy-model

A structured symbolic proxy model for closed-loop geothermal simulator studies.

The pipeline learns a closed-form surrogate for a multi-output simulator. Rather
than fitting one model per output, it first searches the *target space* for exact
algebraic relations, learns only the outputs that are irreducible, and
reconstructs the rest analytically. The surrogate therefore satisfies those
relations by construction instead of approximately.

## Method

The model composes four stages. Each is fitted from training data only, so the
whole object can be dropped into a cross-validation loop without leaking
information between folds.

**1. Target-structure discovery** (`modeling/structure`)

Searches every ordered pair of targets for an exact relation in four families:
a fixed multiple, an affine map, a product or quotient that collapses onto a
polynomial of the features, and a target that is a closed-form function of the
features alone. Feature *aggregates* are discovered too: features entering a
relation with the same least-squares coefficient are summed into one variable,
which is how a quantity such as total drilled length is recovered without being
supplied.

A candidate is accepted only if it reproduces the target to within a strict
relative-error tolerance (default `1e-6`), so an accepted identity is exact up
to the numerical precision of the source data. Variables are then eliminated
greedily for as long as the fit stays exact, because a degree-two polynomial in
four variables can absorb a relation that genuinely involves one.

Two further guards stop a flexible fit being mistaken for an identity on a small
design, where an in-sample residual of zero carries almost no information:

- a polynomial is only fitted when there are at least `min_samples_per_parameter`
  (default 10) samples per coefficient, and
- an exact candidate is refitted on 70% of the data and must stay exact on the
  withheld 30%. A real algebraic relation is a property of the system, so
  coefficients estimated from a subset reproduce the whole; an overparameterised
  fit does not.

On this data set the identities survive both guards down to 50 samples, so the
guards cost nothing here and bound the failure mode elsewhere.

The smallest set of targets that reconstructs all the others is selected, with
ties broken by a cross-validated learnability score.

**2. Dimensional reduction** (`modeling/features`)

Builds a Buckingham-Pi basis from the declared units. Classical dimensional
analysis leaves the choice of repeating variables to the analyst; here every
admissible repeating set is enumerated and scored on the data by the log-scale
spread of its response group, which turns a modelling convention into a
measurable decision.

The response group is then refined in closed form. Any product
`Pi_0 * prod(Pi_i ** c_i)` is dimensionless, and the exponents minimising the
spread of the result are the negated coefficients of a least-squares fit of
`log Pi_0` on the `log Pi_i`. The refined group is the dimensionally consistent
power law that best explains the response, obtained without search; exponents
are rounded to simple fractions when that costs little.

**3. Regression of the residual** (`modeling/learners`)

A pluggable learner fits the dimensionless departure from the power law on a log
scale. Because stages 1, 2 and 4 are held fixed, swapping the learner produces a
controlled comparison rather than a comparison of unrelated pipelines.

`symbolic_residual`, the default, solves the linear part exactly and spends the
whole genetic-programming budget on what the linear fit cannot express. A search
started from random expressions has to rediscover the linear part first and
usually runs out of budget, which is why a plain symbolic search can score below
the least-squares fit it should dominate.

**4. Reconstruction** (`modeling/structure`)

Predictions of the independent targets are expanded to the full target set
through the discovered identities.

## Results on the HOCLOOP design

The bundled study is a 4948-point Latin hypercube over seven parameters with
three outputs.

Structure discovery reduces the three targets to one. The relations found are
exact to `8e-8` relative error:

```
LCOH_i = 21.4015571459 * LCOH
LCOH   = (7.86601029e-07 * L**2 + 3.19315067e-03 * L + 2.06239302) / w_out,  L = depth + l_horiz
```

The second is a capital cost quadratic in total drilled length divided by
thermal output: the definition of a levelised cost, recovered from data. Only
`w_out` is learnt.

Dimensional analysis selects the power law

```
w_out ~ k_rock * gradT * (depth + l_horiz) * depth
```

which removes 62% of the log-scale spread of the response before any fitting.

Strict Pi reduction was measured to lose real information on this design
(log-scale R² 0.83 against 0.96 for the raw features). This is expected and
reported rather than hidden: exact dimensional similarity requires every
governing dimensional quantity to appear in the parameter table, and this study
holds pipe geometry, fluid properties and the simulation horizon fixed. The
default feature space therefore keeps the raw features alongside the groups,
retaining the valuable half of the reduction — a unit-invariant response scale —
without forcing an assumption the data does not support.

Two parameters, `rho_cp_rock` and `u_inf`, are inert: removing both changes
accuracy by less than 0.001 R².

## Validation

`modeling/validation` runs three protocols:

- **Repeated stratified k-fold**, stratified on binned log response so folds stay
  comparable across four orders of magnitude of output.
- **Extrapolation shell**: train on the interior of the hypercube, test on the
  outer shell. Proxies are used outside the region they were fitted on, and a
  random split cannot see that failure mode.
- **Learning curve**: accuracy against training-set size. Every training point is
  one simulator run, so this measures how many runs the proxy saves.

`modeling/conformal` adds split/CV+ prediction intervals calibrated on the log
scale, which makes them constant *relative* intervals in physical units.
Intervals propagate to derived targets by evaluating the exact identity at the
endpoints, so they inherit coverage with no assumed error budget.

### Bounded predictions

The learner predicts a log quantity that the inverse transform exponentiates, so
a moderate error there becomes a factor of many orders of magnitude in the
output. On a small training subset the dimensional stage can settle on a poor Pi
basis whose response group spans several log units; the fitted model then emitted
values from `1e-4` to `3e6` for a target observed between `5` and `2.3e4`. Since
the derived targets are reconstructed by division, one such value corrupts every
target at once — a learning-curve point at 100 samples scored R² = −6.4e8.

Every target, learnt or reconstructed, is therefore clipped to its observed
training range widened by `prediction_range_margin` (default 10, a
multiplicative factor for positive targets). Bounding the latent instead is not
enough, because that bound is derived from the very spread that has gone wrong.
Set the margin to `inf` to disable.

Measured effect on the worst learning-curve fold:

| guard | worst R² |
|---|---|
| none | −1.3e9 |
| latent clipped | −1569 |
| learnt target clipped | −18.3 |
| all targets clipped | see `learning_curve.csv` |

### Metrics

Raw R² is a poor summary of a target with skew 60: the sum of squares is
dominated by the largest few samples, so a model accurate in relative terms
everywhere can still score negatively. `r2_log_score` (R² on logs) and
`mean_absolute_percentage_error` are reported alongside it and are the numbers to
quote for these targets. The residual negative raw R² values at small training
sizes are a property of the metric, not a defect in the model.

## The exported formula is the model

`closed_form()` and `final_model.json` are the numbers a paper quotes, so they
carry a hard guarantee: **evaluating the printed formula on the raw feature
columns reproduces the model's own predictions to machine precision.**

That does not come for free, and three things had to be right for it to hold.

*Standardisation.* The genetic search evolves on standardised features and a
standardised target, so `str(best_individual)` is a function of scaled
variables. Printed next to unscaled column names it silently means something
else — it was out by a factor of 7 in absolute terms. Each variable is now
back-substituted as `(v - mean) / std` and the result rescaled by the target
statistics, so the exported expression is a function of the columns it names.
The scaled form is still exported as `expression_scaled`, along with the
standardisation constants, for reproducibility.

*Precision.* Coefficients are printed at 17 significant digits, not 6. Six
digits left a relative error of ~1e-5 in the reconstructed prediction.

*Every step of the inverse transform.* The Duan smearing factor is a
multiplicative constant of a few percent applied on the way back from log space,
and omitting it from the formula produced a silent 2.5% bias. It now appears
explicitly. The only step deliberately omitted is the clip to the observed
target range, which binds on no fitted sample.

`closed_form_report()` (written into `final_model.json`) ships the definitions
needed to evaluate the formula: the response scale, the smearing factor, the Pi
group definitions, and any discovered derived variable such as
`depth_plus_l_horiz = depth + l_horiz`. With those, the artefact alone
reconstructs the model:

```
w_out  = (k_rock * gradT * depth_plus_l_horiz * depth) * 1.0466087039198544 * exp(...)
LCOH   = (7.8660102900587517e-07*(depth + l_horiz)**2 + ...) / w_out
LCOH_i = (1.6834486840897337e-05*(depth + l_horiz)**2 + ...) / w_out
```

Verified against `predictions_full_fit.csv` from raw features alone: max
relative error 2.9e-14 on all three targets.

Note that `/` is printed as ordinary division, so the small-denominator guard
used during evolution is not reproduced; it differs only where a denominator is
within 1e-12 of zero. Pareto-front entries are reported as `expression_scaled`,
since they are alternatives considered rather than the exported model.

## Hyperparameter optimisation

`modeling/tuning.py` adds an Optuna search, off by default:

```bash
uv run src/main.py ... --tune --n-trials 40 --inner-splits 3
```

The search runs **inside** `TunedProxyModel.fit`, using only the data that call
receives. This is the whole design: every protocol in `modeling/validation`
already refits the model on each training split, so a model that tunes itself
during `fit` produces proper nested cross-validation for free, and the search
can never see the fold it is scored on. Tuning once on the full design and then
cross-validating the winner would report the score of a model chosen with
knowledge of the test data.

Two groups can be searched:

- **Learner hyperparameters** — GP budget and pressure (`population_size`,
  `generations`, `max_tree_height`, `parsimony_coefficient`, `n_islands`,
  operator set, and for `symbolic_deap` also tournament size and the crossover
  and mutation rates); PySR's iteration and size budget; and the usual knobs of
  the black-box baselines.
- **Pipeline choices** (`--tune-pipeline`) — the feature space and the
  power-law refinement switch, which are hyperparameters like any other.

The objective defaults to `r2_log_score` rather than raw R², because on a target
spanning four orders of magnitude the raw score is decided by a handful of
extreme points, and tuning against a noisy objective mostly fits the noise.
Trials are pruned with a median pruner on the inner folds, and a combination
that cannot be fitted is pruned rather than aborting the run.

The ablation deliberately does **not** tune: it isolates the contribution of a
pipeline stage, and letting the search compensate for a removed stage would
confound exactly what the table is meant to measure.

Cost is `n_trials` × `inner_splits` fits per model fit, and the protocols fit
many models — so tuning the genetic-programming learners over a full study is
expensive. Use `--tune-learners` to restrict it, and note that `power_law_ols`
has nothing to tune and is skipped automatically.

Best hyperparameters and the inner score are written to `hyperparameters.json`
and included in `final_model.json` under `tuning`.

## Design rules

`modeling/design_rules` differentiates the fitted closed form. For a levelised
cost of the form `C(L) / w_out` with `w_out` growing as `L**p`, setting the
logarithmic derivative to zero gives `L * C'(L) - p * C(L) = 0`, solved exactly
for the optimal drilled length. A black-box surrogate cannot supply this.

The exponent `p` is not a single number. The fitted power law
`w_out ~ k_rock * gradT * L * depth` contains the total length and the vertical
depth as separate factors, so the response to an extra metre of well depends on
which section is extended, and one rule for "length" conflates two different
design decisions. A rule is therefore derived per section:

| extend | p | condition | result |
|---|---|---|---|
| `depth` | 2 | `-b*L - 2c = 0` | no positive root: LCOH monotone decreasing, deeper always pays |
| `l_horiz` | 1 | `a*L² - c = 0` | interior minimum at `L* = sqrt(c/a)` = 1619 m |

Deepening raises both length and temperature drive, so it pays throughout the
sampled range. Extending horizontally raises length alone and has a genuine
optimum — but at 1619 m that optimum sits *below* the sampled range
[2025, 9883] m, so within the design actually explored the horizontal section
should be as short as possible. Rules are flagged with `within_sampled_range`
so an extrapolated root is never read as an observed optimum.

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv)

## Installation

```bash
uv sync
```

PySR needs a working Julia. If `import pysr` fails with a missing `sys.dll`, a
Microsoft Store stub named `julia` is shadowing the real install; point
`juliapkg` at the real one once:

```bash
PYTHON_JULIAPKG_EXE="$HOME/.julia/juliaup/julia-<version>/bin/julia.exe" uv run python -c "import pysr"
```

The path is cached afterwards. PySR is optional: it is one benchmark entry and
its absence is reported rather than allowed to abort the suite.

## Usage

```bash
uv run src/main.py \
  --features-file data_source/lhc_parameters.csv \
  --targets-file  data_source/lhc_results.csv \
  --units-file    data_source/units.json \
  --output-path   results/study
```

Add `--quick` for a smoke run with a small genetic-programming budget.

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--features-file` | yes | Feature CSV, with header |
| `--targets-file` | yes | Target CSV, with header, one row per feature row |
| `--output-path` | yes | Directory for all artefacts |
| `--units-file` | no | Units JSON; defaults to the built-in HOCLOOP units. Without units the dimensional stage is switched off rather than guessed at |
| `--learners` | no | Learners in the benchmark table |
| `--primary-learner` | no | Learner used for the ablation, learning curve, intervals and reported formula (default `symbolic_residual`) |
| `--cv-splits`, `--cv-repeats` | no | Cross-validation budget |
| `--shell-fraction` | no | Fraction of each feature range treated as the extrapolation shell |
| `--conformal-alpha` | no | 1 − nominal coverage (default 0.1) |
| `--generations`, `--population-size`, `--max-tree-height`, `--n-islands` | no | Genetic-programming budget |
| `--parallel-islands` | no | Evolve islands in worker processes |
| `--skip-*` | no | Skip benchmarks, ablation, learning curve, conformal or design rules |
| `--quick` | no | Small budget smoke run |

### Units file

```json
{
  "features": {"k_rock": "W/m/K", "depth": "m"},
  "targets":  {"w_out": "W"}
}
```

Symbols `kg m s K A mol cd N Pa J W`, joined with `*` and `/`, optional integer
powers with `^`, and `1` for dimensionless. Only the flat form is accepted, so a
bracketing mistake cannot pass silently.

## Output artefacts

| File | Contents |
|---|---|
| `target_structure.json` | Discovered identities, residuals, independent targets |
| `dimensional_analysis.json` | Pi basis, repeating set, refined power law |
| `final_model.json` | Full fit details and the closed form per target |
| `predictions_full_fit.csv` | Actual and predicted values for every target |
| `benchmark_learners.{json,csv}` | Learner comparison, cross-validated and on the shell |
| `ablation.{json,csv}` | Pipeline stages switched off one at a time |
| `learning_curve.csv` | Accuracy against number of simulator runs |
| `conformal_intervals.json` | Calibrated radius and empirical coverage per target |
| `design_rules.json` | Elasticities and the analytic optimal-length rule |

## Logging

Logs go to the console and to `hocloop-proxy-model.log` in the working directory.
