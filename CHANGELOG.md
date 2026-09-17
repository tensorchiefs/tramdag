## Unreleased

### BREAKING CHANGE

- code reading `Term.term` reads `Term.name`.
- a spec dict or YAML written with `effect:` no longer
loads; rename the key. Checkpoints saved before this commit need the same
rename inside their spec dict.
- fit(vc_ehat=) removed; VC(center=True) must become
VC(center='<column>'); ShiftTerm.shift_value/score_columns lose the
ehat parameter; side_keys/check_side became side_columns/check_column.
- additive-CI checkpoints re-key —
nodes.*.intercept_nets.* becomes nodes.*.intercept.nets.* (parameters
proven bit-identical under the rename; smoke baseline consciously
re-recorded here; 0.4 is unreleased, refit instead of shimming).
- RestoreBest is gone — use EarlyStopping() (restore
without stopping) or EarlyStopping(patience=N).
- before_fit_callbacks / after_epoch_callbacks /
after_fit_callbacks are gone; RestoreBest.restore no longer exists
(restoration is automatic).
- fit(epochs=) is now required, and every default in the
package states a reason.

### Feat

- **spec**: a custom term serializes as its import path
- **flow**: log_prob takes a node subset
- **transforms**: a continuous node starts at its empirical marginal
- **conditioners**: optional batch norm, settable from the spec
- **fit**: the optimizer's learning rate is on record, history["lr"]
- tramdag.plots — plot_dag, plot_marginals, plot_training
- **flow**: a frame that lacks a spec column fails by name
- **callbacks**: PerNodePlateau records the epoch each node froze
- **docs**: the class UML joins the generated views
- **exps**: the blueprint — full DAG spec and every training number in YAML
- **carefl**: train on the reference's own rows — the figure gap was the data
- **rc**: step 12 — fn_shift and register_term are the extension points
- **callbacks**: RestoreBest merges into EarlyStopping
- **fit**: one callbacks= kwarg replaces the three hook lists
- Keras-shaped fit — validation_data/-split, verbose; Logger dissolves
- **notebooks**: the demo self-stops — PerNodePlateau over per_node_adam; NN naming
- **experiments**: DGP-truth columns in the CI reports; fit wall time recorded
- **flow**: init_marginals — the calibrated start as an explicit, repeatable step
- **flow**: callback lists around the loop, the recipes ship in tramdag.callbacks
- **flow**: net_input_scaling="minmax" — the reference's scale_df for the nets
- fit(marginal_init=) defaults to True
- density() — the analytic conditional density of a continuous node
- one intercept, first, written or not — the formula is canonical
- the paper's own architecture, stated in the configs, not inherited
- score ordinal counterfactuals against what is actually identifiable
- the two benchmarks are back, running, and their docs resolve again
- clean cut — simulations and data move out of the framework
- paper-aligned SI/CI constructors; transform kwargs pass straight through
- surface the hidden training knobs as optional kwargs
- additive intercepts only via allow_interaction=False
- VC takes modifiers positionally, units= everywhere, one MLP builder
- transformation syntax -- + sums, bare I, allow_interaction, basis on I

### Fix

- two defects the review of the simplification turned up
- **readouts**: shift_curve builds its grid on the flow's device
- four defects a review reproduced, each with a regression test
- **readouts**: shift_curve copies its grid, like every other entry point
- **docs**: the second consistency pass
- **docs**: a grammar slip in the ADR's revision note
- **docs**: the consistency sweep after the restructure
- **spec,flow,transforms**: five defects the review of today's work turned up
- **docs**: the site renders notebook math and docstring tables
- **docs**: the API cross-references stop overflowing their table column
- **docs**: display math must not wrap an align environment
- **docs**: the PDF build actually runs, and its tables get column widths
- **flow**: an ordinal value must be a level index everywhere
- **plots**: an empty spec and an unfitted flow fail by name
- **plots**: the freeze labels hang from the visible top of the zoomed axis
- **docs**: render the mermaid diagrams — superfences needs the mermaid fence
- **experiments**: CAREFL settles at 3000 @ 0.002 — held-out beats one noisy point
- **experiments**: the triangle net was misread — hidden is (25,25), not [2,25,25,2]
- **experiments**: VACA/CAREFL back to the reference protocols — the visual pass
- **tests**: the state-dict tripwire is Linux-only
- **rc**: review sweep — Term pickles again; the honest-claims fixes
- **callbacks**: refuse a zero start rate, note the scheduler conflict, renderer-proof anchor
- **callbacks**: review round — group-stamped rate restore, stale-val after classical fit, doc dedup
- keras-fit review round — stale-val guards, side-input strictness, trims
- **docs**: exclude hooks.py's __pycache__ from the built site
- **docs**: load MathJax's boldsymbol extension statically
- **ci,docs**: the CI comment table survives cml; math renders on every docs page
- **ci,docs**: final-round findings — group leakage, PDF dedup, py.typed
- **notebooks**: the conf-int helpers were prose — a missing cell marker
- **notebooks**: finish the classical-fit port to the 0.4 fit surface
- **flow**: callbacks fail fast, review-cycle polish across notebooks and docs
- **experiments**: the classical anchor's bounds carry the cross-machine spread
- **flow**: glorot init keeps the VC head's zero start; init="normal"; stricter fit inputs
- **ci**: pass the notebook markdown list to pandoc as an array
- **carefl**: the paper's x_obs is in CAREFL's standardized units, not the SCM's
- a source node is canonical and hashable; specs survive json; the inverse warns
- close the findings of the four-way review against main
- a "why" silenced both edges of the band check, and centers decayed unseen
- ground truth that describes the code, and bounds that can catch a regression
- each experiment takes its architecture from its own reference script
- the restructure had dropped a load-bearing hyperparameter
- the review's findings — CI never ran the area tests, and more
- two silent-then-confusing failures found comparing against main
- ls_coefficients crashed on a node mixing LS and CS terms
- the intro notebook imports logging
- load() closes the ordinal marginal-init guard too
- export ordinal_marginal_init_theta in transforms.__all__
- migrate the last 0.3 call site, unstale the spec docstring
- **ci**: execute notebooks in notebooks/, restrict to the two showcases
- **ci**: install pandoc, it is not preinstalled on the runners
- **ci**: build docs without the target ref's docs group

### Refactor

- CausalFlowDAG reads top to bottom; three small folds
- one spelling per thing across the package; prose follows the code
- a term module is constructed from its term; no build classmethods
- every term sets its name; a term holds its module class; spec imports modules
- a term names its module by import path; no registration stamp
- conditioners.py and terms.py are one module, modules.py
- **spec**: a term's spec checks and its edge ownership are two things
- node-kind branches are Node methods; no private cross-imports
- six mechanisms that existed for one situation each; Node is public
- **spec**: three spellings with one caller each are gone
- required arguments replace None defaults; cells carry a joint flag
- inline the single-use helpers
- delete the dead parameters, branches and probes
- **spec**: a term is its __init__ keywords; no dataclass machinery
- close the four consistency gaps a review named
- delete what is dead, and ask the spec instead of guessing
- **spec**: `I` names its options, and a guard pins their defaults
- **spec**: a term class carries `name`, not `term`
- **spec**: a term serializes as `term:`, not `effect:`
- **spec**: only the paper's symbols alias the term classes
- **spec**: pythonic class names; the term contract slims down
- **spec**: an effect is a Term subclass; the registry is gone
- **spec**: "basis" is the base distribution; the monotone map is the transform
- **spec**: the ordinal basis check is its own helper
- no magic — terms calibrate themselves, the start is always yours
- 1.0 is a clean cut — drop the pre-1.0 loading paths
- **rc**: lean pass + critical test review — both loops' findings
- **rc**: propensities are a column — fit(vc_ehat=) dies
- **rc**: methods stay methods — mixins replace the delegate layer
- **rc**: deletion pass — the audit's cuts, behavior identical
- **rc**: step 13 — options live per effect; the spec closes its gaps
- **rc**: step 11 — the last dispatch sites move onto the registry
- **rc**: step 10 — side inputs are a term contract
- **rc**: step 9 — the intercept slot is a term
- **rc**: step 8 — score columns come from the terms; scan takes column=
- **rc**: step 7b — shift terms own their behavior
- **rc**: step 7a — the effect registry; validators live on their entries
- **rc**: step 6 — the kind branches concentrate; marginal init is a hook
- **rc**: step 5 — read-outs move to readouts.py; shift_curve is public
- **rc**: step 4 — the fitting paths move to fitting.py
- **rc**: step 3 — the node model moves to nodes.py, verbatim
- **rc**: step 2 — node_terms dies; the term list is node.terms
- **rc**: step 1 — purely additive prep
- input_transform review round — keys live where they exist, docs de-drifted
- input_transform= per term replaces the flow-level net scaling
- **experiments**: the tuned protocol — 30-71% less training, every bound kept
- uniform (df, node) queries, loud degenerate data, no utils module
- **experiments**: the R code's do-values, triangle init, and batch 256 for CI
- **experiments**: the paper protocol and the benchmark on the hooks
- **flow**: fit is one loop — optimizer= and callback= hooks, calibrate()
- **experiments**: second review cycle — shared truths, one cs-curve helper
- **experiments**: the review cycle — one fit helper, shared plots, dead code out
- **experiments**: the paper replications follow the paper's R code 1:1
- the follow-up list — no key-set check, one L-BFGS call, a batch_size guard
- dissolve every complexity hotspot; the gates now bind everywhere
- the last of the unreachable code, and a hook that guarded nothing
- one implementation per fact, in src and in the experiments
- drop code that no caller and no branch can reach
- delete what an audit proved dead
- drop two bits of ceremony the review found
- utils takes parsed config, and collects the non-modelling helpers
- one maintained notebook set, no parked directory
- one directory per experiment area, config reader into the framework
- one self-contained script and YAML per experiment
- sweep the orphans of the special-case removal
- progress goes through module loggers, not print
- drop the special cases nothing uses
- terms, not transformation; and one flat formula shape
- one home for the parent encoding and the stroke spec
- delete term(), the string-label term factory
- **flow**: one node accessor, one unknown-node message
- strict zips, and small idiom cleanups in src
- **simulations**: one home for the shared dataset draws and docs
- **experiments**: share the plotting boilerplate
- collapse four small redundancies
- **simulations**: share the generator boilerplate
- **flow**: one tensorizer, one source-proxy builder
- **spec**: one intercept scan, and drop the unused Term.slot
- **spec**: serialize Term.options whole
- pythonic names for every term constructor
- drop every backward-compatibility path
- one _np_dtype property replaces eight dtype dances
- remove the dead names found by the review
- generic Term of effect, parents and options
- pass the VC-only Term settings as options kwargs
- drop Transformation, + sums build plain lists
- def intercept(), keep I as the exported alias
- restore the CLeaR paper replications, drop dead runner helpers
- give the runner scripts argparse CLIs
- read __version__ from package metadata
- remove unused imports
- drop the sys.path hacks in experiments
- drop the unused load_rct alias
- use td.machine_info in perf_machine


- one introduction, a fitting-API notebook, and two new guides
- remove the autoresearch guard, and gate what was ungated
- MkDocs replaces pdoc; the PDF is typeset by pandoc + XeLaTeX

## v0.3.0 (2026-06-19)

### Feat

- **flow**: opt-in calibrated warm-start init for unconditional nodes

### Refactor

- rename warm_start -> marginal_init (pure rename, no logic change)

## v0.1.0 (2026-06-11)
