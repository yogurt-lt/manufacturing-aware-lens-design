# Method and interpretation

The input is an optical design in DeepLens JSON format. Catalog-material proposals
form an outer discrete search. Each proposal first undergoes nominal optical
parameter recovery, then differentiable manufacturing refinement. The beam keeps
multiple candidate material combinations for subsequent rounds. This is inspired
by substitution/reoptimization workflows, not a reproduction of Zemax Hammer.

Manufacturing refinement samples surface and assembly errors in the ray-tracing
model. A smooth sigmoid approximation to pass/fail supplies gradients. Relative
blank-material expenditure and process assumptions are combined, and the objective
uses expected relative accepted-unit cost C / max(soft pass fraction, floor).
This assumes no rework, constant per-attempt expenditure and scrapped failures.

RMS is the ray-intercept root-mean-square merit implemented by the code. Q90 is
its 90th percentile across sampled manufacturing errors: 90% of samples have
merit no larger than Q90. Lower Q90 indicates better tolerance performance in
that model; it does not by itself establish a measured production yield.
Soft yield is a training approximation; independently sampled hard pass fraction
uses a declared RMS threshold. Neither is a measured manufacturing yield.

Search score combines normalized accepted-unit cost, normalized Q90, pass
shortfall and (optionally) inverse normalized MTF. Q90 weight affects outer
candidate selection, not a directly differentiated Q90 loss. Demonstration
weights and thresholds are illustrative; paper settings are separate.

MTF is estimated from the Fourier magnitude of geometric ray intercepts, while
an optional separate geometric PSF diagnostic is also available. Diffraction,
pixel response and experimental MTF are not represented by the training estimate.
The public short demo uses RMS only. Paper training settings use central-field
50 lp/mm; engineering full-field 93 lp/mm results are a separate evaluation.

Core code map:
- `deeplens/geolens_pkg/manufacturing.py`: perturbations, pass and cost objectives.
- `deeplens/geolens_pkg/differentiable_mtf.py`: geometric MTF gradients.
- `8_manufacturing_design.py`: single design optimization and independent samples.
- `17_manufacturing_expert.py`: optical-neighbor material proposals.
- `20_hammer_beam_search.py`: multi-round material/parameter optimization.

Legacy internal field names containing `proxy` are retained for compatibility;
the scientific term for reported cost is relative manufacturing cost index.
