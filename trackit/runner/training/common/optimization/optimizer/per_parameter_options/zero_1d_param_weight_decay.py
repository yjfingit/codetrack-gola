from typing import Dict, Optional, Tuple
import warnings

import torch.nn as nn

from ._common import filter_out_params_by_rule_, get_common_per_parameter_optimizer_options


def apply_zero_1d_param_weight_decay_rule_(rule: dict, base_lr: float,
                                           base_weight_decay: Optional[float],
                                           module_parameters: Dict[str, nn.Parameter],
                                           optimizer_param_dict: list,
                                           decay_parameter_names: Tuple[str, ...]):
    """Collect the parameters that must NOT be decayed, into one group with ``weight_decay=0``.

    "Must not be decayed" is ``name not in decay_parameter_names``, where the decay list is
    built from ``ndim > 1`` and non-norm modules -- i.e. exactly the 1-D parameters (biases)
    plus every weight inside a norm layer.

    The rule's own filters (``name_regex`` / ``name_prefix`` / ``name`` / ``ndim``) are honoured
    through ``filter_out_params_by_rule_``, and the rule may carry its own ``lr``.

    Why this matters (measured, not theoretical).  ``filter_out_params_by_rule_`` *removes* what
    it matches from the pool, so an unfiltered rule here is a catch-all: it drained the 1-D and
    norm parameters of the *entire* model into a single group.  With CodeTrack enabled there are
    53 such parameters (``refiner.norm.weight``, ``denoiser.norm1/2.{weight,bias}``,
    ``head.*.bias``, ``lora.*.bias``, ``token_type_embed.bias``, ...), so a config that scoped
    the rule with ``name_regex: 'codetrack\\.'`` and ``lr: 1.e-4`` silently put the LoRA and head
    biases there too -- the per-group learning rates of every 1-D parameter were wrong, and the
    ``^head\\.`` / trailing rules matched nothing at all.  Ignoring the filter also made the
    ``name_regex`` a lie to whoever read the config.

    Upstream ``config/GOLA/run.yaml`` writes the rule with no filters at all, which still passes
    every parameter and therefore produces exactly the previous behaviour.

    Only ``ndim < 2`` (and norm-layer) parameters can belong to this group, so a matching
    *decay-eligible* parameter is deliberately left in the pool for a later rule rather than
    being dropped from the optimizer: silently losing a tensor whose ``requires_grad`` is True is
    exactly the class of failure this file has already been fixed for once.
    """
    # ``filter_out_params_by_rule_`` already *removed* the matches from the pool, so the
    # selection must be taken from what it returned, not popped again from ``module_parameters``.
    named_params = filter_out_params_by_rule_(rule, module_parameters)
    one_dim_params, left_behind = [], []
    for name, param in named_params.items():
        if name in decay_parameter_names:
            left_behind.append((name, param))
        else:
            one_dim_params.append(param)

    # The restore MUST happen before the empty-set early return.  ``filter_out_params_by_rule_``
    # already popped the matches out of the pool, so if this rule matched *only* decay-eligible
    # parameters (e.g. CodeTrack's 3-D ``codetrack.memory.base_prior`` reaching the trailing
    # catch-all rule) and the guard returned first, that parameter would be in no group at all:
    # ``requires_grad`` is True but nothing ever updates it, and the parameter-count check in the
    # preflight is the only thing that notices.  That was measured: 1405 of 1406 tensors owned.
    if left_behind:
        for name, param in left_behind:
            module_parameters[name] = param
        warnings.warn(
            f"zero_1d_param_weight_decay rule matched {len(left_behind)} decay-eligible parameter(s) "
            f"({[n for n, _ in left_behind][:3]}); they were returned to the pool. Add "
            f"`ndim: [0, 1]` to the rule (or order the tensor rules first) to avoid the ambiguity.",
            stacklevel=2)

    if len(one_dim_params) == 0:
        return

    optimizer_options = get_common_per_parameter_optimizer_options(rule, base_lr, base_weight_decay)
    # weight decay is the whole point of this rule type: it always wins over the rule's own
    # ``weight_decay`` / ``weight_decay_mult``.
    optimizer_options['weight_decay'] = 0.
    optimizer_param_dict.append({'params': tuple(one_dim_params), **optimizer_options})
