from typing import Dict, Optional, Tuple

import torch.nn as nn

from ._common import _Filter, get_common_per_parameter_optimizer_options


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

    A *decay-eligible* parameter is simply never considered here: the check comes first, so it
    stays in the pool for a later rule.  Silently dropping a tensor whose ``requires_grad`` is
    True is exactly the class of failure this file has already been fixed for once.
    """
    # The filter is evaluated *before* anything is popped, so a decay-eligible parameter is never
    # removed from the pool in the first place.  An earlier revision filtered via
    # ``filter_out_params_by_rule_`` (which pops as it matches) and then had to push the
    # non-decayed matches back -- and if the empty-set early return ran first, they were lost
    # silently (measured: 1405 of 1406 tensors owned; ``codetrack.memory.base_prior`` was never
    # updated).  Checking inside the loop removes that failure mode entirely, and keeps the code
    # path for an *unfiltered* rule -- which is what upstream ``config/GOLA/run.yaml`` uses -- as
    # close to the original implementation as possible.
    parameter_filter = _Filter(rule)
    one_dim_params = []
    for name in list(module_parameters.keys()):
        if name in decay_parameter_names:
            continue
        param = module_parameters[name]
        if not parameter_filter(name, param):
            continue
        one_dim_params.append(module_parameters.pop(name))
    if not one_dim_params:
        return

    optimizer_options = get_common_per_parameter_optimizer_options(rule, base_lr, base_weight_decay)
    # weight decay is the whole point of this rule type: it always wins over the rule's own
    # ``weight_decay`` / ``weight_decay_mult``.
    optimizer_options['weight_decay'] = 0.
    optimizer_param_dict.append({'params': tuple(one_dim_params), **optimizer_options})
