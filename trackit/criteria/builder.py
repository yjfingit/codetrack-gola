import torch.nn as nn
from trackit.core.runtime.build_context import BuildContext


def build_criterion(criterion_config: dict, build_context: BuildContext, num_total_iterations: int) -> nn.Module:
    if criterion_config['type'] == 'box_with_score_map':
        from .methods.box_with_score_map.builder import build_box_with_score_map_criteria
        return build_box_with_score_map_criteria(criterion_config)
    elif criterion_config['type'] == 'codetrack':
        # CodeTrack objective = the upstream tracking loss (corrupted + clean branches) plus
        # the architecture figure's auxiliary terms (diagnosis / recovery / alignment /
        # preservation / memory / gate / motion).  It is parameter-free: every term is a
        # function of the model output, so the whole graph lives in the model.
        from codetrack.criteria import CodeTrackCriteria
        return CodeTrackCriteria(
            w_track_corr=float(criterion_config.get('w_track_corr', 1.0)),
            w_track_clean=float(criterion_config.get('w_track_clean', 0.25)),
            lambda_diag=float(criterion_config.get('lambda_diag', 0.5)),
            lambda_q_prior=float(criterion_config.get('lambda_q_prior', 0.0)),
            lambda_diag_rank=float(criterion_config.get('lambda_diag_rank', 0.0)),
            diag_rank_margin=float(criterion_config.get('diag_rank_margin', 0.2)),
            lambda_causal_rank=float(criterion_config.get('lambda_causal_rank', 0.0)),
            causal_rank_margin=float(criterion_config.get('causal_rank_margin', 0.2)),
            lambda_rec=float(criterion_config.get('lambda_rec', 0.2)),
            lambda_gain=float(criterion_config.get('lambda_gain', 0.2)),
            gain_margin=float(criterion_config.get('gain_margin', 0.8)),
            lambda_align=float(criterion_config.get('lambda_align', 0.2)),
            lambda_pres=float(criterion_config.get('lambda_pres', 0.01)),
            lambda_mem=float(criterion_config.get('lambda_mem', 0.1)),
            lambda_gate=float(criterion_config.get('lambda_gate', 0.1)),
            lambda_motion=float(criterion_config.get('lambda_motion', 0.2)),
            lambda_trc=float(criterion_config.get('lambda_trc', 0.1)),
            diagnosis_alpha=float(criterion_config.get('diagnosis_alpha', 0.5)),
        )
    else:
        raise ValueError("unknown criterion type")
