from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
from .sample_data_generator import build_sample_input_data_generator


def get_GOLA_build_context(config: dict):
    print('GOLA model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_GOLA_model(config, impl_advice),
                                lambda impl_advice: get_GOLA_build_string(config['model']['type'], impl_advice),
                                build_sample_input_data_generator(config))


def _unfreeze_backbone_scope(model, codetrack_config) -> None:
    """Re-enable gradient on an explicitly named slice of the otherwise-frozen DINOv2 trunk.

    ``GOLA_DINOv2.__init__`` sets ``requires_grad = False`` on every parameter before applying the
    LoRA adapters, so a stage that wants to fine-tune a couple of transformer blocks cannot do it
    purely from the optimizer configuration: ``parse_optimizer_per_params_config`` only ever sees
    parameters whose ``requires_grad`` is already True.

    Stage S3 needs exactly that (DINOv2 last-2 blocks), so the unlock is config-driven and
    narrow: ``codetrack.backbone_scope`` is a list of name prefixes, and only parameters with a
    *base* (non-adapter) name under those prefixes are unlocked.  Any other stage leaves the key
    absent and the trunk stays fully frozen, exactly as before.
    """
    if not codetrack_config:
        return
    scope = codetrack_config.get('backbone_scope') or []
    if not scope:
        return
    unlocked = []
    for name, param in model.named_parameters():
        # never touch the LoRA adapters here -- they are already trainable and are governed by the
        # adapter learning-rate group, not by the foundation group.
        if "lora" in name:
            continue
        if any(name.startswith(prefix) for prefix in scope):
            param.requires_grad = True
            unlocked.append(name)
    print(f"backbone_scope: unlocked {len(unlocked)} frozen DINOv2 parameters under {list(scope)}")
    for name in unlocked[:6]:
        print(f"  + {name}")
    if len(unlocked) > 6:
        print(f"  ... and {len(unlocked) - 6} more")


def build_GOLA_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    # When CodeTrack is switched on we must build the *training* class even in inference mode.
    # The inference branch below returns ``GOLABaseline_DINOv2``, which has no CodeTrack branch
    # and loads pre-merged LoRA weights -- so enabling the flag while still taking that branch
    # would silently evaluate a plain GOLA and ignore every new module.
    codetrack_on = bool((model_config.get('codetrack') or {}).get('enabled', False))
    if model_type == 'dinov2':
        if model_impl_suggestions.optimize_for_inference and not codetrack_on:
            from .gola_full_finetune import GOLABaseline_DINOv2
            model = GOLABaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])

            # Zekai Shao: Use the weight loading function of the inference model
            # to load the weights after merging the LoRA parameters
            if config['model']['eval']:
                for path in config['model']['weight_path']:
                    model.load_state_dict_from_file(path)
        else:
            from .gola import GOLA_DINOv2
            model = GOLA_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['lora']['r'], model_config['lora']['alpha'],
                                 model_config['lora']['dropout'], model_config['lora']['use_rslora'],
                                 codetrack_config=model_config.get('codetrack'))
            _unfreeze_backbone_scope(model, model_config.get('codetrack'))
    elif model_type == 'dinov2_full_finetune':
        from .gola_full_finetune import GOLABaseline_DINOv2
        model = GOLABaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model


def get_GOLA_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'GOLA'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string
