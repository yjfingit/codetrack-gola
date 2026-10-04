from dataclasses import dataclass, field
from typing import Optional

from trackit.core.runtime.build_context import BuildContext
from trackit.data import HostDataPipeline
from . import ExtraTransform, ExtraTransform_DataCollector


def build_plugins(transform_config: dict, config: dict, build_context: BuildContext):
    register = PluginRegistrationHelper()
    if 'plugin' in transform_config:
        for plugin_config in transform_config['plugin']:
            if plugin_config['type'] == 'box_with_score_map_label_generation':
                from .box_with_score_map_label_gen.builder import build_box_with_score_map_label_generator
                register.register(*build_box_with_score_map_label_generator(config))
            elif plugin_config['type'] == 'template_foreground_indicating_mask_generation':
                from .template_foreground_indicating_mask_gen.builder import (
                    build_template_feat_foreground_mask_generator)
                register.register(*build_template_feat_foreground_mask_generator(config))
            elif plugin_config['type'] == 'image_corruption':
                # model-side block-1 corruption: attacks the normalised search crop and
                # exposes the per-channel mask so the diagnosis head sees image damage too,
                # not just the token-level erasure it had been trained on alone
                from .image_corruption import build_image_corruption_collector
                register.register(data_collator=build_image_corruption_collector(config))
    return register.extra_transforms, register.extra_data_collators, register.extra_host_data_pipelines


@dataclass(frozen=True)
class PluginRegistrationHelper:
    extra_transforms = []
    extra_data_collators = []
    extra_host_data_pipelines = []

    def register(self, transform: Optional[ExtraTransform] = None,
                 data_collator: Optional[ExtraTransform_DataCollector] = None,
                 host_data_pipeline: Optional[HostDataPipeline] = None):
        if transform is not None:
            self.extra_transforms.append(transform)
        if data_collator is not None:
            self.extra_data_collators.append(data_collator)
        if host_data_pipeline is not None:
            self.extra_host_data_pipelines.append(host_data_pipeline)
