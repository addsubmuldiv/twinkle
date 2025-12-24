from typing import overload, Type, Optional

from transformers import PreTrainedModel, PretrainedConfig

from twinkle import remote_class


@remote_class()
class TransformersModel(PreTrainedModel):

    @overload
    def __init__(self, *, model_cls: Type[PreTrainedModel], config: PretrainedConfig, remote_group, **kwargs) -> None:
        ...

    @overload
    def __init__(self, *, pretrained_model_name_or_path: str, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        ...

    def __init__(self, # noqa
                 model_cls: Optional[Type[PreTrainedModel]] = None,
                 pretrained_model_name_or_path: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 **kwargs):
        if pretrained_model_name_or_path is None:
            self.model = model_cls(config, **kwargs)
        elif model_cls :
            self.model = model_cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)

    def forward(self, *, input_ids, **kwargs):
        self.model(input_ids, **kwargs)