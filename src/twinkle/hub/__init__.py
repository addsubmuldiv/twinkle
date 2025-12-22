from .hub import MSHub, HFHub
from ..infra import prepare_one

ms = prepare_one(MSHub)
hf = prepare_one(HFHub)

__all__ = ['ms', 'hf']