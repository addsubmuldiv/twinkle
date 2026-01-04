from typing import List, Dict, Any, TypedDict, Optional
from .message import Message, Tool


class Trajectory(TypedDict, total=False):
    messages: List[Message]
    tools: List[Tool]
    generation_config: Dict[str, Any]
    experts: Any
    rewards: List[float]
    user_data: Dict[str, Any]
