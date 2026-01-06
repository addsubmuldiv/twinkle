import sys
from typing import Literal, Union, List, Dict, Optional, Required, Any
if sys.version_info <= (3, 11):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class ToolCall(TypedDict, total=False):
    tool_name: str
    arguments: str


class Tool(TypedDict, total=False):
    server_name: str
    tool_name: Required[str]
    description: Required[str]
    parameters: Dict[str, Any]


class Message(TypedDict, total=False):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Union[str, List[Dict[str, str]]]
    tool_calls: List[ToolCall]
    tool_call_id: Optional[str]
    name: Optional[str]
    reasoning_content: str
    id: str
    partial: bool
    prefix: bool
    completion_tokens: int
    prompt_tokens: int
    api_calls: int
