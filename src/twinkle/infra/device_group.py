from dataclasses import dataclass
from typing import Union, List


@dataclass
class DeviceGroup:

    name: str
    ranks: Union[List[int], int]
    device_type: str
