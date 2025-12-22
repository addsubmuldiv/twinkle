from abc import abstractmethod, ABC


class Platform(ABC):

    @staticmethod
    @abstractmethod
    def visible_device_env() -> str:
        ...

    @staticmethod
    @abstractmethod
    def device_prefix() -> str:
        ...

    @staticmethod
    def get_platform(platform: str) -> 'Platform':
        if platform.upper() == "GPU":
            return GPU
        elif platform.upper() == "NPU":
            return NPU
        else:
            raise ValueError(f"Unsupported platform: {platform}")


class GPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'CUDA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'cuda'


class NPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'ASCEND_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'npu:'
