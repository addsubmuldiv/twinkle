from torch.utils.data import DataLoader as TorchDataLoader

from twinkle import remote_class


@remote_class()
class DataLoader(TorchDataLoader):

    pass

