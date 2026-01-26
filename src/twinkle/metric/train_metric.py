# Copyright (c) ModelScope Contributors. All rights reserved.
import time

from .base import Metric


class TrainMetric(Metric):
    """The training metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
    """

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.lr = ''
        self.step = -1
        self.time = time.time()

    def accumulate(self, inputs, outputs):
        lr = outputs.get('lr')
        if isinstance(lr, list):
            lr = ','.join(lr)
        self.lr = lr
        self.step = outputs.get('step')

    def reset(self):
        pass

    def calculate(self):
        results = {}
        if self.lr is not None:
            results['lr'] = self.lr
        if self.step is not None:
            results['step'] = self.step
            interval = time.time() - self.time
            speed = self.step / interval
            results['speed'] = f'{speed:.2f} steps/s'
        return results