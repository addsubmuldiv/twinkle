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
        self.time = None

    def accumulate(self, inputs, outputs):
        if self.time is None:
            self.time = time.time()
        lr = outputs.get('lr')
        if isinstance(lr, list):
            lr = [f'{x:.2e}' for x in lr]
            lr = ','.join(lr)
        else:
            lr = f'{lr:.2e}'
        self.lr = lr
        self.step = outputs.get('step')

    def reset(self):
        pass

    def calculate(self):
        results = {}
        if self.lr is not None:
            results['last lr(by param_groups)'] = self.lr
        if self.step is not None:
            results['forward count'] = self.step
            interval = time.time() - self.time
            speed = self.step / interval
            if interval < 60:
                results['total time elapse'] = f'{interval:.0f} seconds'
            else:
                results['total time elapse'] = f'{interval/60:.1f} minutes'
            results['total avg speed'] = f'{speed:.2f} forwards/s'
        return results