

class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, base_dataset_fetcher: _BaseDatasetFetcher, batch_size: int, device_mesh: DeviceMesh):
        self.base_dataset_fetcher = base_dataset_fetcher
        self.batch_size = batch_size
        self.device_mesh = device_mesh

    def fetch(self, possibly_batched_index):
        data = self.base_dataset_fetcher.fetch(list(range(self.batch_size)))
        if not self.device_mesh:
            yield batch
        else:
            data = batch[self.device_mesh.get_slice(len(batch))]
            # No this is wrong, should maintain the same behaviour with local
            #if not data:
                # Use rank0 if data is not enough
            #    data = batch[self.device_mesh.get_slice(len(batch), 0)]
            yield data