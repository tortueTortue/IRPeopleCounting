from dataset.cifar_data_loaders import Cifar10Dataset
from dataset.imagenette import ImageNetteDataset
from dataset.mnist import MNISTDataset

class Datasetmanager:

    def get_dataset(self, dataset_name: str, batch_size: int, subset=False, subset_size=1000, test_subset_size=500):
        if dataset_name.lower() in "ImageNette".lower():
            return ImageNetteDataset(batch_size, subset, subset_size, test_subset_size)
        elif dataset_name.lower() in "Cifar10".lower():
            return Cifar10Dataset(batch_size, subset, subset_size, test_subset_size)
        elif dataset_name.lower() in "MNIST".lower():
            return MNISTDataset(batch_size, subset, subset_size, test_subset_size)

datasetManager = Datasetmanager()