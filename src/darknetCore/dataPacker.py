import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import samplers


class ForeignObjectTestDataset(Dataset):
    def __init__(self, image_root, data_csv_file, transform=None):
        assert os.path.exists(data_csv_file), f'{data_csv_file} not exists!'

        self.image_root = image_root
        # with open(data_csv_file) as csvfile:
        #     datareader = csv.reader(csvfile, delimiter=',')
        #     data = [row[0] for row in datareader]
        df = pd.read_csv(data_csv_file, names=['image_name'], na_filter=False)
        self.data = df
        # self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        image_path = os.path.join(self.image_root, self.data.iloc[idx].image_name)
        image = cv2.imread(image_path)

        height, width = image.shape[:2]
        if self.transform is not None:
            image, _ = T.apply_transform_gens(self.transform, image)

        image = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()

        return {'file_name': image_path, 'height': height, 'width': width, 'image': image}


def packData(input_csv_file, cfg):
    transform = utils.build_transform_gen(cfg, is_train=False)
    dataset = ForeignObjectTestDataset('', input_csv_file, transform=transform)
    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
