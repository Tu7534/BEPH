import os
import glob
import torch
from torch_geometric.data import Dataset
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class MorphologicalDropEdge(object):
    def __init__(self, p_overall=0.4):
        self.p_overall = p_overall

    def __call__(self, data: Data) -> Data:
        aug_data = data.clone()
        if aug_data.edge_attr is None:
            return aug_data
        edge_attr = aug_data.edge_attr.squeeze()
        edge_index = aug_data.edge_index

        row, col = edge_index
        mask_upper = row <= col

        w_upper = edge_attr[mask_upper]
        row_upper = row[mask_upper]
        col_upper = col[mask_upper]

        w_min = edge_attr.min()
        w_max = edge_attr.max()
        if w_max - w_min < 1e-6:
            s_ij = w_upper
        else:
            s_ij = (w_upper - w_min) / (w_max - w_min)

        p_drop = self.p_overall * (1.0 - s_ij)
        p_drop = torch.clamp(p_drop, min=0.0, max=1.0)

        p_keep = 1.0 - p_drop
        keep_mask = torch.bernoulli(p_keep).to(torch.bool)

        kept_row = row_upper[keep_mask]
        kept_col = col_upper[keep_mask]
        kept_attr = w_upper[keep_mask]
        kept_edge_index = torch.stack([kept_row, kept_col], dim=0)

        new_edge_index, new_edge_attr = to_undirected(
            kept_edge_index,
            kept_attr,
            num_nodes=data.num_nodes
        )

        aug_data.edge_index = new_edge_index
        aug_data.edge_attr = new_edge_attr
        return aug_data


def apply_feature_masking(x, drop_prob=0.2):
    mask = torch.rand(x.size(0), device=x.device) > drop_prob
    x_masked = x.clone()
    x_masked[~mask] = 0.0
    return x_masked


class ContrastiveGraphDataset(Dataset):
    """Loads .pt torch_geometric Data objects and returns (orig, corrupted) pairs."""
    def __init__(self, root_dir, p_overall=0.4, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.augmentor = MorphologicalDropEdge(p_overall=p_overall)
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        path = self.file_list[idx]
        data_orig = torch.load(path)
        # basic sanity check
        if not hasattr(data_orig, 'x') or data_orig.x is None:
            raise ValueError(f"Bad data file: {path}")
        if data_orig.x.shape[1] < 1:
            raise ValueError(f"Bad feature dim in: {path}")

        data_corr = self.augmentor(data_orig)
        data_orig.x = apply_feature_masking(data_orig.x, drop_prob=0.1)
        data_corr.x = apply_feature_masking(data_corr.x, drop_prob=0.2)
        return data_orig, data_corr


def make_loaders(data_dir, batch_size=4, p_overall=0.4, split=0.8, num_workers=0):
    full_dataset = ContrastiveGraphDataset(data_dir, p_overall=p_overall)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No .pt files found in {data_dir}")
    train_size = int(split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return full_dataset, train_loader, val_loader
