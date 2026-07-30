import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO)


class ICUDataset(Dataset):
    """
    Unified dataset outputting plan.md §5.1 standardized tensor format:
        values   : (T, V) float32
        mask     : (T, V) bool
        abg_mask : (T,)   bool
        c_mask   : (T, 5) bool
        label    : (T,)   float32  (hourly sepsis or broadcast binary)
        site_id  : int
    """

    def __init__(self, samples_list):
        self.samples = []
        for s in samples_list:
            def _t(v, dtype):
                if isinstance(v, torch.Tensor):
                    return v.clone().to(dtype)
                return torch.tensor(v, dtype=dtype)

            hourly_label = _t(s["label"], torch.float32)
            sepsis_binary = torch.tensor(float(hourly_label.sum() > 0), dtype=torch.float32)
            self.samples.append({
                "x": _t(s["values"], torch.float32),
                "mask": _t(s["mask"], torch.bool),
                "abg_mask": _t(s["abg_mask"], torch.bool),
                "c_mask": _t(s["c_mask"], torch.bool),
                "label": hourly_label,
                "sepsis": sepsis_binary,
                "site_id": _t(s["site_id"], torch.long),
                "patient_id": s.get("patient_id", ""),
                "stay_id": s.get("stay_id", 0),
                "mortality": _t(s.get("mortality", 0), torch.float32),
                "los_3d": _t(s.get("los_3d", 0), torch.float32),
                "env_id": _t(s["site_id"], torch.long),
                "hospital_id": _t(s.get("hospital_id", 0), torch.long),
                "unit_type": s.get("unit_type", "unknown"),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def get_labels(self, task_name="mortality"):
        return np.array([s[task_name].item() for s in self.samples])

    def coverage_report(self):
        from .variables import ABG_VARIABLES
        total = len(self.samples)
        if total == 0:
            logging.info("Empty dataset")
            return

        n_abg = sum(1 for s in self.samples if s["abg_mask"].any())
        n_hemo = sum(1 for s in self.samples if s["c_mask"][:, 0].any())
        n_sepsis = sum(1 for s in self.samples if s["sepsis"].item() > 0)
        n_mort = sum(1 for s in self.samples if s["mortality"].item() > 0)

        logging.info(f"Dataset: {total} patients")
        logging.info(f"  Hemodynamic data: {n_hemo}/{total} ({100*n_hemo/total:.1f}%)")
        logging.info(f"  ABG data:         {n_abg}/{total} ({100*n_abg/total:.1f}%)")
        logging.info(f"  Sepsis positive:  {n_sepsis}/{total} ({100*n_sepsis/total:.1f}%)")
        logging.info(f"  Mortality:        {n_mort}/{total} ({100*n_mort/total:.1f}%)")


def make_patient_split_loaders(dataset, train_frac=0.8, batch_size=16, seed=42,
                                stratify_key="mortality"):
    labels = dataset.get_labels(stratify_key)

    if len(np.unique(labels)) < 2:
        n = len(dataset)
        n_train = int(n * train_frac)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        train_ds = Subset(dataset, idx[:n_train].tolist())
        val_ds = Subset(dataset, idx[n_train:].tolist())
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - train_frac, random_state=seed
        )
        idx = np.arange(len(labels))
        train_idx, val_idx = next(sss.split(idx, labels))
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logging.info(f"Split: train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader


def make_site_split_loaders(dataset, train_sites, test_sites, batch_size=16,
                             val_frac=0.2, seed=42, stratify_key="mortality"):
    train_indices = []
    test_indices = []

    for i, s in enumerate(dataset.samples):
        sid = s["site_id"].item()
        if sid in train_sites:
            train_indices.append(i)
        elif sid in test_sites:
            test_indices.append(i)

    labels = np.array([dataset.samples[i][stratify_key].item() for i in train_indices])
    if len(np.unique(labels)) >= 2 and len(train_indices) > 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        idx_arr = np.arange(len(train_indices))
        tr_sub, va_sub = next(sss.split(idx_arr, labels))
        tr_final = [train_indices[i] for i in tr_sub]
        va_final = [train_indices[i] for i in va_sub]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_indices))
        n_val = max(1, int(len(train_indices) * val_frac))
        va_final = [train_indices[i] for i in perm[:n_val]]
        tr_final = [train_indices[i] for i in perm[n_val:]]

    train_loader = DataLoader(
        Subset(dataset, tr_final), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        Subset(dataset, va_final), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=batch_size, shuffle=False
    )

    logging.info(
        f"Site split: train={len(tr_final)}, val={len(va_final)}, "
        f"test(OOD)={len(test_indices)}"
    )
    return train_loader, val_loader, test_loader
