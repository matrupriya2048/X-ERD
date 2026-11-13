from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import importlib.metadata
import platform
import numpy as np
import pandas as pd

META = {"sample_id", "sample_type", "family_label", "type_label"}

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    # A single writer per stage; replace atomically to avoid partial result logs.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def environment():
    names = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "shap", "matplotlib", "pyarrow"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in names}}

def labels(df):
    y = (pd.to_numeric(df["type_label"], errors="raise") > 0).astype(np.int8).to_numpy()
    if not np.array_equal(y, df["sample_type"].to_numpy()):
        raise ValueError("type_label and sample_type disagree")
    return y

def features(df):
    cols = [c for c in df if c not in META]
    if not cols or any(not str(c).isdigit() for c in cols):
        raise ValueError("Inputs must be declared numeric feature IDs; metadata is forbidden")
    values = df[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected finite binary behavioral indicators")
    return cols, values

@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    columns: list
    names: list
    X: np.ndarray
    y: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    metadata: pd.DataFrame
    manifest: dict

def load(data):
    data = Path(data)
    paths = {k: data / v for k,v in {
        "train": "MLRan_X_train_RFE.csv", "test": "MLRan_X_test_RFE.csv",
        "names": "RFE_selected_feature_names_dic.json", "metadata": "mlran_dataset_metadata.csv"}.items()}
    tr,te = (pd.read_csv(paths[k]) for k in ["train", "test"])
    cols,X = features(tr)
    tc,Xt = features(te)
    if cols != tc or len(cols) != 483: raise ValueError("Official 483-feature order mismatch")
    if tr.sample_id.duplicated().any() or te.sample_id.duplicated().any(): raise ValueError("Duplicate IDs")
    if set(tr.sample_id) & set(te.sample_id): raise ValueError("Train/test overlap")
    y,yt = labels(tr),labels(te)
    if (len(tr),len(te),int((yt==0).sum()),int(yt.sum())) != (3905,975,510,465):
        raise ValueError("Official split or true label counts differ from benchmark")
    names = json.loads(paths["names"].read_text(encoding="utf-8"))
    return Dataset(tr,te,cols,[names[c] for c in cols],X,y,Xt,yt,pd.read_csv(paths["metadata"]),
        {k:{"file":p.name,"sha256":digest(p)} for k,p in paths.items()})
