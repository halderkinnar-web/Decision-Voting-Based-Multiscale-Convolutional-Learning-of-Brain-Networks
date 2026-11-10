from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch

@dataclass
class Config:
    data_path: str = ""
    output_dir: str = "results"
    n_subjects: int = 54
    n_folds: int = 5
    scales_roi: tuple = (83, 129, 234)
    diffusion_t: float = 1.0
    beta: float = 0.5
    fc_top_bottom_m: int = 10
    sc_percentile: float = 50.0
    noise_sigma: float = 2.0
    n_augments: int = 9
    cheb_k: int = 3
    hidden_dim: int = 64
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 150
    seed: int = 42
    explain: bool = True
    device: str = "auto"

    def resolved_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "scales_roi" in d:
            d["scales_roi"] = tuple(d["scales_roi"])
        return cls(**d)
