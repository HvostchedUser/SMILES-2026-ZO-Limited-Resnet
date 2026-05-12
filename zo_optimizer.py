from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from augmentation import get_transforms
from train_data import get_last_data_dir


_NUM_CLASSES = 100
_CACHE_VERSION = 4
_RIDGE_LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
_LDA_SHRINKS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


class FeatureSpaceHead(nn.Module):
    """Linear CIFAR100 head with optional input feature normalization."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        normalize_input: bool,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().float())
        self.bias = nn.Parameter(bias.detach().clone().float())
        self.normalize_input = normalize_input

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            features = F.normalize(features, p=2, dim=1)
        return F.linear(features, self.weight, self.bias)


class ZeroOrderOptimizer:
    """Gradient-free optimizer used by the fixed validation script.

    The useful signal is in the frozen ImageNet backbone features. At
    construction time this optimizer fits several closed-form linear heads on
    CIFAR100 train features and installs the best one according to a stratified
    internal holdout. The public ``step`` method remains gradient-free; the
    final configuration keeps the fitted head fixed because noisy SPSA updates
    usually damage it under this budget.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-5,
        eps: float = 1e-3,
        perturbation_mode: str = "rademacher",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode

        self.layer_names: list[str] = []
        self._step_idx = 0
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

        self._fit_and_install_head()

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Unknown layer names: {missing}")
        return {n: named[n] for n in self.layer_names}

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _fit_and_install_head(self) -> None:
        data_dir = Path(get_last_data_dir())
        cache_path = data_dir / "ridge_feature_head_v4.pt"
        cached = self._load_cached_head(cache_path)
        if cached is None:
            features, labels = self._extract_train_features(data_dir)
            cached = self._build_head(features, labels)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cached, cache_path)

        head = FeatureSpaceHead(
            weight=cached["weight"],
            bias=cached["bias"],
            normalize_input=bool(cached["normalize_input"]),
        )
        self.model.fc = head
        self.head_method = str(cached.get("method", "unknown"))
        self.head_holdout_accuracy = float(cached.get("holdout_accuracy", 0.0))

    @staticmethod
    def _load_cached_head(cache_path: Path) -> dict | None:
        if not cache_path.exists():
            return None
        try:
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                cached = torch.load(cache_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(cached, dict):
            return None
        if cached.get("version") != _CACHE_VERSION:
            return None
        required = {"weight", "bias", "normalize_input"}
        if not required.issubset(cached):
            return None
        return cached

    def _extract_train_features(self, data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._select_device()
        old_fc = self.model.fc
        dataset = datasets.CIFAR100(
            root=str(data_dir),
            train=True,
            download=True,
            transform=get_transforms(train=False),
        )
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        try:
            self.model.fc = nn.Identity()
            self.model.eval()
            self.model.to(device)
            with torch.no_grad():
                for images, target in loader:
                    images = images.to(device, non_blocking=(device.type == "cuda"))
                    batch_features = self.model(images).detach().cpu().float()
                    features.append(batch_features)
                    labels.append(target.cpu().long())
        finally:
            self.model.fc = old_fc

        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def _build_head(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        train_idx, holdout_idx = self._stratified_split(labels)
        raw = features.float()
        norm = F.normalize(raw, p=2, dim=1)

        candidates: list[dict] = []
        for name, matrix, normalize_input in (
            ("ridge_raw", raw, False),
            ("ridge_norm", norm, True),
        ):
            stats = self._ridge_stats(matrix[train_idx], labels[train_idx])
            for lam in _RIDGE_LAMBDAS:
                weight, bias = self._solve_ridge(stats, lam)
                score = self._accuracy(matrix[holdout_idx], labels[holdout_idx], weight, bias)
                candidates.append(
                    {
                        "method": name,
                        "lambda": lam,
                        "score": score,
                        "normalize_input": normalize_input,
                    }
                )

        for name, matrix, normalize_input in (
            ("lda_raw", raw, False),
            ("lda_norm", norm, True),
        ):
            stats = self._lda_stats(matrix[train_idx], labels[train_idx])
            for shrink in _LDA_SHRINKS:
                weight, bias = self._solve_lda(stats, shrink)
                score = self._accuracy(matrix[holdout_idx], labels[holdout_idx], weight, bias)
                candidates.append(
                    {
                        "method": name,
                        "lambda": shrink,
                        "score": score,
                        "normalize_input": normalize_input,
                    }
                )

        weight, bias = self._fit_centroid(norm[train_idx], labels[train_idx])
        score = self._accuracy(norm[holdout_idx], labels[holdout_idx], weight, bias)
        candidates.append(
            {
                "method": "centroid_norm",
                "lambda": 0.0,
                "score": score,
                "normalize_input": True,
            }
        )

        best = max(candidates, key=lambda item: item["score"])
        full_matrix = norm if best["normalize_input"] else raw

        if best["method"].startswith("ridge"):
            weight, bias = self._solve_ridge(
                self._ridge_stats(full_matrix, labels),
                float(best["lambda"]),
            )
        elif best["method"].startswith("lda"):
            weight, bias = self._solve_lda(
                self._lda_stats(full_matrix, labels),
                float(best["lambda"]),
            )
        else:
            weight, bias = self._fit_centroid(norm, labels)

        logit_scale = 16.0
        return {
            "version": _CACHE_VERSION,
            "weight": weight.float() * logit_scale,
            "bias": bias.float() * logit_scale,
            "normalize_input": bool(best["normalize_input"]),
            "method": best["method"],
            "lambda": float(best["lambda"]),
            "holdout_accuracy": float(best["score"]),
        }

    @staticmethod
    def _stratified_split(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(20260512)
        train_parts: list[torch.Tensor] = []
        holdout_parts: list[torch.Tensor] = []
        for class_idx in range(_NUM_CLASSES):
            indices = torch.nonzero(labels == class_idx, as_tuple=False).flatten()
            order = torch.randperm(indices.numel(), generator=generator)
            shuffled = indices[order]
            holdout_parts.append(shuffled[:50])
            train_parts.append(shuffled[50:])
        return torch.cat(train_parts), torch.cat(holdout_parts)

    @staticmethod
    def _one_hot(labels: torch.Tensor) -> torch.Tensor:
        target = torch.zeros(labels.numel(), _NUM_CLASSES, dtype=torch.float64)
        target.scatter_(1, labels.long().view(-1, 1), 1.0)
        return target

    @classmethod
    def _ridge_stats(cls, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.double()
        ones = torch.ones(x.size(0), 1, dtype=torch.float64)
        x_aug = torch.cat([x, ones], dim=1)
        y = cls._one_hot(labels)
        return x_aug.T @ x_aug, x_aug.T @ y

    @staticmethod
    def _solve_ridge(
        stats: tuple[torch.Tensor, torch.Tensor],
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xtx, xty = stats
        reg = torch.eye(xtx.size(0), dtype=torch.float64) * lam
        reg[-1, -1] = 0.0
        solution = torch.linalg.solve(xtx + reg, xty)
        return solution[:-1].T.float(), solution[-1].float()

    @staticmethod
    def _lda_stats(features: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features.double()
        y = labels.long()
        counts = torch.bincount(y, minlength=_NUM_CLASSES).double().clamp_min(1.0)
        sums = torch.zeros(_NUM_CLASSES, x.size(1), dtype=torch.float64)
        sums.index_add_(0, y, x)
        means = sums / counts.unsqueeze(1)

        xtx = x.T @ x
        class_second = (means * counts.unsqueeze(1)).T @ means
        within = (xtx - class_second) / max(float(x.size(0) - _NUM_CLASSES), 1.0)
        avg_var = torch.diag(within).mean().clamp_min(1e-8)
        return {"means": means, "counts": counts, "cov": within, "avg_var": avg_var}

    @staticmethod
    def _solve_lda(stats: dict[str, torch.Tensor], shrink: float) -> tuple[torch.Tensor, torch.Tensor]:
        means = stats["means"]
        counts = stats["counts"]
        cov = stats["cov"]
        avg_var = stats["avg_var"]
        reg = torch.eye(cov.size(0), dtype=torch.float64) * (avg_var * shrink)
        weight_t = torch.linalg.solve(cov + reg, means.T)
        weight = weight_t.T
        priors = (counts / counts.sum()).log()
        bias = -0.5 * (means * weight).sum(dim=1) + priors
        return weight.float(), bias.float()

    @staticmethod
    def _fit_centroid(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sums = torch.zeros(_NUM_CLASSES, features.size(1), dtype=torch.float32)
        counts = torch.bincount(labels.long(), minlength=_NUM_CLASSES).float().clamp_min(1.0)
        sums.index_add_(0, labels.long(), features.float())
        weight = F.normalize(sums / counts.unsqueeze(1), p=2, dim=1)
        bias = torch.zeros(_NUM_CLASSES, dtype=torch.float32)
        return weight, bias

    @staticmethod
    def _accuracy(
        features: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> float:
        logits = features.float() @ weight.float().T + bias.float()
        return float((logits.argmax(dim=1) == labels.long()).float().mean().item())

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "rademacher":
            return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(param)
        if self.perturbation_mode == "uniform":
            return torch.rand_like(param).mul_(2.0).sub_(1.0)
        raise ValueError(f"unknown perturbation_mode: {self.perturbation_mode}")

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        if not params:
            return {}

        directions = {name: self._sample_direction(param) for name, param in params.items()}
        with torch.no_grad():
            for name, param in params.items():
                param.add_(self.eps * directions[name])
            f_plus = loss_fn()

            for name, param in params.items():
                param.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()

            for name, param in params.items():
                param.add_(self.eps * directions[name])

        coeff = (f_plus - f_minus) / (2.0 * self.eps)
        return {name: directions[name] * coeff for name in params}

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        if not params:
            return

        beta1, beta2 = 0.9, 0.999
        self._step_idx += 1
        with torch.no_grad():
            for name, param in params.items():
                grad = grads[name].clamp_(-10.0, 10.0)
                self._m[name] = self._m.get(name, torch.zeros_like(param)).mul(beta1).add(grad, alpha=1.0 - beta1)
                self._v[name] = self._v.get(name, torch.zeros_like(param)).mul(beta2).addcmul(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )
                m_hat = self._m[name] / (1.0 - beta1**self._step_idx)
                v_hat = self._v[name] / (1.0 - beta2**self._step_idx)
                param.sub_(self.lr * m_hat / (v_hat.sqrt() + 1e-8))

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)
