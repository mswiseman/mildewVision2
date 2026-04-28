import torch


class Recorder:
    """
    Recorder for BCE-style training that supports:
      - Binary: logits (B,1), labels (B,) or (B,1)
      - Multi-head (multi-label): logits (B,H), labels (B,H)

    Tracks:
      - mean loss
      - per-head TP/FP/FN
      - per-head precision/recall/F1
      - macro F1 across heads

    Notes:
      - Uses sigmoid(logits) with per-head thresholds
      - thresholds can be float (applied to all heads) or a list/tuple/tensor of length H
    """

    def __init__(self, name, thresholds=0.5):
        self._name = name
        self.thresholds = thresholds
        self.reset()

    @property
    def name(self):
        return self._name

    def reset(self):
        self.loss_sum = 0.0
        self.total_counts = 0
        self.tp = None
        self.fp = None
        self.fn = None

    @property
    def loss(self):
        return self.loss_sum / max(self.total_counts, 1)

    def _as_2d_float_labels(self, labels, device):
        """
        Ensure labels is float32 tensor of shape (B,H).
        Accepts labels as:
          - (B,) int/float
          - (B,1)
          - (B,H)
        """
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)

        labels = labels.to(device=device)

        # If scalar labels (B,), make (B,1)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        # Convert to float targets in {0,1}
        if labels.dtype not in (torch.float16, torch.float32, torch.float64):
            labels = labels.float()
        else:
            labels = labels.to(dtype=torch.float32)

        return labels

    def _as_2d_logits(self, logits):
        """
        Ensure logits is tensor of shape (B,H).
        Accepts logits as:
          - (B,)  -> (B,1)
          - (B,1)
          - (B,H)
        """
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return logits

    def _get_thresholds_tensor(self, num_heads, device):
        """
        Returns thresholds tensor of shape (H,) on device.
        """
        if isinstance(self.thresholds, (float, int)):
            return torch.full((num_heads,), float(self.thresholds), device=device)
        if isinstance(self.thresholds, (list, tuple)):
            if len(self.thresholds) != num_heads:
                raise ValueError(f"thresholds length ({len(self.thresholds)}) != num_heads ({num_heads})")
            return torch.tensor(self.thresholds, dtype=torch.float32, device=device)
        if torch.is_tensor(self.thresholds):
            thr = self.thresholds.to(device=device, dtype=torch.float32).view(-1)
            if thr.numel() != num_heads:
                raise ValueError(f"thresholds tensor size ({thr.numel()}) != num_heads ({num_heads})")
            return thr
        raise TypeError("thresholds must be float, list/tuple, or torch tensor")

    def _ensure_counters(self, num_heads):
        """
        Initialize/resize TP/FP/FN counters to match num_heads.
        """
        if self.tp is None or self.tp.numel() != num_heads:
            self.tp = torch.zeros(num_heads, dtype=torch.long)
            self.fp = torch.zeros(num_heads, dtype=torch.long)
            self.fn = torch.zeros(num_heads, dtype=torch.long)

    def update(self, logits, labels, loss, batch_size=None):
        """
        logits: (B,H) or (B,) or (B,1)
        labels: (B,H) or (B,) or (B,1)
        loss: scalar (mean loss for batch)
        """
        logits = self._as_2d_logits(logits)

        if batch_size is None:
            batch_size = logits.size(0)

        # accumulate loss as total (sum over samples)
        self.loss_sum += float(loss) * batch_size
        self.total_counts += batch_size

        with torch.no_grad():
            labels = self._as_2d_float_labels(labels, device=logits.device)

            # labels/logits must match in head dimension
            if labels.size(1) != logits.size(1):
                raise ValueError(
                    f"Labels head dim {labels.size(1)} != logits head dim {logits.size(1)}. "
                    f"labels shape={tuple(labels.shape)} logits shape={tuple(logits.shape)}"
                )

            num_heads = logits.size(1)
            self._ensure_counters(num_heads)

            thr = self._get_thresholds_tensor(num_heads, logits.device)  # (H,)
            probs = torch.sigmoid(logits)  # (B,H)

            preds = probs > thr  # (B,H) broadcast
            y = labels > 0.5  # (B,H)

            tp = (preds & y).sum(dim=0).cpu()
            fp = (preds & (~y)).sum(dim=0).cpu()
            fn = ((~preds) & y).sum(dim=0).cpu()

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def _prf(self):
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
        return precision, recall, f1

    @property
    def precision(self):
        return self._prf()[0]

    @property
    def recall(self):
        return self._prf()[1]

    @property
    def f1(self):
        return self._prf()[2]

    @property
    def f1_macro(self):
        return float(self.f1.mean().item())

    def summary(self, head_names=None):
        """
        head_names: optional list like ["infected"] or ["infected","sporulating"]
        """
        if self.tp is None:
            return {"loss": self.loss, "f1_macro": 0.0}

        p, r, f1 = self._prf()
        H = len(p)
        if head_names is None:
            head_names = [f"head{i}" for i in range(H)]
        elif len(head_names) != H:
            raise ValueError(f"head_names length ({len(head_names)}) != num_heads ({H})")

        out = {"loss": self.loss, "f1_macro": float(f1.mean().item())}
        for i, name in enumerate(head_names):
            out[f"{name}_precision"] = float(p[i].item())
            out[f"{name}_recall"] = float(r[i].item())
            out[f"{name}_f1"] = float(f1[i].item())
        return out
