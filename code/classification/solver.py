### Solver (train and test) — MULTI-LABEL / DUAL-HEAD VERSION
#
# Assumes:
#   - Dataset returns labels as float targets in {0,1} with shape (B, 2)
#       [infected, sporulating]
#   - Model outdim == 2 and returns logits shape (B, 2)
#   - Recorder is the BCE/multi-head compatible Recorder (no .correct/.total; uses .summary())
#
# Notes:
#   - This file removes multiclass confusion-matrix evaluation.
#   - “val_accuracy” is replaced with macro-F1 (percent) as a placeholder so your existing
#     checkpoint saving / printing stays coherent.
#   - Weighted loss: the old class-weight logic is incompatible with dual-head labels.
#     Use pos_weight (optional) instead (example included below).

import os
import time
import copy
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from recorder import Recorder
from utils import makeSubdir, logInfoWithDot, timeSince, init_model, init_optimizer

from termcolor import colored
import optuna

if __name__ == "__main__":
    from utils import init_model, load_model, parse_model, plot_confusion_matrix


class HierarchicalBCEWithLogits(nn.Module):
    def __init__(self, pos_weight_h1=None, pos_weight_h2=None, eps=1e-6):
        super().__init__()
        self.register_buffer("pos_weight_h1", pos_weight_h1 if pos_weight_h1 is not None else torch.tensor(1.0))
        self.register_buffer("pos_weight_h2", pos_weight_h2 if pos_weight_h2 is not None else torch.tensor(1.0))
        self.eps = eps

    def forward(self, logits, labels):
        """
        logits: (B,2)
        labels: (B,2) -> [infected, sporulating]
        """

        infected = labels[:, 0]
        spor = labels[:, 1]

        # ---- Head 1 loss (all samples) ----
        l1 = F.binary_cross_entropy_with_logits(
            logits[:, 0], infected,
            pos_weight=self.pos_weight_h1.to(logits.device, logits.dtype),
            reduction="mean"
        )

        # ---- Head 2 masked loss ----
        per_l2 = F.binary_cross_entropy_with_logits(
            logits[:, 1], spor,
            pos_weight=self.pos_weight_h2.to(logits.device, logits.dtype),
            reduction="none"
        )

        l2 = (per_l2 * infected).sum() / (infected.sum() + self.eps)

        return l1 + l2


def _compute_pos_weight_from_raw_labels(raw_labels_012: np.ndarray):
    """
    raw labels: 0=clear, 1=hyphae, 2=sporulating

    Head 0 (infected vs clear):
      pos = {1,2}, neg = {0}

    Head 1 (sporulating vs hyphae) **infected-only**:
      pos = {2}, neg = {1}   (ignore 0 entirely)
    """
    y = np.asarray(raw_labels_012).reshape(-1).astype(np.int64)

    # head 0
    infected = (y == 1) | (y == 2)
    pos0 = int(infected.sum())
    neg0 = int((~infected).sum())
    pw0 = float(neg0 / max(pos0, 1))

    # head 1 (infected-only)
    pos1 = int((y == 2).sum())  # sporulating
    neg1 = int((y == 1).sum())  # hyphae
    pw1 = float(neg1 / max(pos1, 1))

    return np.array([pw0, pw1], dtype=np.float32)


class Solver():
    def __init__(self, model, dataloader, optimizer, scheduler, logger, writer, trial=None):
        """
        Args:
            model:         Model params
            dataloader:    Dataloader params
            optimizer:     Optimizer params
            scheduler:     Scheduler params
            logger:        Logger params
            writer:        Tensorboard writer params
        """
        self.model = init_model(model)
        self.pretrained = model['pretrained']
        self.loading_epoch = model['loading_epoch']
        self.total_epochs = model['total_epochs']
        self.model_path = model['model_path']
        self.model_filename = model['model_filename']
        self.outdim = model['outdim']  # MUST be 2 for dual-head
        self.save = model['save']
        self.patience = model['patience']
        self.model_name = model['model_type']
        self.model_fullpath = str(self.model_path / self.model_filename)
        self.max_grad_norm = float(optimizer.get('max_grad_norm', 0.0) or 0.0)  # 0.0 means disabled
        self.trial = trial
        self.is_trial_run = trial is not None

        self.init_random_seed = model['manual_seed']

        # Best model
        self.is_best = False
        self.best_model = None
        self.best_acc = 0
        self.best_optim = None
        self.best_metrics = None
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.best_model_filepath = str(self.model_path / 'best_model_checkpoint.pth.tar')

        # Logger
        self.logger = logger

        # Loss tracking / early stop
        self.best_loss = float('inf')
        self.patience_counter = 0

        self.trainloader = dataloader['train']
        self.validloader = dataloader['valid']

        # Writer: Default path is runs/CURRENT_DATETIME_HOSTNAME
        writer_fullname = writer['writer_path'] / writer['writer_filename']
        self.writer = SummaryWriter(log_dir=str(writer_fullname))

        # Device
        if model.get('gpu', False) and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            try:
                self.scaler = torch.amp.GradScaler('cuda')  # PyTorch ≥ 2.3
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler()  # older API
            logInfoWithDot(self.logger, "USING GPU")

        elif model.get('mps', False) and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.model = self.model.to(self.device)
            self.scaler = None
            logInfoWithDot(self.logger, "USING MPS")

        else:
            self.device = torch.device('cpu')
            self.scaler = None
            logInfoWithDot(self.logger, "USING CPU")

        # Resume model
        checkpoint = None
        if model['resume']:
            load_model_path = self.model_fullpath.format(self.model_name, self.loading_epoch)
            checkpoint = torch.load(load_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logInfoWithDot(self.logger, "LOADING MODEL FINISHED")

        # --- Loss: BCEWithLogitsLoss for dual-head ---
        # Old weighted_loss (class weights) is NOT compatible with dual-head.
        # If you want weighting, use pos_weight per head (optional).
        pos_weight = None
        if optimizer.get('weighted_loss', False) and hasattr(self.trainloader.dataset, 'labels'):
            # NOTE: dataset.labels are raw 0/1/2 ints loaded from HDF5 in your dataset class
            raw = np.asarray(self.trainloader.dataset.labels).reshape(-1).astype(np.int64)
            pw = _compute_pos_weight_from_raw_labels(raw)  # shape (2,)
            pos_weight = torch.tensor(pw, device=self.device, dtype=torch.float32)
            logInfoWithDot(self.logger, f"Using pos_weight={pw.tolist()} for BCE heads")

        if pos_weight is None:
            self.criterion = HierarchicalBCEWithLogits()
        else:
            self.criterion = HierarchicalBCEWithLogits(
                pos_weight_h1=pos_weight[0],
                pos_weight_h2=pos_weight[1],
            )

        # Optimizer
        self.optimizer = init_optimizer(optimizer, self.model)

        # Scheduler
        self.scheduler = None
        if scheduler['use']:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler['milestones'],
                gamma=scheduler['gamma']
            )

        # Resume optimizer/scheduler/scaler state (if checkpoint exists)
        if optimizer.get('resume', False) and checkpoint is not None:
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logInfoWithDot(self.logger, "LOADING OPTIMIZER FINISHED")

            if self.scheduler and ('scheduler_state_dict' in checkpoint) and (
                    checkpoint['scheduler_state_dict'] is not None):
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logInfoWithDot(self.logger, "LOADING SCHEDULER FINISHED")
                except Exception as e:
                    self.logger.warning(f"Could not load scheduler state: {e}")

            if self.scaler and ('scaler_state_dict' in checkpoint) and (checkpoint['scaler_state_dict'] is not None):
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logInfoWithDot(self.logger, "LOADING AMP SCALER FINISHED")
                except Exception as e:
                    self.logger.warning(f"Could not load AMP scaler state: {e}")

        # --- Evaluation / Recorder (BCE multi-head) ---
        # Tune thresholds later. These starting values often help when sporulation vs infected are confused.
        self.head_names = ["infected", "sporulating"]
        self.thresholds = optimizer.get("thresholds", [0.5, 0.7])  # allow config override
        self.train_recorder = Recorder('train', thresholds=self.thresholds)
        self.test_recorder = Recorder('val', thresholds=self.thresholds)

        # Timer
        self.start_time = time.time()

    def train_one_epoch(self, ep):
        raise NotImplementedError

    def test_one_epoch(self, ep):
        raise NotImplementedError

    def forward(self):
        start_epoch = self.loading_epoch + 1
        end_epoch = self.total_epochs + 1

        for ep in range(start_epoch, end_epoch):
            np.random.seed(self.init_random_seed + ep)
            torch.manual_seed(self.init_random_seed + ep)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.init_random_seed + ep)

            self.train_one_epoch(ep)

            # Returns: (val_loss, val_accuracy_placeholder, metrics dict with f1/recall/precision in [0,1])
            val_loss, val_accuracy, val_metrics = self.test_one_epoch(ep)

            # Convert to percentage once
            val_f1 = float(100.0 * val_metrics["f1"])
            val_recall = float(100.0 * val_metrics["recall"])
            val_precision = float(100.0 * val_metrics["precision"])

            # Step LR
            if self.scheduler:
                self.scheduler.step()
                cur_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LR', cur_lr, ep)

            # Optuna: report/prune on F1
            if self.trial is not None:
                self.trial.report(val_f1, step=ep)
                if self.trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {ep} (val_f1={val_f1:.3f})")
                    raise optuna.TrialPruned()

            # Save "last" (non-trial runs only)
            if self.save and not self.is_trial_run:
                makeSubdir(self.model_path)
                last_path = str(self.model_path / 'last_model_checkpoint.pth.tar')
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler else None),
                    'scaler_state_dict': (self.scaler.state_dict() if self.scaler else None),

                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_accuracy),  # placeholder (macro-F1%)
                    'val_f1': val_f1,  # %
                    'val_recall': val_recall,  # %
                    'val_precision': val_precision,  # %
                }, last_path)
                logInfoWithDot(self.logger, f"SAVED LAST to {last_path}")

            # Check improvement + save "best" (use F1)
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_loss = float(val_loss)
                self.best_acc = float(val_accuracy)
                self.best_epoch = ep
                self.patience_counter = 0

                self.best_metrics = {
                    "epoch": ep,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_accuracy),
                    "val_f1": float(val_f1),
                    "val_precision": float(val_precision),
                    "val_recall": float(val_recall),
                }

                if self.save and not self.is_trial_run:
                    makeSubdir(self.model_path)
                    best_path = self.model_fullpath.format(self.model_name, ep)
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler else None),
                        'scaler_state_dict': (self.scaler.state_dict() if self.scaler else None),

                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_accuracy),
                        'val_f1': val_f1,
                        'val_recall': val_recall,
                        'val_precision': val_precision,
                    }, best_path)
                    logInfoWithDot(self.logger, f"SAVED BEST to {best_path}")
            else:
                self.patience_counter += 1

            # Early stop
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping at epoch {ep}. "
                    f"Best F1={self.best_f1:.3f}%, "
                    f"best loss={self.best_loss:.6f}, best placeholder-acc={self.best_acc:.3f}% "
                    f"(best epoch={self.best_epoch})"
                )
                break

        if self.best_metrics is not None:
            self.logger.info("============================================")
            self.logger.info(
                f"✅ BEST MODEL @ epoch {self.best_metrics['epoch']} | "
                f"Val F1: {self.best_metrics['val_f1']:.3f}% | "
                f"Val Precision: {self.best_metrics['val_precision']:.3f}% | "
                f"Val Recall: {self.best_metrics['val_recall']:.3f}% | "
                f"Val Loss: {self.best_metrics['val_loss']:.6f}"
            )
            self.logger.info("============================================")


class HyphalSolver(Solver):
    def train_one_epoch(self, ep, log_interval=50):
        self.model.train()
        self.train_recorder.reset()
        lr = self.optimizer.param_groups[0]['lr']

        for i, (images, labels) in enumerate(self.trainloader, 0):
            images = images.to(self.device, dtype=torch.float, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.float32, non_blocking=True)  # (B,2)

            amp_enabled = self.device.type in ('cuda', 'mps') and self.scaler is not None
            with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                if (not self.pretrained) and self.model_name == 'GoogleNet':
                    preds, aux2, aux1 = self.model(images)
                    loss1 = self.criterion(preds, labels)
                    loss2 = self.criterion(aux1, labels)
                    loss3 = self.criterion(aux2, labels)
                    loss = loss1 + 0.3 * (loss2 + loss3)
                elif self.model_name == 'Inception3':
                    preds, aux = self.model(images)
                    loss1 = self.criterion(preds, labels)
                    loss2 = self.criterion(aux, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    preds = self.model(images)  # (B,2)
                    loss = self.criterion(preds, labels)

            # Recorder expects logits (B,H) and labels (B,H) for BCE
            logits_for_metrics = preds.detach().clone()
            clear_mask = labels[:, 0] < 0.5
            logits_for_metrics[clear_mask, 1] = -20.0
            self.train_recorder.update(logits_for_metrics, labels, loss.item(), batch_size=images.size(0))

            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == 'cuda' and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if i % log_interval == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLearning Rate: {}\tLoss: {:.6f}\tTime Usage:{:.8}'
                    .format(ep, i * len(images), len(self.trainloader.dataset),
                            100. * i / len(self.trainloader), lr, loss.item(),
                            timeSince(self.start_time))
                )

            if i == len(self.trainloader) - 1:
                summ = self.train_recorder.summary(self.head_names)
                self.logger.info(f"Loss on {self.train_recorder.total_counts} train images: {summ['loss']:.6f}")
                self.logger.info(
                    f"Train F1(macro): {100.0 * summ['f1_macro']:.3f}% | "
                    f"Infected F1: {100.0 * summ['infected_f1']:.3f}% "
                    f"(P {100.0 * summ['infected_precision']:.2f}%, R {100.0 * summ['infected_recall']:.2f}%) | "
                    f"Spor F1: {100.0 * summ['sporulating_f1']:.3f}% "
                    f"(P {100.0 * summ['sporulating_precision']:.2f}%, R {100.0 * summ['sporulating_recall']:.2f}%)"
                )

                self.writer.add_scalar('Loss/train', summ['loss'], ep)
                self.writer.add_scalar('F1_macro/train', 100.0 * summ['f1_macro'], ep)
                self.writer.add_scalar('F1_infected/train', 100.0 * summ['infected_f1'], ep)
                self.writer.add_scalar('F1_sporulating/train', 100.0 * summ['sporulating_f1'], ep)

    def test_one_epoch(self, ep):
        self.model.eval()
        self.test_recorder.reset()

        total_loss, total = 0.0, 0
        log_interval = 50  # or whatever you like for val

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.validloader):
                images = images.to(self.device, dtype=torch.float, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.float32, non_blocking=True)  # (B,2)

                amp_enabled = self.device.type in ('cuda', 'mps')
                with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                    logits = self.model(images)  # (B,2)
                    loss = self.criterion(logits, labels)

                # Sanity check (log occasionally)
                if i % log_interval == 0:
                    infected_count = (labels[:, 0] > 0.5).sum().item()
                    spor_count = (labels[:, 1] > 0.5).sum().item()
                    self.logger.info(
                        f"[Val ep {ep} batch {i}] infected: {infected_count}/{labels.size(0)} | spor: {spor_count}/{labels.size(0)}"
                    )

                bs = images.size(0)
                total_loss += loss.item() * bs
                total += bs

                # ---- metrics: ignore clears for head 2 ----
                logits_for_metrics = logits.detach().clone()
                clear_mask = labels[:, 0] < 0.5
                logits_for_metrics[clear_mask, 1] = -20.0

                self.test_recorder.update(logits_for_metrics, labels, loss.item(), batch_size=bs)

                #self.test_recorder.update(logits, labels, loss.item(), batch_size=bs)

        avg_loss = total_loss / max(total, 1)
        summ = self.test_recorder.summary(self.head_names)

        # Define macro metrics across heads (in [0,1]) for forward()
        macro_precision = float(self.test_recorder.precision.mean().item())
        macro_recall = float(self.test_recorder.recall.mean().item())
        macro_f1 = float(self.test_recorder.f1.mean().item())

        metrics = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}

        # TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, ep)
        self.writer.add_scalar('F1_macro/val', 100.0 * summ['f1_macro'], ep)
        self.writer.add_scalar('F1_infected/val', 100.0 * summ['infected_f1'], ep)
        self.writer.add_scalar('F1_sporulating/val', 100.0 * summ['sporulating_f1'], ep)
        self.writer.add_scalar('Precision_macro/val', 100.0 * macro_precision, ep)
        self.writer.add_scalar('Recall_macro/val', 100.0 * macro_recall, ep)

        # Logger
        self.logger.info(
            f"Val loss: {avg_loss:.6f} | "
            f"Val F1(macro): {100.0 * summ['f1_macro']:.3f}% | "
            f"Val Recall(macro): {100.0 * macro_recall:.3f}% | "
            f"Val Precision(macro): {100.0 * macro_precision:.3f}% | "
            f"Infected F1: {100.0 * summ['infected_f1']:.3f}% | "
            f"Spor F1: {100.0 * summ['sporulating_f1']:.3f}%"
        )

        # Placeholder "accuracy" (percent) so forward()/checkpoint code stays intact.
        # Using macro-F1% is usually a sensible stand-in.
        val_accuracy_placeholder = 100.0 * summ['f1_macro']

        return avg_loss, val_accuracy_placeholder, metrics


def main():
    pass
