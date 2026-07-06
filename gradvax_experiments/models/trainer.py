"""FixMatch trainer with optional GradVax."""
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class FixMatchTrainer:
    def __init__(self, model, labeled_loader, unlabeled_loader, test_loader,
                 partition, optimizer, scheduler=None,
                 threshold=0.95, lambda_u=1.0,
                 total_steps=131072, eval_every=500,
                 gradvax=None, oracle_labels=False,
                 head_downweight=1.0, grad_clip=0.0,
                 device='cuda', log_dir='logs', run_id='run'):
        self.model = model
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader
        self.partition = partition
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.total_steps = total_steps
        self.eval_every = eval_every
        self.gradvax = gradvax
        self.oracle_labels = oracle_labels
        self.head_downweight = head_downweight
        self.grad_clip = grad_clip
        self.device = device
        self.log_dir = log_dir
        self.run_id = run_id

        self.head_cls = set(partition['head'])
        self.tail_cls = set(partition['tail'])

        self._diag_buffer = []  # buffered diagnostic stats
        self._diag_flush_every = 50  # flush every N steps
        self._results = []

    def train(self):
        self.model.to(self.device)
        labeled_iter = _inf_loader(self.labeled_loader)
        unlabeled_iter = _inf_loader(self.unlabeled_loader)

        for step in range(1, self.total_steps + 1):
            self.model.train()
            imgs_l, labels = next(labeled_iter)
            imgs_w, imgs_s, true_u = next(unlabeled_iter)

            imgs_l = imgs_l.to(self.device)
            labels = labels.to(self.device)
            imgs_w = imgs_w.to(self.device)
            imgs_s = imgs_s.to(self.device)
            true_u = true_u.to(self.device)

            # Forward
            logits_l = self.model(imgs_l)
            with torch.no_grad():
                probs_w = F.softmax(self.model(imgs_w), dim=-1)
            max_probs, pseudo_labels = probs_w.max(dim=-1)
            conf_mask = max_probs >= self.threshold

            if self.oracle_labels:
                pseudo_labels = true_u
                conf_mask = torch.ones_like(conf_mask)

            logits_s = self.model(imgs_s)

            # Losses
            loss_sup = F.cross_entropy(logits_l, labels)
            if conf_mask.any():
                if self.head_downweight != 1.0:
                    head_mask = torch.tensor(
                        [p.item() in self.head_cls for p in pseudo_labels],
                        dtype=torch.bool, device=self.device) & conf_mask
                    non_head_mask = conf_mask & ~head_mask
                    loss_u = torch.tensor(0.0, device=self.device)
                    if non_head_mask.any():
                        loss_u = loss_u + F.cross_entropy(logits_s[non_head_mask], pseudo_labels[non_head_mask])
                    if head_mask.any():
                        loss_u = loss_u + self.head_downweight * F.cross_entropy(logits_s[head_mask], pseudo_labels[head_mask])
                else:
                    loss_u = F.cross_entropy(logits_s[conf_mask], pseudo_labels[conf_mask])
            else:
                loss_u = torch.tensor(0.0, device=self.device)

            loss_total = loss_sup + self.lambda_u * loss_u

            need_diag = self.gradvax is None and step % 20 == 0
            need_retain = self.gradvax is not None or need_diag
            self.optimizer.zero_grad(set_to_none=True)
            loss_total.backward(retain_graph=need_retain)

            # GradVax
            if self.gradvax is not None:
                self.gradvax.apply(
                    logits=logits_l,
                    labels=labels,
                    pseudo_logits=logits_s,
                    pseudo_labels=pseudo_labels,
                    labeled_mask=torch.ones(len(labels), dtype=torch.bool, device=self.device),
                    unlabeled_mask=conf_mask,
                    lambda_u=self.lambda_u,
                    head_downweight=self.head_downweight,
                )

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Diagnostics (buffered)
            self._collect_diag(step, logits_l, labels, logits_s, pseudo_labels, conf_mask, need_diag)
            if len(self._diag_buffer) >= self._diag_flush_every:
                self._flush_diag()

            # Evaluation
            if step % self.eval_every == 0:
                metrics = self._evaluate(step)
                self._results.append(metrics)
                self._save_results()
                tqdm.write(f"[{self.run_id}] step={step} "
                           f"overall={metrics['overall']:.3f} "
                           f"tail={metrics['tail']:.3f}")

        self._flush_diag()
        return self._results

    @torch.no_grad()
    def _evaluate(self, step):
        self.model.eval()
        num_classes = len(self.partition['head']) + len(self.partition['medium']) + len(self.partition['tail'])
        per_class_correct = torch.zeros(num_classes)
        per_class_total = torch.zeros(num_classes)
        total_correct = 0
        total_seen = 0

        for imgs, labels in self.test_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds = self.model(imgs).argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.numel()
            for c in range(num_classes):
                mask = labels == c
                per_class_correct[c] += (preds[mask] == c).sum().item()
                per_class_total[c] += mask.sum().item()

        per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).numpy()
        head_idx = self.partition['head']
        med_idx = self.partition['medium']
        tail_idx = self.partition['tail']

        return {
            'step': step,
            'overall': total_correct / max(total_seen, 1),
            'head': float(per_class_acc[head_idx].mean()),
            'medium': float(per_class_acc[med_idx].mean()),
            'tail': float(per_class_acc[tail_idx].mean()),
            'balanced': float(per_class_acc.mean()),
        }

    def _collect_diag(self, step, logits_l, labels, logits_s, pseudo_labels, conf_mask, need_diag=False):
        """Collect gradient diagnostic stats (no disk I/O per step)."""
        if self.gradvax is None:
            if not need_diag:
                return
            with torch.no_grad():
                tail_mask = torch.tensor([l.item() in self.tail_cls for l in labels],
                                         dtype=torch.bool, device=self.device)
                head_mask = conf_mask & torch.tensor(
                    [p.item() in self.head_cls for p in pseudo_labels],
                    dtype=torch.bool, device=self.device)
                if not tail_mask.any() or not head_mask.any():
                    return
            # Compute gradients for diagnostics only
            try:
                loss_t = F.cross_entropy(logits_l[tail_mask], labels[tail_mask])
                loss_h = F.cross_entropy(logits_s[head_mask], pseudo_labels[head_mask])
                classifier = self._get_classifier()
                if classifier is None:
                    return
                params = list(classifier.parameters())
                g_t = torch.autograd.grad(loss_t, params, retain_graph=True, allow_unused=True)
                g_h = torch.autograd.grad(loss_h, params, retain_graph=False, allow_unused=True)
                g_t_flat = torch.cat([g.reshape(-1) for g in g_t if g is not None])
                g_h_flat = torch.cat([g.reshape(-1) for g in g_h if g is not None])
                cos = F.cosine_similarity(g_t_flat.unsqueeze(0), g_h_flat.unsqueeze(0)).item()
                ratio = (g_h_flat.norm() / (g_t_flat.norm() + 1e-8)).item()
                self._diag_buffer.append({
                    'step': step,
                    'cos': cos,
                    'norm_ratio': ratio,
                    'conflict': int(cos < 0),
                    'domination': int(ratio > 5.0),
                })
            except Exception:
                pass

    def _get_classifier(self):
        # Try to find the last linear layer
        for m in reversed(list(self.model.modules())):
            if isinstance(m, nn.Linear):
                return m
        return None

    def _flush_diag(self):
        if not self._diag_buffer:
            return
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        path = f"{self.log_dir}/{self.run_id}_diag.jsonl"
        with open(path, 'a') as f:
            for entry in self._diag_buffer:
                f.write(json.dumps(entry) + '\n')
        self._diag_buffer.clear()

    def _save_results(self):
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        path = f"{self.log_dir}/{self.run_id}_results.json"
        with open(path, 'w') as f:
            json.dump(self._results, f, indent=2)


def _inf_loader(loader):
    while True:
        yield from loader
