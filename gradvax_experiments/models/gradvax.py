"""GradVax: Norm-Aware Gradient Projection for Tail-Class Preservation in LTSSL."""
import torch
import torch.nn as nn


class GradVax:
    """Classifier-only gradient surgery for LTSSL.

    Modifies only classifier.weight.grad (and classifier.bias.grad if present).
    Backbone gradients follow the standard total-loss backward path unchanged.

    Args:
        classifier: final linear layer (nn.Linear)
        num_classes: total number of classes
        class_partition: dict with keys 'head', 'medium', 'tail' → list of class indices
        beta: EMA momentum for tail anchor (default 0.99)
        tau: domination threshold (default 5.0)
        e_warm: warm-up steps before applying projection/dampening (default 2500)
    """

    def __init__(self, classifier, num_classes, class_partition,
                 beta=0.99, tau=5.0, e_warm=2500):
        self.classifier = classifier
        self.head_cls = set(class_partition['head'])
        self.medium_cls = set(class_partition['medium'])
        self.tail_cls = set(class_partition['tail'])
        self.beta = beta
        self.tau = tau
        self.e_warm = e_warm
        self.step = 0
        self._ema = None  # EMA-smoothed tail anchor (flattened classifier params)
        self._eps = 1e-8

    def _classifier_params(self):
        return list(self.classifier.parameters())

    def _extract_grad(self, loss, retain):
        """Extract classifier gradients for a scalar loss."""
        params = self._classifier_params()
        grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                    allow_unused=True)
        # Flatten to a single vector
        parts = [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
                 for g, p in zip(grads, params)]
        return torch.cat(parts)

    def _write_grad(self, flat_grad):
        """Write a flat gradient vector back to classifier .grad tensors."""
        offset = 0
        for p in self._classifier_params():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat_grad[offset:offset + n].reshape(p.shape))
            offset += n

    def _update_ema(self, g_sup_tail_flat):
        if self._ema is None:
            self._ema = g_sup_tail_flat.detach().clone()
        else:
            self._ema = self.beta * self._ema + (1 - self.beta) * g_sup_tail_flat.detach()

    def _ema_valid(self):
        return self._ema is not None and torch.isfinite(self._ema).all() and self._ema.norm() > self._eps

    def apply(self, logits, labels, pseudo_logits, pseudo_labels,
              labeled_mask, unlabeled_mask, lambda_u=1.0, head_downweight=1.0):
        """Apply GradVax gradient surgery in-place on classifier .grad.

        Must be called AFTER loss_total.backward() has already been called.
        Assumes optimizer.zero_grad() was called before the forward pass.

        Args:
            logits: classifier output for labeled batch [N_l, C]
            labels: ground-truth labels for labeled batch [N_l]
            pseudo_logits: classifier output for unlabeled batch [N_u, C]
            pseudo_labels: pseudo-labels for unlabeled batch [N_u] (hard)
            labeled_mask: bool tensor [N_l] — which labeled samples are in batch
            unlabeled_mask: bool tensor [N_u] — which unlabeled samples have confident pseudo-labels
            lambda_u: unsupervised loss weight (must match trainer)
            head_downweight: head class loss weight (must match trainer)
        """
        self.step += 1
        device = logits.device

        # Identify tail labeled samples and head pseudo-labeled samples
        tail_mask = torch.tensor([l.item() in self.tail_cls for l in labels],
                                 dtype=torch.bool, device=device)
        head_pseudo_mask = unlabeled_mask & torch.tensor(
            [p.item() in self.head_cls for p in pseudo_labels],
            dtype=torch.bool, device=device)

        # --- Update EMA anchor (always, even during warm-up) ---
        n_tail = tail_mask.sum().item()
        g_sup_tail_flat = None
        if n_tail >= 2:
            loss_sup_tail = nn.functional.cross_entropy(
                logits[tail_mask], labels[tail_mask])
            # Scale matches loss_total supervised contribution (lambda_u not applied to sup loss)
            g_sup_tail_flat = self._extract_grad(loss_sup_tail, retain=True)
            self._update_ema(g_sup_tail_flat)

        # --- Skip GradVax if warm-up not done or conditions not met ---
        if self.step <= self.e_warm:
            return
        if not self._ema_valid():
            return
        if not head_pseudo_mask.any():
            return

        # --- Extract head pseudo gradient ---
        # Scale must match the contribution in loss_total: lambda_u * head_downweight * CE
        loss_head_pseudo = (lambda_u * head_downweight) * nn.functional.cross_entropy(
            pseudo_logits[head_pseudo_mask], pseudo_labels[head_pseudo_mask])
        g_head = self._extract_grad(loss_head_pseudo, retain=False)

        g_anchor = self._ema
        anchor_norm = g_anchor.norm() + self._eps

        # Rule 1: Conflict projection
        cos_theta = torch.dot(g_head, g_anchor) / (g_head.norm() * anchor_norm + self._eps)
        g_mod = g_head
        no_conflict = getattr(self, '_no_conflict', False)
        no_domination = getattr(self, '_no_domination', False)
        no_ema = getattr(self, '_no_ema', False)

        # no_ema: use instantaneous g_sup_tail instead of EMA anchor
        if no_ema:
            if g_sup_tail_flat is not None:
                g_anchor = g_sup_tail_flat
                anchor_norm = g_anchor.norm() + self._eps
                cos_theta = torch.dot(g_head, g_anchor) / (g_head.norm() * anchor_norm + self._eps)
            else:
                return  # no valid instantaneous anchor

        if not no_conflict and cos_theta < 0:
            g_mod = g_head - (torch.dot(g_head, g_anchor) / (anchor_norm ** 2)) * g_anchor

        # Rule 2: Domination dampening (applied after projection)
        if not no_domination:
            ratio = g_mod.norm() / anchor_norm
            if ratio > self.tau:
                g_mod = (self.tau / ratio) * g_mod

        # Write modified head gradient back to classifier .grad
        # We need to add the delta: (g_mod - g_head) to existing .grad
        delta = g_mod - g_head
        offset = 0
        for p in self._classifier_params():
            n = p.numel()
            if p.grad is not None:
                p.grad.add_(delta[offset:offset + n].reshape(p.shape))
            offset += n

    def log_stats(self):
        """Return diagnostic stats dict (call before apply to get pre-intervention values)."""
        if not self._ema_valid():
            return {}
        return {'ema_norm': self._ema.norm().item()}
