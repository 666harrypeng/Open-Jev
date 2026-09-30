import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)
        self.dropout = nn.Dropout(.2)
        self.model_config = {"fixture": True}

    def forward(self, questions, observations):
        score = self.head(self.dropout(observations["features"])).squeeze(-1)
        return torch.stack([score*0, score], dim=-1)

    def save(self, path):
        path = Path(path); path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "fixture.pt")


class VisualTrainingTests(unittest.TestCase):
    def batches(self, epoch, start):
        # Different epoch order; a cursor is part of the continuation contract.
        data = [(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[0., 1.], [1., 0.]])),
                (torch.tensor([[1., 1.]]), torch.tensor([[0., 1.]]))]
        if epoch % 2:
            data.reverse()
        for features, targets in data[start:]:
            yield {"inputs": {"questions": ["q"]*len(features), "observations": {"features": features}},
                   "targets": targets, "sample_ids": ["fixture"]*len(features)}

    def test_resume_matches_uninterrupted_updates_and_rng(self):
        from jev.visual_training import fit_updates
        cfg = dict(max_steps=4, accumulation=2, lr=.01, head_lr=.01, weight_decay=0.,
                   warmup_steps=0, clip_grad_norm=1., brier_weight=.1, save_every=1, eval_every=0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.manual_seed(7); np.random.seed(7); random.seed(7)
            full = FixtureModel()
            fit_updates(full, self.batches, cfg, root/'full', identity={"data":"fixture"})
            torch.manual_seed(7); np.random.seed(7); random.seed(7)
            paused = FixtureModel()
            first = fit_updates(paused, self.batches, cfg, root/'resume', identity={"data":"fixture"}, stop_after=2)
            resumed = FixtureModel()  # Its construction consumes RNG; restoration must undo that.
            result = fit_updates(resumed, self.batches, cfg, root/'resume', identity={"data":"fixture"}, resume=first['checkpoint'])
            self.assertEqual(result['completed_step'], 4)
            for a, b in zip(full.parameters(), resumed.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(ValueError):
                fit_updates(FixtureModel(), self.batches, {**cfg,"lr":.02}, root/'resume',
                            identity={"data":"fixture"}, resume=result['checkpoint'])

    def test_noul_loss_respects_yes_no_target_order(self):
        from jev.visual_training import noul_loss
        logits = torch.tensor([[0., 3.], [0., -3.]], requires_grad=True)
        targets = torch.tensor([[0., 1.], [1., 0.]])
        good = noul_loss(logits, targets, .1)
        bad = noul_loss(logits, targets.flip(-1), .1)
        self.assertLess(float(good.mean()), float(bad.mean()))
        good.mean().backward()
        self.assertLess(float(logits.grad[0,1]), 0.)
        with self.assertRaises(ValueError):
            noul_loss(logits, torch.ones_like(targets), .1)

    def test_older_checkpoint_is_refused_before_logs_are_modified(self):
        from jev.visual_training import fit_updates
        cfg = dict(max_steps=4, accumulation=1, lr=.01, head_lr=.01, weight_decay=0.,
                   warmup_steps=0, clip_grad_norm=1., brier_weight=.1, save_every=1, eval_every=0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            fit_updates(FixtureModel(),self.batches,cfg,root,identity={"data":"fixture"},stop_after=3)
            before=(root/'training.jsonl').read_bytes()
            with self.assertRaisesRegex(ValueError,"latest"):
                fit_updates(FixtureModel(),self.batches,cfg,root,identity={"data":"fixture"},
                            resume=root/'checkpoints/step-00000001')
            self.assertEqual((root/'training.jsonl').read_bytes(),before)

    def test_repeated_interruptions_before_same_snapshot_remain_recoverable(self):
        from jev.visual_training import fit_updates
        class FailingSave(FixtureModel):
            def save(self,path):
                raise RuntimeError('interrupted snapshot')
        cfg = dict(max_steps=2, accumulation=1, lr=.01, head_lr=.01, weight_decay=0.,
                   warmup_steps=0, clip_grad_norm=1., brier_weight=.1, save_every=1, eval_every=0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=fit_updates(FixtureModel(),self.batches,cfg,root,identity={"data":"fixture"},stop_after=1)
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError,'interrupted snapshot'):
                    fit_updates(FailingSave(),self.batches,cfg,root,identity={"data":"fixture"},resume=first['checkpoint'])
            result=fit_updates(FixtureModel(),self.batches,cfg,root,identity={"data":"fixture"},resume=first['checkpoint'])
            self.assertEqual(result['status'],'completed')


if __name__ == "__main__":
    unittest.main()
