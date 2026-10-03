import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from torch import nn


class FixtureProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][-1]["text"]

    def __call__(self, text, images, **kwargs):
        self.images = images
        values = [[sum(image.getpixel((0, 0))) / 765 for image in images[2*i:2*i+2]]
                  for i in range(len(text))]
        return {"input_ids": torch.ones(len(text), 3, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0]] * len(text)),
                "pixel_values": torch.tensor(values)}


class FixtureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)
        self.language_model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.visual.weight.copy_(torch.eye(2))
            self.language_model.weight.copy_(torch.eye(2))
        self.requires_grad_(False)

    def forward(self, pixel_values, **kwargs):
        value = self.language_model(self.visual(pixel_values))
        hidden = torch.stack([value * 0, value, value + 1000], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


class VisualModelTests(unittest.TestCase):
    def model(self):
        from jev.visual_model import VisualDecisionModel
        head = nn.Linear(2, 1)
        with torch.no_grad():
            head.weight.fill_(1); head.bias.zero_()
        return VisualDecisionModel(FixtureBackbone(), FixtureProcessor(), head,
                                   {"max_length": 64, "camera_names": ["overview", "wrist"]})

    def test_both_images_reach_backbone_and_padding_is_not_pooled(self):
        model = self.model()
        black = torch.zeros(1, 1, 3, 8, 8, dtype=torch.uint8)
        white = torch.full_like(black, 255)
        baseline = model(["Is it open?"], {"overview": black, "wrist": black})
        changed = model(["Is it open?"], {"overview": black, "wrist": white})
        self.assertEqual(baseline.tolist(), [[0., 0.]])
        self.assertEqual(changed.tolist(), [[0., 1.]])
        self.assertEqual(model.processor.images[0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(model.processor.images[1].getpixel((0, 0)), (255, 255, 255))
        changed.sum().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_prepared_inputs_match_raw_images_and_gradients(self):
        from jev.visual_model import encode_visual_inputs
        model=self.model()
        image=torch.full((1,1,3,8,8),100,dtype=torch.uint8)
        images={"overview":image,"wrist":image+20}
        encoded=encode_visual_inputs(model.processor,["Is it tilted?"],images,model.camera_names,64)
        direct=model(["Is it tilted?"],images)
        prepared=model(encoded=encoded)
        torch.testing.assert_close(direct,prepared)
        prepared.sum().backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):model(["q"],images,encoded=encoded)

    def test_history_and_wrong_camera_sets_are_rejected(self):
        model = self.model()
        image = torch.zeros(1, 1, 3, 8, 8, dtype=torch.uint8)
        for images in ({"overview": image}, {"overview": image.repeat(1, 2, 1, 1, 1), "wrist": image}):
            with self.assertRaises(ValueError):
                model(["Is it open?"], images)

    def test_pooling_handles_left_and_right_padding(self):
        from jev.visual_model import last_valid_hidden
        hidden = torch.arange(12).reshape(2, 3, 2).float()
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        self.assertEqual(last_valid_hidden(hidden, mask).tolist(), [[2., 3.], [10., 11.]])
        with self.assertRaises(ValueError):
            last_valid_hidden(hidden, torch.zeros_like(mask))

    def test_text_loader_rejects_visual_checkpoint_before_loading_base_weights(self):
        from jev.model import DecisionModel
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,"model.json").write_text(json.dumps({"method":"native_qwen_visual_noul"}))
            with self.assertRaisesRegex(ValueError,"VisualDecisionModel"):
                DecisionModel.load(temp,device="cpu")


if __name__ == "__main__":
    unittest.main()
