"""Native Qwen vision-language backbone with a shared non-generative Noul head."""
import json
import math
from pathlib import Path

import torch
from torch import nn


def last_valid_hidden(hidden, attention_mask):
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden states and input attention mask do not align")
    positions = torch.arange(attention_mask.shape[1], device=hidden.device).expand(attention_mask.shape[0], -1)
    last = positions.masked_fill(~attention_mask.to(hidden.device).bool(), -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("A decision input has no valid tokens")
    return hidden[torch.arange(len(last), device=hidden.device), last]


class VisualDecisionModel(nn.Module):
    """One current RGB image per named camera, natural-language question, two logits."""
    def __init__(self, backbone, processor, head, model_config):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.head = head
        self.model_config = dict(model_config)
        self.camera_names = tuple(model_config.get("camera_names", ["overview", "wrist"]))
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))
        self.last_input_tokens = 0
        self.last_sequence_length = 0

    @classmethod
    def from_pretrained(cls, model_id, revision, *, device="cuda:0", dtype="bfloat16",
                        lora_rank=8, max_length=2048, min_pixels=65536, max_pixels=65536,
                        gradient_checkpointing=True):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not Path(model_id).exists() and (not isinstance(revision, str) or len(revision) != 40
                                           or any(c not in "0123456789abcdef" for c in revision)):
            raise ValueError("Pin the remote base model to a full commit revision")
        if dtype not in ("float32", "float16", "bfloat16") or lora_rank < 0:
            raise ValueError("Invalid dtype or LoRA rank")
        if max_length <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Invalid token or image-pixel budget")
        processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                                  min_pixels=min_pixels, max_pixels=max_pixels)
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        full = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype),
            device_map={"": device}, attn_implementation="sdpa")
        backbone = full.model
        if not hasattr(backbone, "visual") or not hasattr(backbone, "language_model"):
            raise ValueError("Expected the native Qwen multimodal model wrapper")
        yes = processor.tokenizer.encode("Yes", add_special_tokens=False)
        no = processor.tokenizer.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("Yes/No head initialization requires single-token answers")
        output_weights = full.get_output_embeddings().weight
        initial = (output_weights[yes[0]] - output_weights[no[0]]).detach().float().clone()
        head = nn.Linear(initial.numel(), 1, bias=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            head.weight.copy_(initial.unsqueeze(0)); head.bias.zero_()
        del initial, output_weights, full
        backbone.requires_grad_(False)
        if lora_rank:
            from peft import LoraConfig, get_peft_model
            available = {name.rsplit(".", 1)[-1] for name, _ in backbone.language_model.named_modules()}
            targets = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj")
                       if name in available]
            if not targets:
                raise ValueError("No supported language LoRA target modules")
            backbone.language_model = get_peft_model(backbone.language_model, LoraConfig(
                r=lora_rank, lora_alpha=2*lora_rank, target_modules=targets, lora_dropout=0., bias="none"))
        config = {"model_id": str(model_id), "revision": revision, "dtype": dtype, "lora_rank": lora_rank,
                  "max_length": max_length, "min_pixels": min_pixels, "max_pixels": max_pixels,
                  "gradient_checkpointing": gradient_checkpointing, "camera_names": ["overview", "wrist"],
                  "history_frames": 1, "method": "native_qwen_visual_noul", "format_version": 1}
        model = cls(backbone, processor, head, config)
        model._gradient_checkpointing(gradient_checkpointing)
        return model

    def _gradient_checkpointing(self, enabled):
        language = self.backbone.language_model
        if enabled and self.model_config.get("lora_rank"):
            language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            language.enable_input_require_grads()

    def train(self, mode=True):
        super().train(mode)
        # The frozen visual encoder must remain deterministic during language/head training.
        self.backbone.visual.eval()
        return self

    def prepare_inputs(self, questions, observations):
        from PIL import Image
        if not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
            raise ValueError("Provide nonempty natural-language questions")
        if set(observations) != set(self.camera_names):
            raise ValueError("The model requires exactly its configured camera views")
        for values in observations.values():
            if values.ndim != 5 or values.shape[:3] != (len(questions), 1, 3) or values.dtype != torch.uint8:
                raise ValueError("Expected B,1,3,H,W uint8 current-camera images, without history")
        images, prompts = [], []
        for index, question in enumerate(questions):
            content = []
            for camera in self.camera_names:
                image = Image.fromarray(observations[camera][index, 0].detach().cpu().permute(1, 2, 0).numpy())
                images.append(image)
                content.extend([{"type": "text", "text": f"Current {camera} camera:"},
                                {"type": "image", "image": image}])
            content.append({"type": "text", "text": question + "\nJudge the current images. Is the answer yes or no?"})
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
        encoded = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        if "pixel_values" not in encoded:
            raise ValueError("Processor failed to produce actual visual inputs")
        lengths = encoded["attention_mask"].sum(dim=-1)
        if lengths.max().item() > self.model_config["max_length"]:
            raise ValueError("Multimodal input exceeds max_length; images/text are not silently truncated")
        self.last_input_tokens = int(lengths.sum())
        self.last_sequence_length = int(encoded["attention_mask"].shape[1])
        device = self.head.weight.device
        return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in encoded.items()}

    def forward(self, questions, observations):
        encoded = self.prepare_inputs(questions, observations)
        output = self.backbone(**encoded, use_cache=False, return_dict=True)
        hidden = last_valid_hidden(output.last_hidden_state, encoded["attention_mask"])
        score = self.head(hidden.float()).squeeze(-1)
        return torch.stack([torch.zeros_like(score), score], dim=-1)

    @torch.inference_mode()
    def probabilities(self, questions, observations):
        self.eval()
        return (self(questions, observations) / self.temperature.to(self.head.weight.device)).softmax(dim=-1)

    def set_temperature(self, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.temperature.fill_(value)

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        if self.model_config["lora_rank"]:
            self.backbone.language_model.save_pretrained(output / "adapter")
        self.processor.save_pretrained(output / "processor")
        torch.save(self.head.state_dict(), output / "head.pt")
        (output / "model.json").write_text(json.dumps({**self.model_config, "temperature": float(self.temperature)}, indent=2)+"\n")

    @classmethod
    def load(cls, output, *, device="cuda:0", trainable=False):
        from transformers import AutoProcessor
        output = Path(output)
        config = json.loads((output / "model.json").read_text())
        if config.get("method") != "native_qwen_visual_noul" or config.get("history_frames") != 1:
            raise ValueError("This is not a current-dual-camera visual checkpoint")
        model = cls.from_pretrained(
            config["model_id"], config["revision"], device=device, dtype=config["dtype"], lora_rank=0,
            max_length=config["max_length"], min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
            gradient_checkpointing=False)
        model.processor = AutoProcessor.from_pretrained(output / "processor")
        model.processor.tokenizer.padding_side = "right"
        if config["lora_rank"]:
            from peft import PeftModel
            model.backbone.language_model = PeftModel.from_pretrained(
                model.backbone.language_model, output / "adapter", is_trainable=trainable)
        model.head.load_state_dict(torch.load(output / "head.pt", map_location=device, weights_only=True))
        model.model_config = config
        model.set_temperature(config.get("temperature", 1.0))
        if trainable:
            model._gradient_checkpointing(config["gradient_checkpointing"])
            return model.train()
        return model.eval()
