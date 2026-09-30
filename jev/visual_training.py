"""Single-device visual Noul optimization with restartable step-boundary snapshots."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def noul_loss(logits, targets, brier_weight=.1):
    if logits.ndim != 2 or logits.shape[1] != 2 or targets.shape != logits.shape:
        raise ValueError("Expected B,2 Noul logits and targets")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all() or (targets < 0).any() or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0])):
        raise ValueError("Invalid logits or target distributions")
    if not math.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("Invalid Brier weight")
    return -(targets * logits.log_softmax(-1)).sum(-1) + brier_weight * ((logits.softmax(-1)-targets)**2).sum(-1)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""): h.update(block)
    return h.hexdigest()


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(model, optimizer, output, state, identity):
    parent = output / "checkpoints"; parent.mkdir(exist_ok=True)
    destination = parent / f"step-{state['completed_step']:08d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=parent))
    try:
        model.save(temporary / "model")
        payload = {"identity": identity, "trainer": state, "optimizer": optimizer.state_dict(),
                   "parameters": {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                   "rng": _rng()}
        torch.save(payload, temporary / "training_state.pt")
        manifest = {str(p.relative_to(temporary)): _hash(p) for p in temporary.rglob("*") if p.is_file()}
        (temporary / "checkpoint.json").write_text(json.dumps({"identity":identity,"trainer":state,"files_sha256":manifest},indent=2)+"\n")
        os.replace(temporary,destination)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    return str(destination.resolve())


def _load_checkpoint(model, optimizer, path, identity):
    path = Path(path)
    meta = json.loads((path / "checkpoint.json").read_text())
    if meta["identity"] != identity:
        raise ValueError("Resume model/data/source/training identity differs")
    for name,digest in meta["files_sha256"].items():
        file = (path/name).resolve()
        if not file.is_relative_to(path.resolve()) or _hash(file) != digest:
            raise ValueError("Training checkpoint checksum mismatch")
    # These snapshots are produced locally by this trainer; optimizer/RNG state is not an inference upload format.
    payload = torch.load(path / "training_state.pt",map_location="cpu",weights_only=False)
    if payload["identity"] != identity or payload["trainer"] != meta["trainer"]:
        raise ValueError("Checkpoint state and metadata disagree")
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(payload["parameters"]):
        raise ValueError("Resume trainable parameter set differs")
    with torch.no_grad():
        for name,p in trainable.items(): p.copy_(payload["parameters"][name].to(p.device,p.dtype))
    optimizer.load_state_dict(payload["optimizer"])
    return payload["trainer"], payload["rng"]


def _rate(step, warmup, maximum):
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup - 1) / max(1, maximum - warmup)
    return .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))


def fit_updates(model, loader_factory, config, output, *, identity, validation_fn=None, resume=None, stop_after=None):
    """loader_factory(epoch,start_batch) must reconstruct deterministic, cursor-aware order."""
    for key in ("max_steps","accumulation"):
        if type(config[key]) is not int or config[key] < 1: raise ValueError("Invalid "+key)
    if not 0 <= config["warmup_steps"] < config["max_steps"]:
        raise ValueError("warmup_steps must be below max_steps")
    if stop_after is not None and not 1 <= stop_after <= config["max_steps"]:
        raise ValueError("stop_after is outside this run")
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    log_path = output / "training.jsonl"
    if not resume and log_path.exists(): raise FileExistsError(log_path)
    identity = json.loads(json.dumps({"inputs":identity,"model":model.model_config,"training":config},sort_keys=True))
    head = list(model.head.parameters()); head_ids = {id(p) for p in head}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([{"params":other,"lr":config["lr"]}, {"params":head,"lr":config["head_lr"]}],
                                 weight_decay=config["weight_decay"])
    state = {"completed_step":0,"epoch":0,"batch_offset":0,"best_nll":None,"best_step":None}
    restore_rng = None
    checkpoint = None
    if resume:
        resume_path=Path(resume).resolve()
        if resume_path.parent.parent != output:
            raise ValueError("Resume into the original run directory so its best checkpoint remains available")
        snapshots=sorted(p for p in (output/"checkpoints").glob("step-*") if p.name.removeprefix("step-").isdigit())
        if snapshots and resume_path != snapshots[-1].resolve():
            raise ValueError("Resume from the latest checkpoint; older checkpoints cannot rewind this run in place")
        state,restore_rng = _load_checkpoint(model,optimizer,resume,identity)
        checkpoint = str(Path(resume).resolve())
        old = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        prefix = [r for r in old if r["step"] <= state["completed_step"]]
        if [r["step"] for r in prefix] != list(range(1,state["completed_step"]+1)):
            raise ValueError("Training log does not cover the resumed checkpoint")
        if len(prefix) != len(old):
            fd,backup_name=tempfile.mkstemp(prefix=f"training-before-resume-{len(old)}-",suffix=".jsonl",dir=output)
            os.close(fd);backup=Path(backup_name)
            shutil.copy2(log_path,backup)
            log_path.write_text(''.join(json.dumps(r)+"\n" for r in prefix))
    (output / "training_identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    iterator = iter(loader_factory(state["epoch"],state["batch_offset"]))
    if restore_rng is not None: _restore_rng(restore_rng)
    def next_batch():
        nonlocal iterator
        try: batch=next(iterator)
        except StopIteration:
            state["epoch"]+=1;state["batch_offset"]=0
            iterator=iter(loader_factory(state["epoch"],0))
            try: batch=next(iterator)
            except StopIteration: raise ValueError("Training loader is empty") from None
        state["batch_offset"]+=1
        return batch
    while state["completed_step"] < config["max_steps"]:
        step = state["completed_step"]+1
        model.train();optimizer.zero_grad(set_to_none=True)
        multiplier = _rate(step,config["warmup_steps"],config["max_steps"])
        for group,base in zip(optimizer.param_groups,(config["lr"],config["head_lr"])):group["lr"]=base*multiplier
        microbatches=[next_batch() for _ in range(config["accumulation"])]
        sample_count=sum(len(b["targets"]) for b in microbatches)
        loss_value=0.
        for batch in microbatches:
            logits=model(**batch["inputs"])
            targets=batch["targets"].to(logits.device)
            loss=noul_loss(logits,targets,config["brier_weight"]).sum()/sample_count
            loss.backward();loss_value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config["clip_grad_norm"])
        if not torch.isfinite(norm):raise FloatingPointError("Nonfinite training gradient")
        optimizer.step();state["completed_step"]=step
        validation_nll=None;is_best=False
        if validation_fn is not None and (step==config["max_steps"] or config["eval_every"] and step%config["eval_every"]==0):
            model.eval();validation_nll=float(validation_fn(model,step))
            if not math.isfinite(validation_nll):raise FloatingPointError("Nonfinite validation NLL")
            if state["best_nll"] is None or validation_nll<state["best_nll"]:
                state["best_nll"]=validation_nll;state["best_step"]=step;is_best=True
        record={**state,"step":step,"loss":loss_value,"gradient_norm":float(norm),"validation_nll":validation_nll,
                "lr":optimizer.param_groups[0]["lr"],"head_lr":optimizer.param_groups[1]["lr"]}
        with log_path.open("a") as stream:stream.write(json.dumps(record,allow_nan=False)+"\n")
        print(json.dumps({"event":"train_step","step":step,"loss":loss_value,"validation_nll":validation_nll}),flush=True)
        should_stop=stop_after is not None and step>=stop_after
        if is_best or step==config["max_steps"] or should_stop or config["save_every"] and step%config["save_every"]==0:
            checkpoint=_save_checkpoint(model,optimizer,output,state,identity)
        if should_stop:break
    best_step=state["best_step"] or state["completed_step"]
    result={**state,"checkpoint":checkpoint,"best_checkpoint":str(output/"checkpoints"/f"step-{best_step:08d}"),
            "status":"completed" if state["completed_step"]==config["max_steps"] else "paused"}
    (output/"training_status.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
