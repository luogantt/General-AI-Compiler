"""CPU fixtures for dataset indexing and the three-process report protocol.

Uses the real hook source with lightweight dataset/runner stand-ins. This does
not replace the separate real three-GPU BEVFormer preflight.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch


class Registry:
    def register_module(self):
        return lambda cls: cls


class DatasetFixture:
    def __init__(self):
        self.version = "v1.0-mini"
        self.queue_length = 4
        self.data_infos = [dict(token=str(index), prev=str(index - 1),
                                next=str(index + 1), scene_token="scene")
                           for index in range(8)]
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        return index, [item["token"] for item in self.data_infos[max(0, index - 4):index]]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def load_source():
    root = Path(__file__).resolve().parent
    namespace = dict(
        hashlib=hashlib, math=math, Path=Path, torch=torch, np=np,
        HOOKS=Registry(), DATASETS=Registry(), Hook=object,
        BaseTrainvalNuScenesDataset=DatasetFixture,
        training_support=SimpleNamespace(_last_smoke_validation=None),
        _write_json=write_json,
        _optimizer_step=lambda runner: max(
            [int(state.get("step", 0)) for state in runner.optimizer.state.values()] or [0]))
    for file, names in (
        ("base_training_support.py", {"BaseSmokeTrainDataset"}),
        ("ddp_preflight.py", {"RepeatedBaseSmokeDataset", "summarize_rank_reports",
                               "BaseDDPPreflightHook"}),
    ):
        tree = ast.parse((root / file).read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
        exec(compile(tree, str(root / file), "exec"), namespace)
    return namespace


def check_dataset():
    namespace = load_source()
    dataset = namespace["RepeatedBaseSmokeDataset"]()
    assert len(dataset) == 12 and len(dataset.data_infos) == 8 and len(dataset.flag) == 12
    for index in range(12):
        raw_index, context = dataset.prepare_train_data(index)
        assert raw_index == 4 + index % 4
        assert context == [str(item) for item in range(raw_index - 4, raw_index)]
    for invalid in (-1, 12):
        try:
            dataset.prepare_train_data(invalid)
        except IndexError:
            pass
        else:
            raise AssertionError("Invalid index accepted")


class ModelFixture:
    def __init__(self, parameters):
        self.parameters_by_name = parameters

    def named_parameters(self):
        return iter(self.parameters_by_name.items())


def distributed_worker(rank, rendezvous, work_dir):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=3)
    try:
        namespace = load_source()
        names = ["img_backbone.layer2.0.weight", "img_backbone.layer3.conv_offset.weight",
                 "img_neck.weight", "pts_bbox_head.transformer.encoder.layers.0.attentions.0.weight",
                 "pts_bbox_head.transformer.encoder.layers.0.attentions.1.weight",
                 "pts_bbox_head.transformer.decoder.weight", "pts_bbox_head.cls_branches.weight"]
        parameters = {"module." + name: torch.nn.Parameter(torch.ones(300)) for name in names}
        history = {name: value.detach().clone() for name, value in parameters.items()}
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=7.5e-5)
        runner = SimpleNamespace(model=ModelFixture(parameters), eval_model=ModelFixture(history),
                                 optimizer=optimizer, work_dir=work_dir,
                                 logger=SimpleNamespace(info=lambda message: None))
        hook = namespace["BaseDDPPreflightHook"]()
        hook.before_run(runner)
        for iteration in range(4):
            optimizer.zero_grad()
            loss = sum(value.square().mean() * (rank + 1) for value in parameters.values())
            loss.backward()
            for parameter in parameters.values():
                torch.distributed.all_reduce(parameter.grad)
                parameter.grad.div_(3)
            optimizer.step()
            for name, parameter in parameters.items():
                history[name].copy_(parameter.detach())
            runner.outputs = {"log_vars": {"loss": float(loss.detach())}}
            hook.after_train_iter(runner)
        if rank == 0:
            namespace["training_support"]._last_smoke_validation = dict(
                validation_samples=3, finite_predictions=True, accuracy_metrics_computed=False)
            torch.save(dict(meta=dict(epoch=1, iter=4), optimizer=optimizer.state_dict(),
                            state_dict={name[7:]: value.detach() for name, value in parameters.items()}),
                       str(Path(work_dir) / "epoch_1.pth"))
        hook.after_run(runner)
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-communication", action="store_true")
    args = parser.parse_args()
    check_dataset()
    if not args.skip_communication:
        with tempfile.TemporaryDirectory(prefix="base-ddp-fixture-") as directory:
            root = Path(directory)
            torch.multiprocessing.spawn(distributed_worker,
                args=((root / "rendezvous").as_uri(), directory), nprocs=3, join=True)
            report = json.loads((root / "base_ddp_smoke_check.json").read_text())
            assert report["passed"] is True
            summarize = load_source()["summarize_rank_reports"]
            reports = report["ranks"]
            assert summarize(reports[:2])["passed"] is False
            reports[1]["parameter_probe_sha256"] = "different"
            assert summarize(reports)["passed"] is False
            reports[1]["parameter_probe_sha256"] = reports[0]["parameter_probe_sha256"]
            reports[1]["optimizer_updates_per_iteration"][-1] = 0
            assert summarize(reports)["passed"] is False
    print(json.dumps(dict(passed=True, dataset="8 real positions / 4 targets / 12 sampler indices",
                          three_process_cpu_fixture=not args.skip_communication,
                          real_three_gpu_model_test=False)))


if __name__ == "__main__":
    main()
