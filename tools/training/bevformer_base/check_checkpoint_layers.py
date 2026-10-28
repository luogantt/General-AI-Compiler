"""Check checkpoint call semantics with a small differentiable CPU fixture.

No dataset/model weights or MMCV CUDA extension are needed. This validates the
wrapper's gradients/RNG/dispatch, not full BEVFormer accuracy or GPU memory.
Run with the training Python; optionally add --device cuda to check CUDA AMP.
"""

import argparse
import ast
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class FixtureLayer(nn.Module):
    """Use all feature paths, nested tensors and dropout in a cheap operation."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.25)
        self.calls = 0

    def forward(self, query, key=None, value=None, **kwargs):
        self.calls += 1
        result = query + 2 * key + 3 * value
        for name in ("bev_pos", "query_pos", "key_pos", "ref_2d", "ref_3d",
                     "reference_points_cam", "prev_bev"):
            if kwargs[name] is not None:
                result = result + kwargs[name]
        result = result + kwargs["attn_masks"][0] + kwargs["nested"]["bias"][0]
        return self.dropout(torch.tanh(self.projection(result)))


def load_wrapper():
    # Execute the actual wrapper/helpers, replacing only the expensive upstream
    # base class. This keeps the test independent of MMCV and its CUDA extension.
    path = Path(__file__).with_name("checkpoint_layers.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(torch=torch, checkpoint=checkpoint, BEVFormerLayer=FixtureLayer)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["CheckpointedBEVFormerLayer"]


def run(model, tensors, device, seed, gradients=True):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.set_grad_enabled(gradients):
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(tensors[0], tensors[1], tensors[2],
                           bev_pos=tensors[3], query_pos=tensors[4], key_pos=tensors[5],
                           ref_2d=tensors[6], ref_3d=tensors[7],
                           reference_points_cam=tensors[8], prev_bev=tensors[9],
                           attn_masks=[tensors[10], None], nested={"bias": (tensors[11],)},
                           bev_h=2, bev_w=2,
                           spatial_shapes=torch.tensor([[2, 2]], device=device),
                           level_start_index=torch.tensor([0], device=device))
        if gradients:
            output.float().square().sum().backward()
    rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    return output.detach(), rng


def assert_same(a, b, label, device):
    tolerance = 2e-3 if device.type == "cuda" else 1e-6
    if not torch.allclose(a, b, atol=tolerance, rtol=tolerance):
        raise AssertionError("Checkpoint mismatch: " + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    cls = load_wrapper()
    torch.manual_seed(13)
    eager = FixtureLayer().to(device).train()
    recomputed = cls().to(device).train()
    recomputed.load_state_dict(eager.state_dict())
    inputs = [torch.randn(2, 4, device=device, requires_grad=True) for _ in range(12)]
    copied = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    expected, expected_rng = run(eager, inputs, device, 31)
    actual, actual_rng = run(recomputed, copied, device, 31)
    assert_same(expected, actual, "output", device)
    assert torch.equal(expected_rng, actual_rng), "Dropout RNG state changed after backward."
    for index, (left, right) in enumerate(zip(inputs, copied)):
        assert left.grad is not None and right.grad is not None
        assert_same(left.grad, right.grad, "input %d (including key/value and nested tensors)" % index, device)
    for (name, left), (_, right) in zip(eager.named_parameters(), recomputed.named_parameters()):
        assert_same(left.grad, right.grad, "parameter " + name, device)
    assert eager.calls == 1 and recomputed.calls == 2, "Layer must recompute exactly once."

    # Real BEVFormer commonly passes the same FPN tensor as both key and value.
    eager.zero_grad(); recomputed.zero_grad()
    aliased = [item.detach().clone().requires_grad_() for item in inputs]
    aliased_cp = [item.detach().clone().requires_grad_() for item in inputs]
    aliased[2] = aliased[1]
    aliased_cp[2] = aliased_cp[1]
    expected, _ = run(eager, aliased, device, 53)
    actual, _ = run(recomputed, aliased_cp, device, 53)
    assert_same(expected, actual, "aliased key/value output", device)
    assert_same(aliased[1].grad, aliased_cp[1].grad, "aliased key/value gradient sum", device)

    # Different layer instances in the same graph must keep their own method.
    chain_eager = nn.ModuleList([copy.deepcopy(eager), copy.deepcopy(eager)]).to(device)
    chain_cp = nn.ModuleList([cls().to(device), cls().to(device)])
    for i, (left, right) in enumerate(zip(chain_eager, chain_cp)):
        with torch.no_grad():
            left.projection.weight.add_(i * 0.2)
        right.load_state_dict(left.state_dict())
        left.zero_grad(); right.zero_grad()
    chain_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    chain_copied = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    def chain_forward(layers, items):
        torch.manual_seed(71)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(71)
        query = items[0]
        for layer in layers:
            query = layer(query, items[1], items[2], bev_pos=None, query_pos=None,
                          key_pos=None, ref_2d=None, ref_3d=None,
                          reference_points_cam=None, prev_bev=None,
                          attn_masks=[items[10]], nested={"bias": [items[11]]})
        query.square().sum().backward()
        return query.detach()

    assert_same(chain_forward(chain_eager, chain_inputs), chain_forward(chain_cp, chain_copied),
                "two different layers", device)
    for index in (0, 1, 2, 10, 11):
        assert_same(chain_inputs[index].grad, chain_copied[index].grad, "chained input", device)
    for left, right in zip(chain_eager.parameters(), chain_cp.parameters()):
        assert_same(left.grad, right.grad, "chained parameter", device)

    for training, gradients in ((True, False), (False, True), (False, False)):
        model = cls().to(device).train(training)
        run(model, [item.detach().clone().requires_grad_(gradients) for item in inputs],
            device, 41, gradients=gradients)
        assert model.calls == 1, "History/eval must use the original direct path."
    parameter_only = cls().to(device).train()
    run(parameter_only, [item.detach().clone() for item in inputs], device, 47)
    assert parameter_only.calls == 1 and parameter_only.projection.weight.grad is not None
    print("PASS: output, all input/parameter gradients, nested tensors, dropout RNG, "
          "aliased key/value, one recomputation per layer, independent layer binding, "
          "history/eval and parameter-only direct paths; device=" + str(device))


if __name__ == "__main__":
    main()
