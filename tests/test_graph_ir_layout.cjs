"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
require("../tools/templates/graph-ir-layout.js");

const {build, getScope, view, focus, neighbors, layout} = globalThis.GraphIRLayout;

function graph(nodes, extra = {}) {
  return {
    name: "layout-test", inputs: {x: {shape: [2, 2], dtype: "float32"}},
    outputs: nodes.length ? nodes.at(-1).outputs : [],
    nodes, constants: {}, value_specs: {}, ...extra
  };
}

function node(id, inputs, output, op = "Add") {
  return {id, op, inputs, outputs: [output], attrs: {}};
}

function connectionPairs(result) {
  return result.edges.map(edge => [edge.source, edge.target]).sort();
}

function assertValidLayout(result) {
  const keys = new Set(result.nodes.map(item => item.key));
  assert.ok(Number.isFinite(result.width));
  assert.ok(Number.isFinite(result.height));
  for (const item of result.nodes) {
    assert.ok(Number.isFinite(item.x));
    assert.ok(Number.isFinite(item.y));
    assert.ok(item.w > 0 && item.h > 0);
  }
  for (const edge of result.edges) {
    assert.ok(keys.has(edge.source), `missing edge source ${edge.source}`);
    assert.ok(keys.has(edge.target), `missing edge target ${edge.target}`);
    assert.ok(edge.tensors.length > 0);
  }
}

test("edges follow named producer/consumer tensors, never node-list adjacency", () => {
  const model = build(graph([
    node("left", ["x"], "a"),
    node("right", ["x"], "b"),
    node("merge", ["a", "b"], "result")
  ]));
  const result = view(model);
  const operationEdges = result.edges.filter(edge => edge.source.startsWith("node:") && edge.target.startsWith("node:"));
  assert.deepEqual(operationEdges.map(edge => [edge.source, edge.target]).sort(), [
    ["node:left", "node:merge"], ["node:right", "node:merge"]
  ]);
  assert.deepEqual(operationEdges.find(edge => edge.source === "node:left").tensors, ["a"]);
  assert.deepEqual(operationEdges.find(edge => edge.source === "node:right").tensors, ["b"]);
  assert.equal(model.producerByTensor.get("b"), "right");
  assert.deepEqual(model.consumersByTensor.get("a"), ["merge"]);
  assert.ok(!connectionPairs(result).some(([source, target]) => source === "node:left" && target === "node:right"));
  const placed = layout(result);
  assertValidLayout(placed);
  assert.equal(placed.nodes.find(item => item.nodeId === "left").rank, placed.nodes.find(item => item.nodeId === "right").rank);
  assert.ok(placed.nodes.find(item => item.nodeId === "merge").y > placed.nodes.find(item => item.nodeId === "left").y);
});

test("scope hierarchy preserves every node and keeps root constant Identity aliases accessible", () => {
  const nodes = [
    node("Identity_0", ["weight"], "weight_alias", "Identity"),
    node("Identity_1", ["weight_alias"], "weight_alias_2", "Identity"),
    node("/backbone/block/Conv", ["x", "weight_alias_2"], "features", "Conv2D"),
    node("/head/Relu", ["features"], "result", "ReLU")
  ];
  const model = build(graph(nodes, {constants: {weight: {shape: [2, 2], dtype: "float32"}}}));
  assert.deepEqual(getScope(model).nodeIds, nodes.map(item => item.id));
  assert.deepEqual([...model.scopes.values()].flatMap(scope => scope.directNodeIds).sort(), nodes.map(item => item.id).sort());
  assert.equal(new Set([...model.scopes.values()].flatMap(scope => scope.directNodeIds)).size, nodes.length);
  const auxiliary = [...model.scopes.values()].find(scope => scope.synthetic);
  assert.ok(auxiliary);
  assert.deepEqual(auxiliary.nodeIds, ["Identity_0", "Identity_1"]);
  assert.deepEqual(getScope(model, "/backbone/").nodeIds, ["/backbone/block/Conv"]);
  const aliasView = view(model, auxiliary.path);
  assert.deepEqual(aliasView.nodes.filter(item => item.kind === "node").map(item => item.nodeId), ["Identity_0", "Identity_1"]);
  assert.ok(aliasView.nodes.filter(item => item.kind === "node").every(item => item.isConstantOnly));
  assert.ok(!aliasView.nodes.some(item => item.tensor === "weight"), "an initializer must not become a fake operator card");
  assertValidLayout(layout(aliasView));
});

test("before/after module Reshape nodes remain separate so the root overview has no invented cycle", () => {
  const model = build(graph([
    node("Identity_0", ["weight"], "weight_alias", "Identity"),
    node("/Reshape", ["x"], "batch_images", "Reshape"),
    node("/img_backbone/Conv", ["batch_images", "weight_alias"], "features", "Conv2D"),
    node("/img_neck/Conv", ["features"], "pyramid", "Conv2D"),
    node("/Reshape_1", ["pyramid"], "camera_features", "Reshape"),
    node("/pts_bbox_head/MatMul", ["camera_features", "weight_alias"], "result", "MatMul")
  ], {constants: {weight: {shape: [2, 2], dtype: "float32"}}}));
  const result = layout(view(model));
  assertValidLayout(result);
  assert.equal(result.hasAggregatedCycles, false);
  assert.ok(result.edges.every(edge => !edge.cycle));
  const expected = [
    "input:x", "node:/Reshape", "scope:img_backbone", "scope:img_neck",
    "node:/Reshape_1", "scope:pts_bbox_head", "output:result"
  ];
  for (let index = 0; index + 1 < expected.length; index++) {
    assert.ok(result.edges.some(edge => edge.source === expected[index] && edge.target === expected[index + 1]));
    assert.ok(result.nodes.find(item => item.key === expected[index]).rank < result.nodes.find(item => item.key === expected[index + 1]).rank);
  }
});

test("a real scope aggregation cycle is explicitly marked without claiming the tensor DAG cycles", () => {
  const model = build(graph([
    node("/scope_a/first", ["x"], "a"),
    node("/scope_b/middle", ["a"], "b"),
    node("/scope_a/last", ["b"], "result")
  ]));
  const aggregate = layout(view(model));
  assertValidLayout(aggregate);
  assert.equal(aggregate.hasAggregatedCycles, true);
  assert.deepEqual(aggregate.edges.filter(edge => edge.cycle).map(edge => edge.tensors), [["a"], ["b"]]);
  assert.equal(aggregate.nodes.find(item => item.scope === "scope_a").component, aggregate.nodes.find(item => item.scope === "scope_b").component);
  const exact = layout(focus(model, "/scope_b/middle", 1));
  assert.equal(exact.hasAggregatedCycles, false);
  assert.ok(exact.edges.every(edge => !edge.cycle));
});

test("pagination exposes the selected operator with truthful incoming and outgoing boundary tensors", () => {
  const model = build(graph([
    node("/block/first", ["x"], "a"),
    node("/block/middle", ["a"], "b"),
    node("/block/last", ["b"], "result")
  ]));
  const result = view(model, "block", {offset: 1, limit: 1});
  assert.equal(result.total, 3);
  assert.equal(result.offset, 1);
  assert.equal(result.truncated, true);
  assert.deepEqual(result.nodes.filter(item => item.kind === "node").map(item => item.nodeId), ["/block/middle"]);
  assert.deepEqual(result.edges.find(edge => edge.target === "node:/block/middle").tensors, ["a"]);
  assert.deepEqual(result.edges.find(edge => edge.source === "node:/block/middle").tensors, ["b"]);
  assert.ok(result.nodes.some(item => item.kind === "external" && item.direction === "input"));
  assert.ok(result.nodes.some(item => item.kind === "external" && item.direction === "output"));
  assertValidLayout(layout(result));
});

test("node focus provides actual upstream/downstream neighbors and their tensor names", () => {
  const model = build(graph([
    node("/block/first", ["x"], "a"),
    node("/block/middle", ["a"], "b"),
    node("/block/last", ["b"], "result"),
    node("/unrelated/other", ["x"], "unrelated")
  ], {outputs: ["result", "unrelated"]}));
  assert.deepEqual(neighbors(model, "/block/middle", "upstream"), ["/block/first"]);
  assert.deepEqual(neighbors(model, "/block/middle", "downstream"), ["/block/last"]);
  const focused = focus(model, "/block/middle", 1);
  assert.deepEqual(focused.nodes.filter(item => item.kind === "node").map(item => item.nodeId), [
    "/block/first", "/block/middle", "/block/last"
  ]);
  assert.deepEqual(focused.edges.filter(edge => edge.source.startsWith("node:") && edge.target.startsWith("node:")).map(edge => edge.tensors), [["a"], ["b"]]);
  assert.equal(focused.selectedNodeId, "/block/middle");
  assertValidLayout(layout(focused));
});

test("empty graph has a finite, valid empty layout", () => {
  const model = build(graph([], {inputs: {}, outputs: []}));
  const result = layout(view(model));
  assertValidLayout(result);
  assert.deepEqual(result.nodes, []);
  assert.deepEqual(result.edges, []);
  assert.equal(result.hasAggregatedCycles, false);
  assert.equal(result.total, 0);
});
