/* Offline Graph IR grouping and layout. Every edge comes from a named tensor. */
(function (global) {
  "use strict";

  const clean = value => String(value || "").replace(/^\/+|\/+$/g, "");
  const leaf = value => clean(value).split("/").pop() || "完整模型";
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);

  function build(graph) {
    const model = {
      graph, nodeById: new Map(), nodeIndex: new Map(), scopes: new Map(),
      nodeScopes: new Map(), producerByTensor: new Map(), consumersByTensor: new Map(),
      incoming: new Map(), outgoing: new Map(), constantOnly: new Set(), edges: []
    };
    model.producers = model.producerByTensor;
    model.consumers = model.consumersByTensor;
    function scope(path) {
      path = clean(path);
      if (!model.scopes.has(path)) {
        const parent = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
        const record = {path, label: leaf(path), parent: path ? parent : null,
          children: [], directNodeIds: [], nodeIds: [], ops: {}, firstIndex: Infinity};
        model.scopes.set(path, record);
        if (path) scope(parent).children.push(path);
      }
      return model.scopes.get(path);
    }
    scope("");
    const records = graph.nodes || [];
    const rootIds = [];
    records.forEach((node, index) => {
      const id = String(node.id);
      model.nodeById.set(id, node);
      model.nodeIndex.set(id, index);
      model.incoming.set(id, new Set());
      model.outgoing.set(id, new Set());
      const normalized = clean(id);
      const path = normalized.includes("/") ? normalized.slice(0, normalized.lastIndexOf("/")) : "";
      if (!path) rootIds.push(id);
      model.nodeScopes.set(id, path);
      scope(path).directNodeIds.push(id);
      (node.outputs || []).forEach(tensor => model.producerByTensor.set(tensor, id));
      (node.inputs || []).forEach(tensor => {
        if (!tensor) return;
        if (!model.consumersByTensor.has(tensor)) model.consumersByTensor.set(tensor, []);
        model.consumersByTensor.get(tensor).push(id);
      });
    });

    // Identify initializer-derived work without deleting or folding it. It stays
    // searchable and visible; the UI may distinguish it from activation work.
    const known = new Set(Object.keys(graph.constants || {}));
    const pending = new Map(records.map(node => [String(node.id),
      new Set((node.inputs || []).filter(tensor => tensor && !known.has(tensor)))]));
    const ready = records.filter(node => (node.inputs || []).length && !pending.get(String(node.id)).size)
      .map(node => String(node.id));
    for (let index = 0; index < ready.length; index++) {
      const id = ready[index];
      if (model.constantOnly.has(id)) continue;
      model.constantOnly.add(id);
      for (const tensor of model.nodeById.get(id).outputs || []) {
        known.add(tensor);
        for (const consumer of model.consumersByTensor.get(tensor) || []) {
          const remaining = pending.get(consumer);
          remaining.delete(tensor);
          if (!remaining.size) ready.push(consumer);
        }
      }
    }
    model.constantTensors = known;

    // Only root-level constant work is grouped. Keep the two real Reshape nodes
    // separate: merging a before-backbone and an after-neck reshape would create
    // a misleading cycle in the initial model overview.
    const rootConstantIds = rootIds.filter(id => model.constantOnly.has(id));
    if (rootConstantIds.length && model.scopes.get("").children.length) {
      let auxiliary = "@root-ops";
      while (model.scopes.has(auxiliary)) auxiliary += "_";
      const group = scope(auxiliary);
      group.label = "权重与常量辅助算子（原图根层）";
      group.synthetic = true;
      group.directNodeIds = rootConstantIds;
      model.scopes.get("").directNodeIds = rootIds.filter(id => !model.constantOnly.has(id));
      rootConstantIds.forEach(id => model.nodeScopes.set(id, auxiliary));
    }
    records.forEach(node => {
      const id = String(node.id);
      let path = model.nodeScopes.get(id);
      while (true) {
        const group = model.scopes.get(path);
        group.nodeIds.push(id);
        group.firstIndex = Math.min(group.firstIndex, model.nodeIndex.get(id));
        group.ops[node.op] = (group.ops[node.op] || 0) + 1;
        if (!path) break;
        path = group.parent;
      }
      (node.inputs || []).forEach((tensor, inputIndex) => {
        const source = model.producerByTensor.get(tensor);
        if (source === undefined) return;
        model.edges.push({source, target: id, tensor, inputIndex});
        model.outgoing.get(source).add(id);
        model.incoming.get(id).add(source);
      });
    });
    for (const group of model.scopes.values()) {
      group.children.sort((a, b) => model.scopes.get(a).firstIndex - model.scopes.get(b).firstIndex);
    }

    return model;
  }

  function getScope(model, path = "") {
    return model.scopes.get(clean(path)) || null;
  }

  function tensorSpec(model, tensor) {
    return model.graph.inputs?.[tensor] || model.graph.value_specs?.[tensor] ||
      model.graph.constants?.[tensor] || null;
  }

  function nodeCard(model, id) {
    const node = model.nodeById.get(id);
    const spec = tensorSpec(model, node.outputs?.[0]);
    return {key: "node:" + id, kind: "node", nodeId: id,
      label: leaf(id), op: node.op, scope: model.nodeScopes.get(id), count: 1,
      nodeIds: [id], ops: {[node.op]: 1}, firstIndex: model.nodeIndex.get(id),
      isConstantOnly: model.constantOnly.has(id), shape: spec?.shape,
      dtype: spec?.dtype, constantInputs: (node.inputs || []).filter(name => model.constantTensors.has(name)).length};
  }

  function scopeCard(model, path) {
    const group = getScope(model, path);
    return {key: "scope:" + path, kind: "scope", scope: path, label: group.label,
      count: group.nodeIds.length, nodeIds: group.nodeIds, ops: {...group.ops},
      firstIndex: group.firstIndex, synthetic: Boolean(group.synthetic),
      constantOnlyCount: group.nodeIds.filter(id => model.constantOnly.has(id)).length};
  }

  function assemble(model, cards, options = {}) {
    const owner = new Map();
    const cardByKey = new Map(cards.map(card => [card.key, card]));
    cards.forEach(card => (card.nodeIds || []).forEach(id => owner.set(id, card.key)));
    const edgeByPair = new Map();
    const boundaryLimit = options.boundaryLimit ?? 16;
    const externalKeys = {input: new Set(), output: new Set()};
    function boundary(tensor, direction, otherId) {
      const graphInput = own(model.graph.inputs, tensor);
      const graphOutput = (model.graph.outputs || []).includes(tensor);
      let key, label, kind, scope;
      if (direction === "input" && graphInput) {
        key = "input:" + tensor; label = tensor; kind = "input";
      } else if (direction === "output" && graphOutput) {
        key = "output:" + tensor; label = tensor; kind = "output";
      } else {
        // Group external connections by their actual exported module scope.
        scope = otherId === undefined ? "" : model.nodeScopes.get(otherId);
        const groupLabel = getScope(model, scope)?.label;
        key = "external-" + direction + ":" + (scope || otherId || tensor);
        label = (direction === "input" ? "来自 " : "送往 ") + (groupLabel || leaf(otherId || tensor));
        kind = "external";
        const seen = externalKeys[direction];
        if (!seen.has(key) && seen.size >= boundaryLimit) {
          key = "external-" + direction + ":@remaining";
          label = direction === "input" ? "其余外部输入" : "其余外部使用者";
          scope = undefined;
        } else seen.add(key);
      }
      if (!cardByKey.has(key)) {
        const spec = tensorSpec(model, tensor);
        cardByKey.set(key, {key, label, kind, direction, tensor, tensors: [],
          count: 0, scope, shape: spec?.shape, dtype: spec?.dtype, firstIndex: direction === "input" ? -1 : Infinity});
      }
      const card = cardByKey.get(key);
      if (!card.tensors.includes(tensor)) card.tensors.push(tensor);
      card.count = card.tensors.length;
      if (card.count > 1) {card.shape = undefined; card.dtype = undefined;}
      return key;
    }
    function add(source, target, tensor) {
      if (source === target) return;
      const key = source + "\u0000" + target;
      if (!edgeByPair.has(key)) edgeByPair.set(key, {source, target, tensors: [], count: 0});
      const edge = edgeByPair.get(key);
      edge.count++;
      if (!edge.tensors.includes(tensor)) edge.tensors.push(tensor);
    }
    for (const edge of model.edges) {
      const source = owner.get(edge.source), target = owner.get(edge.target);
      if (source && target) add(source, target, edge.tensor);
      else if (target) add(boundary(edge.tensor, "input", edge.source), target, edge.tensor);
      else if (source) add(source, boundary(edge.tensor, "output", edge.target), edge.tensor);
    }
    for (const [id, target] of owner) {
      const node = model.nodeById.get(id);
      for (const tensor of node.inputs || []) {
        // Initializers belong in the tensor/weight inspector, not thousands of
        // fake operator cards. Graph inputs remain explicit even when unused.
        if (!tensor || model.producerByTensor.has(tensor) || own(model.graph.constants, tensor)) continue;
        add(boundary(tensor, "input"), target, tensor);
      }
      for (const tensor of node.outputs || []) {
        if ((model.graph.outputs || []).includes(tensor)) add(target, boundary(tensor, "output"), tensor);
      }
    }
    return {nodes: [...cardByKey.values()], edges: [...edgeByPair.values()]};
  }

  function view(model, path = "", options = {}) {
    path = clean(path);
    const group = getScope(model, path);
    if (!group) throw new Error("Unknown graph scope: " + path);
    let entries = group.children.map(child => scopeCard(model, child));
    entries.push(...group.directNodeIds.map(id => nodeCard(model, id)));
    // Module folders precede the root-level weight aliases, regardless of the
    // ONNX node numbering. Within each category original order is preserved.
    const total = entries.length;
    const limit = Math.max(1, Math.min(500, Number(options.limit) || 100));
    const offset = Math.max(0, Math.min(Math.max(0, total - 1), Number(options.offset) || 0));
    entries = entries.slice(offset, offset + limit);
    return {...assemble(model, entries, options), scope: path, total, offset, limit,
      truncated: total > entries.length, scopeNodeCount: group.nodeIds.length};
  }

  function neighbors(model, id, direction = "both") {
    if (!model.nodeById.has(id)) return [];
    const result = new Set();
    if (direction === "both" || direction === "upstream" || direction === "input") {
      for (const other of model.incoming.get(id)) result.add(other);
    }
    if (direction === "both" || direction === "downstream" || direction === "output") {
      for (const other of model.outgoing.get(id)) result.add(other);
    }
    return [...result].sort((a, b) => model.nodeIndex.get(a) - model.nodeIndex.get(b));
  }

  function focus(model, id, radius = 1, options = {}) {
    if (!model.nodeById.has(id)) throw new Error("Unknown graph node: " + id);
    const chosen = new Set([id]);
    let frontier = [id];
    const limit = Math.max(1, Number(options.limit) || 100);
    const distance = Math.max(0, Math.min(5, Number(radius) || 0));
    let truncated = false;
    for (let step = 0; step < distance; step++) {
      const next = [];
      for (const current of frontier) {
        for (const other of neighbors(model, current)) {
          if (chosen.has(other)) continue;
          if (chosen.size >= limit) {truncated = true; continue;}
          chosen.add(other); next.push(other);
        }
      }
      frontier = next;
    }
    const ids = [...chosen].sort((a, b) => model.nodeIndex.get(a) - model.nodeIndex.get(b));
    return {...assemble(model, ids.map(nodeId => nodeCard(model, nodeId)), options),
      scope: model.nodeScopes.get(id), selectedNodeId: id, radius: distance,
      total: ids.length, limit, offset: 0, truncated, isFocus: true};
  }

  function layout(graphView, options = {}) {
    const nodes = graphView.nodes || [], edges = graphView.edges || [];
    const byKey = new Map(nodes.map(node => [node.key, node]));
    const adjacency = new Map(nodes.map(node => [node.key, new Set()]));
    const reverse = new Map(nodes.map(node => [node.key, new Set()]));
    for (const edge of edges) {
      if (!byKey.has(edge.source) || !byKey.has(edge.target)) continue;
      adjacency.get(edge.source).add(edge.target);
      reverse.get(edge.target).add(edge.source);
    }
    // Iterative Kosaraju avoids recursion limits. Aggregating module scopes can
    // introduce a cycle even when the underlying tensor graph is a DAG; those
    // edges are explicitly marked instead of inventing a sequential order.
    const visited = new Set(), finished = [];
    for (const node of nodes) {
      if (visited.has(node.key)) continue;
      visited.add(node.key);
      const stack = [{key: node.key, next: [...adjacency.get(node.key)], index: 0}];
      while (stack.length) {
        const top = stack[stack.length - 1];
        if (top.index < top.next.length) {
          const key = top.next[top.index++];
          if (!visited.has(key)) {
            visited.add(key); stack.push({key, next: [...adjacency.get(key)], index: 0});
          }
        } else {finished.push(top.key); stack.pop();}
      }
    }
    const component = new Map(), components = [];
    for (let i = finished.length - 1; i >= 0; i--) {
      const key = finished[i];
      if (component.has(key)) continue;
      const index = components.length, members = [], stack = [key];
      component.set(key, index);
      while (stack.length) {
        const current = stack.pop(); members.push(current);
        for (const previous of reverse.get(current)) {
          if (!component.has(previous)) {component.set(previous, index); stack.push(previous);}
        }
      }
      components.push(members);
    }
    const outgoing = components.map(() => new Set()), indegree = components.map(() => 0);
    for (const edge of edges) {
      const source = component.get(edge.source), target = component.get(edge.target);
      if (source === undefined || target === undefined || source === target || outgoing[source].has(target)) continue;
      outgoing[source].add(target); indegree[target]++;
    }
    const ranks = components.map(() => 0), ready = [];
    indegree.forEach((degree, index) => {if (!degree) ready.push(index);});
    for (let index = 0; index < ready.length; index++) {
      const current = ready[index];
      for (const next of outgoing[current]) {
        ranks[next] = Math.max(ranks[next], ranks[current] + 1);
        if (--indegree[next] === 0) ready.push(next);
      }
    }
    const layers = new Map();
    for (const node of nodes) {
      const rank = ranks[component.get(node.key)] || 0;
      if (!layers.has(rank)) layers.set(rank, []);
      layers.get(rank).push(node.key);
    }
    const compare = (a, b) => (byKey.get(a).firstIndex ?? Infinity) - (byKey.get(b).firstIndex ?? Infinity) || a.localeCompare(b);
    for (const row of layers.values()) row.sort(compare);
    // Barycentric sweeps reduce avoidable crossing while preserving exact links.
    const order = new Map();
    function updateOrder() {for (const row of layers.values()) row.forEach((key, index) => order.set(key, index));}
    updateOrder();
    const rowRanks = [...layers.keys()].sort((a, b) => a - b);
    for (let pass = 0; pass < 4; pass++) {
      const sequence = pass % 2 ? [...rowRanks].reverse() : rowRanks;
      for (const rank of sequence) {
        const average = key => {
          const connected = [...(pass % 2 ? adjacency : reverse).get(key)].filter(other => component.get(other) !== component.get(key));
          return connected.length ? connected.reduce((sum, other) => sum + order.get(other), 0) / connected.length : order.get(key);
        };
        layers.get(rank).sort((a, b) => average(a) - average(b) || compare(a, b));
        updateOrder();
      }
    }
    const w = options.nodeWidth || 248, h = options.nodeHeight || 92;
    const gapX = options.gapX || 42, gapY = options.gapY || 76, margin = options.margin || 42;
    const widest = Math.max(1, ...[...layers.values()].map(row => row.length));
    const width = margin * 2 + widest * w + Math.max(0, widest - 1) * gapX;
    const placed = new Map();
    for (const [rank, row] of layers) {
      const start = (width - (row.length * w + Math.max(0, row.length - 1) * gapX)) / 2;
      row.forEach((key, index) => placed.set(key, {...byKey.get(key),
        x: start + index * (w + gapX), y: margin + rank * (h + gapY),
        w, h, width: w, height: h, rank, component: component.get(key),
        aggregatedCycle: components[component.get(key)].length > 1}));
    }
    const resultEdges = edges.map(edge => ({...edge,
      cycle: component.get(edge.source) === component.get(edge.target),
      aggregatedCycle: component.get(edge.source) === component.get(edge.target)}));
    return {...graphView, nodes: nodes.map(node => placed.get(node.key)), edges: resultEdges,
      width, height: margin * 2 + (rowRanks.length ? (Math.max(...rowRanks) + 1) * h + Math.max(...rowRanks) * gapY : 0),
      hasAggregatedCycles: components.some(members => members.length > 1)};
  }

  global.GraphIRLayout = {build, getScope, view, focus, neighbors, layout, tensorSpec};
})(typeof globalThis !== "undefined" ? globalThis : window);
