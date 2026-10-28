#!/usr/bin/env python3
"""Render the reviewed MLIR topology JSON to offline PNG source diagrams.

This reads JSON only: it does not execute HTML, inspect a browser, or modify
the topology. Coordinates and font sizes are logical pixels; --scale controls
PNG resolution. A 100-pixel title band is added above all JSON coordinates,
and a 60-pixel legend band below them.
``labelAt`` is the preferred centre of an edge label. Labels
are moved locally when necessary to avoid covering nodes or other labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[2]
INK = "#172C43"
MUTED = "#52677C"
NODE_COLORS = {
    "data": ("#EDF5FF", "#2875C9"),
    "shared": ("#F2F5F8", "#627B94"),
    "specific": ("#F4EFFF", "#8060B5"),
    "output": ("#EAF7F1", "#2A8A69"),
}
EDGE_COLORS = {
    "data": "#2875C9", "call": "#52677C",
    "register": "#8060B5", "reference": "#218272",
}


def intersects(a, b, gap=0):
    return (a[0] < b[2] + gap and a[2] + gap > b[0]
            and a[1] < b[3] + gap and a[3] + gap > b[1])


class Diagram:
    def __init__(self, view, scale, font, bold_font, code_font):
        header = 100
        # The JSON also drives the HTML, whose heading lives outside the graph.
        # Add the PNG heading without changing the shared coordinate document.
        self.view = {**view,
                     "nodes": [{**n, "y": n["y"] + header} for n in view["nodes"]],
                     "edges": [{**e,
                                "points": [[x, y + header] for x, y in e["points"]],
                                **({"labelAt": [e["labelAt"][0], e["labelAt"][1] + header]}
                                   if "labelAt" in e else {})}
                               for e in view.get("edges", [])],
                     "notes": [{**n, "y": n["y"] + header} for n in view.get("notes", [])]}
        view = self.view
        self.width = int(view.get("width", 1100))
        self.height = int(view["height"]) + header + 60
        if not 200 <= self.width <= 10000 or not 120 <= self.height <= 16000:
            raise ValueError("Diagram dimensions are outside the supported range")
        self.scale = scale
        self.image = Image.new("RGB", (self.width * scale, self.height * scale), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.paths = {"body": font, "title": bold_font, "code": code_font}
        self.fonts = {}
        self.nodes = {n["id"]: n for n in view["nodes"]}
        if len(self.nodes) != len(view["nodes"]):
            raise ValueError("Duplicate node IDs")
        self.node_boxes = [self.box(n) for n in view["nodes"]]
        self.occupied = []
        self.relocated = []

    @staticmethod
    def box(node):
        return (node["x"], node["y"], node["x"] + node["w"], node["y"] + node["h"])

    def font(self, size=16, role="body"):
        key = (size, role)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(self.paths[role]), size * self.scale)
        return self.fonts[key]

    def width_of(self, text, size=16, role="body"):
        return self.draw.textlength(text, font=self.font(size, role)) / self.scale

    def wrap(self, text, width, size=16, role="body"):
        """Wrap Chinese and Latin text; retain code whitespace when it fits."""
        result = []
        for paragraph in str(text).split("\n"):
            if not paragraph:
                result.append("")
                continue
            tokens = re.findall(r"[A-Za-z0-9_./:<>=*%+\-]+|\s+|.", paragraph)
            line = ""
            for token in tokens:
                if self.width_of(line + token, size, role) <= width:
                    line += token
                    continue
                if line:
                    result.append(line.rstrip())
                    line = ""
                token = token.lstrip()
                for char in token:
                    if line and self.width_of(line + char, size, role) > width:
                        result.append(line)
                        line = ""
                    line += char
            if line:
                result.append(line.rstrip())
        return result

    def text(self, xy, text, size=16, role="body", color=INK):
        self.draw.text(tuple(round(v * self.scale) for v in xy), text,
                       font=self.font(size, role), fill=color, anchor="lt")

    def lines(self, xy, lines, size=16, role="body", color=INK, leading=None):
        line_height = leading or size + 7
        for index, line in enumerate(lines):
            self.text((xy[0], xy[1] + index * line_height), line, size, role, color)
        return len(lines) * line_height

    def rectangle(self, box, fill, outline=None, radius=8, width=1):
        self.draw.rounded_rectangle(tuple(round(v * self.scale) for v in box),
                                    radius=radius * self.scale, fill=fill,
                                    outline=outline, width=width * self.scale)

    def line(self, points, color, width=2, dashed=False):
        scaled = [tuple(round(v * self.scale) for v in p) for p in points]
        if not dashed:
            self.draw.line(scaled, fill=color, width=width * self.scale, joint="curve")
            return
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            if not length:
                continue
            for start in range(0, math.ceil(length), 13):
                stop = min(start + 8, length)
                segment = [tuple(round((a[i] + (b[i] - a[i]) * t / length)
                                       * self.scale) for i in (0, 1))
                           for t in (start, stop)]
                self.draw.line(segment, fill=color, width=width * self.scale)

    def edge(self, edge):
        if edge["from"] not in self.nodes or edge["to"] not in self.nodes:
            raise ValueError(f"Unknown endpoint in edge {edge['id']}")
        points = edge["points"]
        if len(points) < 2:
            raise ValueError(f"Edge {edge['id']} needs two or more points")
        color = EDGE_COLORS[edge.get("kind", "call")]
        self.line(points, color, dashed=edge.get("kind") in {"register", "reference"})
        end = points[-1]
        previous = next((p for p in reversed(points[:-1]) if p != end), None)
        if previous is None:
            raise ValueError(f"Edge {edge['id']} has no direction")
        dx, dy = end[0] - previous[0], end[1] - previous[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        arrow = [end, (end[0] - 11 * ux - 5 * uy, end[1] - 11 * uy + 5 * ux),
                 (end[0] - 11 * ux + 5 * uy, end[1] - 11 * uy - 5 * ux)]
        self.draw.polygon([tuple(round(v * self.scale) for v in p) for p in arrow], fill=color)

    def node(self, node):
        box = self.box(node)
        if box[0] < 0 or box[1] < 0 or box[2] > self.width or box[3] > self.height:
            raise ValueError(f"Node {node['id']} falls outside the canvas")
        fill, border = NODE_COLORS[node.get("kind", "shared")]
        title = self.wrap(node["title"], node["w"] - 26, 18, "title")
        body = [line for item in node.get("body", [])
                for line in self.wrap(item, node["w"] - 26)]
        code = [line for item in node.get("code", [])
                for line in self.wrap(item, node["w"] - 38, role="code")]
        needed = 24 + len(title) * 25
        if body:
            needed += 5 + len(body) * 23
        if code:
            needed += 13 + len(code) * 23
        if needed > node["h"]:
            raise ValueError(f"Node {node['id']} requires height {needed}, has {node['h']}; "
                             "enlarge the node or shorten text (minimum body size is 16)")
        self.rectangle(box, fill, border, width=2)
        x, y = node["x"] + 13, node["y"] + 12
        y += self.lines((x, y), title, 18, "title", leading=25)
        if body:
            y += 5
            y += self.lines((x, y), body, leading=23)
        if code:
            y += 7
            self.rectangle((x - 5, y - 3, box[2] - 9, y + len(code) * 23 + 3), "#FFFFFF", radius=4)
            self.lines((x + 3, y), code, role="code", leading=23)

    def edge_label(self, edge):
        if not edge.get("label"):
            return
        text_lines = self.wrap(edge["label"], 275)
        width = max(self.width_of(t) for t in text_lines) + 16
        height = len(text_lines) * 23 + 8
        anchor = edge.get("labelAt", edge["points"][len(edge["points"]) // 2])
        offsets = [(0, 0)]
        for distance in (24, 44, 68, 92, 120, 152):
            offsets.extend([(0, -distance), (0, distance), (-distance, 0), (distance, 0),
                            (-distance, -distance), (distance, -distance),
                            (-distance, distance), (distance, distance)])
        obstacles = self.node_boxes + self.occupied
        chosen = None
        for dx, dy in offsets:
            x, y = anchor[0] + dx, anchor[1] + dy
            box = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
            if box[0] < 5 or box[1] < 5 or box[2] > self.width - 5 or box[3] > self.height - 5:
                continue
            if not any(intersects(box, other, 4) for other in obstacles):
                chosen = box
                if dx or dy:
                    self.relocated.append(edge["id"])
                break
        if chosen is None:
            raise ValueError(f"No unobstructed space for edge label {edge['id']}; adjust labelAt")
        self.occupied.append(chosen)
        self.rectangle(chosen, "white", radius=3)
        self.lines((chosen[0] + 8, chosen[1] + 4), text_lines,
                   color=EDGE_COLORS[edge.get("kind", "call")], leading=23)

    def render(self):
        y = 20
        title = self.wrap(self.view["title"], self.width - 48, 26, "title")
        y += self.lines((24, y), title, 26, "title", leading=34)
        subtitle = self.wrap(self.view.get("subtitle", ""), self.width - 48)
        if subtitle:
            y += 4 + self.lines((24, y + 4), subtitle, color=MUTED)
        self.occupied.append((16, 12, self.width - 16, y + 6))
        for node in self.view["nodes"]:
            if intersects(self.box(node), self.occupied[0]):
                raise ValueError(f"Node {node['id']} overlaps diagram title; move it below y={y+6}")
        for edge in self.view.get("edges", []):
            self.edge(edge)
        for node in self.view["nodes"]:
            self.node(node)
        for note in self.view.get("notes", []):
            note_lines = self.wrap(note["text"], self.width - note["x"] - 18)
            height = len(note_lines) * 23
            box = (note["x"], note["y"], note["x"] + max(self.width_of(t) for t in note_lines), note["y"] + height)
            if box[3] > self.height or any(intersects(box, n) for n in self.node_boxes):
                raise ValueError(f"Note overlaps a node or canvas boundary: {note['text']}")
            self.lines((note["x"], note["y"]), note_lines, color=MUTED)
            self.occupied.append(box)
        for edge in self.view.get("edges", []):
            self.edge_label(edge)
        legend_y = self.height - 40
        for x, kind, label in ((24, "data", "数据 / 返回结果"),
                               (295, "call", "调用"),
                               (550, "register", "注册"),
                               (805, "reference", "引用 / 包含关系")):
            self.line([(x, legend_y + 11), (x + 36, legend_y + 11)],
                      EDGE_COLORS[kind], dashed=kind in {"register", "reference"})
            self.lines((x + 45, legend_y), [label], color=EDGE_COLORS[kind])
        return self.image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "tools/docs/mlir_source_topology.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "docs/assets/mlir-source")
    parser.add_argument("--view", action="append", help="Render only this view ID (repeatable)")
    parser.add_argument("--scale", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--bold-font", type=Path, default=Path("C:/Windows/Fonts/msyhbd.ttc"))
    parser.add_argument("--code-font", type=Path, default=Path("C:/Windows/Fonts/consola.ttf"))
    args = parser.parse_args()
    for font in (args.font, args.bold_font, args.code_font):
        if not font.is_file():
            parser.error(f"Font not found: {font}; provide the corresponding font option")
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    views = data["views"]
    ids = [v["id"] for v in views]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", v) for v in ids):
        parser.error("View IDs must be unique file-safe names")
    if args.view and set(args.view) - set(ids):
        parser.error("Unknown view ID")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        if args.view and view["id"] not in args.view:
            continue
        diagram = Diagram(view, args.scale, args.font, args.bold_font, args.code_font)
        image = diagram.render()
        destination = args.output_dir / f"{view['id']}.png"
        image.save(destination, format="PNG", optimize=True)
        print(json.dumps({"view": view["id"], "output": str(destination.resolve()),
                          "pixels": image.size, "relocated_labels": diagram.relocated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
