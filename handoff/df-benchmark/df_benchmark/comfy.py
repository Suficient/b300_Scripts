"""A minimal ComfyUI HTTP client. Knows nothing about what a workflow does: inputs are filled in by node id."""
from __future__ import annotations

import copy
import time
import uuid
from pathlib import Path

import requests

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".gif")


def bind(workflow: dict, mapping: dict[str, str], values: dict) -> dict:
    """Copy of the workflow with each value written to the "node_id.field" its mapping names.
    Values whose mapping is empty are left out."""
    wf = copy.deepcopy(workflow)
    for name, value in values.items():
        if target := mapping.get(name):
            node, fieldname = target.split(".", 1)
            if node not in wf:
                raise KeyError(f"input {name!r}: node {node} is not in the workflow")
            wf[node]["inputs"][fieldname] = value
    return wf


class Comfy:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def up(self) -> bool:
        try:
            return requests.get(f"{self.url}/system_stats", timeout=5).ok
        except requests.RequestException:
            return False

    def upload(self, path: Path, name: str | None = None) -> str:
        """Put a file in ComfyUI's input folder (any file type, not only images). Returns the name to use."""
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image", files={"image": (name or path.name, f)},
                              data={"overwrite": "true", "type": "input"}, timeout=300)
        r.raise_for_status()
        d = r.json()
        return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]

    def run(self, workflow: dict, output_node: str, timeout: float) -> dict:
        """Queue the workflow, wait for it, return the saved file ({filename, subfolder, type})."""
        r = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:2000]}")
        pid = r.json()["prompt_id"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            h = requests.get(f"{self.url}/history/{pid}", timeout=60).json().get(pid)
            if h and h.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"workflow failed: {h['status'].get('messages')}")
            if h and h.get("status", {}).get("completed"):
                return _pick_output(h["outputs"], output_node)
            time.sleep(5)
        raise TimeoutError(f"workflow {pid} still running after {timeout:.0f} s")

    def download(self, file: dict, dest: Path) -> None:
        r = requests.get(f"{self.url}/view", params=file, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)


def _pick_output(outputs: dict, output_node: str) -> dict:
    """The first saved video, from output_node if one is set. Nodes list files under different keys
    (VideoHelperSuite: "gifs", core nodes: "images"/"videos"), so every list is searched."""
    for node, out in outputs.items():
        if output_node and node != output_node:
            continue
        for files in out.values():
            for f in files if isinstance(files, list) else []:
                if isinstance(f, dict) and f.get("filename", "").lower().endswith(VIDEO_EXTS):
                    return {k: f.get(k, "") for k in ("filename", "subfolder", "type")}
    raise RuntimeError(f"no video in the workflow outputs (output_node={output_node!r}): {list(outputs)}")
