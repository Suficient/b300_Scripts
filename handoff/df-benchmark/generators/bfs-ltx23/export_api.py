"""Patch the BFS v3 workflow for df-benchmark and convert it to ComfyUI's API format.

  export_api.py UI_WORKFLOW.json OUT_API.json [COMFY_URL]

The conversion runs in ComfyUI's own frontend (headless Chromium): Set/Get nodes and bypassed nodes only
resolve there. The changes are the ones fake-pipeline's README makes by hand, plus the clip length."""
import json
import sys
import urllib.request

from playwright.sync_api import sync_playwright

LORA = "head_swap_v3_rank_adaptive_fro_098.safetensors"
# 573 LoraLoaderModelOnly: the author's local LoRA name -> the downloaded file
# 345 VHS_LoadVideo: skips 28 frames by default; start at the clip's first frame
# 427 "Duration (Seconds)": frames = ((s * 24) // 8) * 8 + 1, so 5.7 s -> 137 frames, the longest that fits a
#     6 s clip (6.0 s would ask for 145 of its 144 frames)
INPUTS = {"video": ("345", "VHS_LoadVideo"), "ref": ("269", "LoadImage"), "seed": ("377", "RandomNoise"),
          "output": ("341", "SaveVideo")}


def main():
    ui_path, out_path = sys.argv[1], sys.argv[2]
    url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8188"
    wf = json.load(open(ui_path))
    nodes = {n["id"]: n for n in wf["nodes"]}
    nodes[573]["widgets_values"][0] = LORA
    nodes[345]["widgets_values"]["skip_first_frames"] = 0
    nodes[427]["widgets_values"][0] = 5.7

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("window.app && window.app.graph", timeout=120_000)
        api = page.evaluate("async wf => { await window.app.loadGraphData(wf);"
                            " return (await window.app.graphToPrompt()).output; }", wf)
        browser.close()

    known = json.load(urllib.request.urlopen(f"{url}/object_info"))
    if missing := sorted({n["class_type"] for n in api.values()} - set(known)):
        sys.exit(f"node types ComfyUI doesn't have (a custom node failed to load?): {missing}")
    for name, (nid, cls) in INPUTS.items():
        if api.get(nid, {}).get("class_type") != cls:
            sys.exit(f"{name}: expected node {nid} to be {cls}; update [comfy.bfs-ltx23] in config.example.toml")
    json.dump(api, open(out_path, "w"), indent=1)
    print(f"wrote {out_path} ({len(api)} nodes)")


if __name__ == "__main__":
    main()
