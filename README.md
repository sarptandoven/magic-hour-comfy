# magic-hour-comfy

Magic Hour ComfyUI custom nodes for AI video/image generation experiments.

## What is included

This repository currently contains one ComfyUI custom-node pack:

- `magic_hour_nodes_clean 4/`
  - `MagicHourFlashUNetCompiler`
  - `MagicHourSmartQuantCascade`
  - `MagicHourSamplerCustomAdvancedEarlyExit`

No workflow JSON files are currently tracked in this repository.

## Install into ComfyUI

Clone or copy this repository under your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/sarptandoven/magic-hour-comfy.git
```

Then restart ComfyUI. The nodes should appear under the `magic hour/model` and `magic hour/sampling` categories.

## Requirements

Run these nodes inside a working ComfyUI Python environment. ComfyUI provides the core node loader and PyTorch runtime expected by the node modules.

Optional features require additional packages:

- `bitsandbytes` for the Q8/Q4 paths in `MagicHourSmartQuantCascade`
- CUDA support for `MagicHourFlashUNetCompiler`

## Local checks

A lightweight syntax check can be run without launching ComfyUI:

```bash
python3 -m compileall "magic_hour_nodes_clean 4"
```

Generated caches, local environment files, and credentials should stay out of Git.
