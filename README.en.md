# live-swap-bg

English | [中文](README.md)

Put a **real** hand-held product clip (an iPhone Live Photo or a short video) in front of a new scene:
the hand and the product stay exactly as shot, the background becomes a scene you generate, and it moves with the
original camera. Exports MP4, and Live Photo too.

![Before and after: original on the left, new background on the right](demo/gallery/before-after.gif)

All three come from the same default pipeline: the hand, the nails and the label text are the original pixels, and the
background pans with the original camera. A still overview is in [`demo/gallery/before-after.jpg`](demo/gallery/before-after.jpg).

## Why

- **No video generation model.** The hand and the product are real footage. The tool only segments and mattes
  (SAM 2 + ViTMatte), so the product never warps and the text on its packaging never gets rewritten. AI video models
  work out the product from a single image and often get the packaging wrong, and then the product in the video no
  longer matches what the buyer receives. Only the background uses an image model, and only for one still image; see
  [Generating the background](#generating-the-background).
- **Local GPU, close to zero marginal cost.** On an Apple M4 Mac, a 2.8-second Live Photo (83 frames) takes 5 to 7
  minutes end to end (4.6 and 6.8 minutes in two runs, depending on machine load) and costs nothing beyond
  electricity. For comparison, a pay-per-clip video generation service we used before charged about ¥2 (roughly
  US$0.30) for one 4-second 720p clip.
- **Looks like it was shot there.** The new background pans with the camera motion measured from the original
  background (a moving product in front of a dead-still background gives itself away at once). The product is relit
  to the new scene's light and casts a shadow, and the old wall colour carried on the matte edge is cleaned off.

The pipeline comes out of our own work making Live Photo posts that sell products on Douyin (the Chinese TikTok).
We have made several hundred of them; the docs and the Claude Code skill are written from the mistakes we hit.

## What it can and can't do

- Output is always portrait 3:4, 720×960, 30 fps. The input is centre-cropped to 3:4, never stretched.
- Only camera **translation** is reconstructed: no rotation, zoom, perspective or parallax. The slight shake of a
  hand-held phone is fine; clips where the camera swings a lot are not.
- **Pick your clips.** The original background needs texture (a plain wall gives no camera motion), the hand has to
  enter from an edge of the frame, and neither the hand nor the product may touch the top edge.
  `live-swap-bg check` scores a clip before you start.
- The matte starts from a few **prompt points** on the first frame (one group for the product, one for the hand), and
  this step decides the quality. We hand it to an AI agent: it reads the first frame with a coordinate grid, writes
  the points, then looks at the outline, adds points and fixes it until the outline sits on the real edges. Nobody
  clicks anything. The method is in the skill, see [With an AI agent](#with-an-ai-agent); without an agent you can
  write the points by hand in the same format.
- **A person must watch every result.** Metrics only tell you which frames to look at. When we review our own
  output at full size, about 8 out of 20 clips turn up a problem that has to be fixed.
- Tested on Apple Silicon Macs (MPS). NVIDIA GPUs (CUDA) are supported in the code but untested.
- Live Photo export is macOS only (it uses AVFoundation), and the input must be the original iPhone Live Photo
  `.MOV`. The tool checks the pairing ID and the still-image marker, but you have to confirm the import on the phone
  yourself.

## Install

Needs Python 3.10+ and [ffmpeg](https://ffmpeg.org/); Live Photo export also needs the Xcode command line tools
(`xcode-select --install`).

```bash
git clone https://github.com/Gxz-NGU/live-swap-bg.git
cd live-swap-bg
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
live-swap-bg download-models        # SAM 2.1 Large + ViTMatte, about 1.3 GB, into ./models
```

## Generating the background

The new background is a single still image made with an image model. We use the image tool built into the
[Codex CLI](https://github.com/openai/codex) (signed in with a ChatGPT account), with the model set to
`gpt-5.6-terra`: about a minute per image, 1086×1448 output. All four backgrounds in this repo's demos were made
this way:

```bash
codex exec -m gpt-5.6-terra --sandbox read-only --skip-git-repo-check \
  "Generate one image. Do not write code, do not explain. $(cat demo/eucerin/background-prompt.txt)"
# the image lands under ~/.codex/generated_images/
```

We have not tried other image models; any of them should work if the prompt follows these rules:

- **Background only**: no people, hands, products, text or logos. Unless told not to, models often draw the product in.
- **Portrait 3:4**, with the centre left open and props only in the corners, away from where the hand and product move.
- **About as bright as the original wall**: when the gap is too large, `plate` refuses to colour-match (see
  [Pipeline](#pipeline)). Keep the product and the background apart in tone: no white bottle against a pure white wall.
- **A scene that fits the product**: a bathroom counter for a cleanser, a bedroom window for a fragrance, and no
  repeats within a batch.

Example prompts: [`demo/eucerin/background-prompt.txt`](demo/eucerin/background-prompt.txt), and the ones behind
the three comparisons at the top in [`demo/gallery/scene-prompts/`](demo/gallery/scene-prompts/).

## Quick start (with the bundled demo)

```bash
live-swap-bg init demo/eucerin/IMG_1848.MOV work/eucerin
cp demo/eucerin/prompts.json work/eucerin/      # prompt points on the first frame; for your own clip, have an agent write them, or write them by hand
live-swap-bg seed  work/eucerin                 # check work/eucerin/seed-review.jpg: the outline should sit on the real edges
live-swap-bg track work/eucerin                 # SAM 2 tracks the whole clip (3 to 4 minutes on an M4)
live-swap-bg matte work/eucerin                 # ViTMatte refines the edges (1.5 to 2 minutes on an M4)
live-swap-bg motion work/eucerin                # camera motion from the original background
live-swap-bg plate work/eucerin demo/eucerin/background.png
live-swap-bg render work/eucerin                # -> work/eucerin/out/eucerin.mp4
live-swap-bg crops work/eucerin                 # full-resolution edge crops for review
live-swap-bg livephoto work/eucerin             # -> work/eucerin/out/livephoto/eucerin.JPG + .MOV
```

Every step prints a JSON summary and writes its results into the working directory. The layout is described at the
top of [`live_swap_bg/clip.py`](live_swap_bg/clip.py).

## With an AI agent

This is how we use it day to day: a person supplies the footage and watches the finished clip, and the agent does
every step in between. With Claude Code, copy the skill to where it can find it:

```bash
cp -r skills/live-swap-bg ~/.claude/skills/
```

Then say "swap the background of this video" and give it the clip (bring your own scene image, or let it write the
prompt and generate one with Codex). The skill walks it through the whole pipeline: placing prompt points on the
gridded frame, fixing them from the outline, what to check at each step, how to fix common problems, and which
problems no metric catches and must go to a person. All of it comes from mistakes we actually made; see
[`skills/live-swap-bg/SKILL.md`](skills/live-swap-bg/SKILL.md) (written in Chinese). The skill is a plain Markdown
guide and is not tied to Claude Code: our production batches have also been run by Codex from the same instructions.

## Pipeline

| Step | Command | What it does |
|---|---|---|
| Pick | `check` | whether the original background gives usable camera motion |
| Prepare | `init` | crop to 3:4, 720×960, 30 fps; write the first frame with a coordinate grid |
| Seed | `seed` | first-frame mask from the boxes and points in `prompts.json` |
| Track | `track` | SAM 2 carries the mask through every frame; `--multi` tracks each group as its own object |
| Matte | `matte` | ViTMatte soft edges; `--thin` keeps thin parts such as pump nozzles and brush tips |
| Stabilize | `stabilize` | optional: mask vote, alpha dropout fill, alpha median; each has side effects, see the skill |
| Motion | `motion` | camera translation from the original background; falls back to a still background when the estimate is unreliable |
| Plate | `plate` | portrait-mode blur, and the ring around the foreground matched to the old wall colour; refuses when the scene and the old wall differ too much in brightness, then `--no-match` only blurs |
| Render | `render` | edge repair, relighting, cast shadow, composite and encode |
| Review | `crops` `jitter` `flicker` | edge close-ups, background shake, single-frame flicker scan |
| Export | `livephoto` | Live Photo image + video, with device model, time and location metadata stripped |

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```
The tests use synthetic data and need neither a GPU nor the models.

## License

The code is [MIT](LICENSE). The models are both Apache-2.0: [SAM 2](https://github.com/facebookresearch/sam2) and
[ViTMatte](https://huggingface.co/hustvl/vitmatte-base-distinctions-646). `download-models` fetches them from their
publishers; they are not shipped with this repo. The Eucerin clip in the quick start was shot by the author. The
three original clips in the comparison at the top are promotional footage supplied by the sellers, used only to show
the results and not shipped with this repo. All brands shown belong to their owners.
