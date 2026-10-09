"""live-swap-bg: put a real hand-held product clip in front of a new scene, without a video generation model.

Stages, in order (each prints a JSON summary and writes its files into the working directory):
    init      video -> 720x960 frames               seed      prompts.json -> seed mask + review image
    track     SAM 2 masks for every frame           matte     ViTMatte alpha for every frame
    stabilize optional temporal clean-up            motion    camera translation from the old background
    plate     prepare your generated scene          render    composite + encode the MP4
    livephoto export a Live Photo pair (macOS)
Checks: check (score a source before starting), flicker, jitter, crops.
First time: download-models.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from . import device  # noqa: F401  (sets the MPS fallback before torch is imported)
from .clip import Clip


def default_models() -> Path:
    return Path(os.environ.get("LIVE_SWAP_BG_MODELS", "models")).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="live-swap-bg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def stage(name, help_text, workdir=True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        if workdir:
            p.add_argument("workdir", type=Path)
        return p

    def gpu(p):
        p.add_argument("--models", type=Path, default=None, help="model directory (default: $LIVE_SWAP_BG_MODELS or ./models)")
        p.add_argument("--device", default=None, help="cuda, mps or cpu (default: whichever GPU is present)")

    p = sub.add_parser("download-models", help="fetch SAM 2.1 Large and ViTMatte into the model directory")
    p.add_argument("--models", type=Path, default=None, help="model directory (default: $LIVE_SWAP_BG_MODELS or ./models)")

    p = sub.add_parser("init", help="cut the input video into frames")
    p.add_argument("video", type=Path)
    p.add_argument("workdir", type=Path)
    p.add_argument("--start", type=float, help="seconds to skip at the start")
    p.add_argument("--end", type=float, help="seconds to stop at")
    p.add_argument("--ffmpeg", default="ffmpeg")

    p = stage("seed", "seed mask from prompts.json")
    p.add_argument("--prompts", type=Path, help="prompt file (default: <workdir>/prompts.json)")
    gpu(p)
    p = stage("track", "track the seed mask through the clip with SAM 2")
    p.add_argument("--multi", action="store_true", help="track each prompt group as its own object")
    gpu(p)
    p = stage("matte", "soft alpha with ViTMatte")
    p.add_argument("--thin", action="store_true", help="keep thin parts (pump nozzle, brush tip) from eroding away")
    gpu(p)
    p = stage("stabilize", "temporal clean-up; see the module docstring for when each one helps or hurts")
    p.add_argument("--vote-masks", action="store_true", help="5-frame majority vote on masks (run before matte)")
    p.add_argument("--fill-dropouts", action="store_true", help="raise one/two-frame alpha dips (after matte)")
    p.add_argument("--median-alpha", action="store_true", help="5-frame alpha median (after matte)")
    p = stage("motion", "estimate camera translation from the original background")
    p.add_argument("--static", action="store_true", help="keep the new background still")
    p = stage("plate", "prepare the generated scene as the background plate")
    p.add_argument("background", type=Path)
    p.add_argument("--blur", type=int, default=3, help="portrait-mode disc blur radius in pixels")
    p.add_argument("--frame-space", action="store_true", help="the scene was composed as the viewer sees the frame")
    p = stage("render", "composite and encode the MP4")
    p.add_argument("--background", type=Path, help="use this image instead of plate.png")
    p.add_argument("--no-edge-repair", action="store_true")
    p.add_argument("--max-gap-pixels", type=int, default=0)
    p.add_argument("--rests-on-surface", action="store_true", help="product stands on a surface: add a contact shadow")
    p.add_argument("--start", type=int, default=0, help="first frame to render")
    p.add_argument("--count", type=int, help="number of frames to render")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p = stage("livephoto", "export a Live Photo pair (needs the original Live Photo .MOV as input; macOS)")
    p.add_argument("--ffmpeg", default="ffmpeg")

    p = sub.add_parser("check", help="score whether a source clip's background can be tracked, before starting")
    p.add_argument("video", type=Path, nargs="+")
    p = stage("flicker", "find one-frame flashes inside a box of the rendered video")
    p.add_argument("box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--threshold", type=int, default=40)
    stage("jitter", "measure how much the new background shakes in the rendered video")
    stage("crops", "full-resolution crops of the frames where edges usually go wrong")
    return parser


def run(args: argparse.Namespace) -> object:
    command = args.command
    if command == "init":
        from .prepare import init_clip
        clip = init_clip(args.video, args.workdir, start=args.start, end=args.end, ffmpeg=args.ffmpeg)
        return clip.info
    if command == "download-models":
        from .models import download_models
        return download_models(args.models or default_models())
    if command == "check":
        from .qa import source_score
        return {str(v): source_score(v) for v in args.video}
    clip = Clip(args.workdir)
    models = (args.models or default_models()) if hasattr(args, "models") else None
    if command == "seed":
        from .seed import make_seed
        return make_seed(clip, models, prompts_path=args.prompts, device=args.device)
    if command == "track":
        from .track import track
        return track(clip, models, multi=args.multi, device=args.device)
    if command == "matte":
        from .matte import matte
        return matte(clip, models, thin=args.thin, device=args.device)
    if command == "stabilize":
        from . import stabilize
        chosen = [name for name, on in (("vote_masks", args.vote_masks), ("fill_alpha_dropouts", args.fill_dropouts),
                                        ("median_alpha", args.median_alpha)) if on]
        if not chosen:
            raise SystemExit("pick at least one of --vote-masks, --fill-dropouts, --median-alpha")
        return {name: getattr(stabilize, name)(clip) for name in chosen}
    if command == "motion":
        from .motion import estimate_motion
        result = estimate_motion(clip, static=args.static)
        return {k: v for k, v in result.items() if k not in ("positions", "quality")}
    if command == "plate":
        from .plate import prepare_plate
        return prepare_plate(clip, args.background, blur=args.blur, frame_space=args.frame_space)
    if command == "render":
        from .render import render
        return render(clip, background=args.background, edge_repair=not args.no_edge_repair,
                      max_gap_pixels=args.max_gap_pixels, rests_on_surface=args.rests_on_surface,
                      start=args.start, count=args.count, ffmpeg=args.ffmpeg)
    if command == "livephoto":
        from .livephoto import export_live_photo
        return export_live_photo(clip, ffmpeg=args.ffmpeg)
    if command == "flicker":
        from .qa import flicker_scan
        return flicker_scan(clip, tuple(args.box), top=args.top, threshold=args.threshold)
    if command == "jitter":
        from .qa import background_jitter
        return background_jitter(clip)
    if command == "crops":
        from .qa import edge_crops
        return edge_crops(clip)
    raise SystemExit(f"unknown command {command}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
