"""Copy the published inputs and original prompt into a new model workspace."""

import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=["procedural", "photo-textured"], default="procedural")
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    case = root / "benchmarks" / "cafe-selfie-v1"
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new directory")
    if args.stage == "photo-textured":
        if args.previous is None:
            parser.error("photo-textured requires --previous pointing to your own first pass")
        previous = args.previous.resolve()
        if not (previous / "build.py").is_file() or not (previous / "character.blend").is_file():
            parser.error("previous must contain build.py and character.blend")
        if output.is_relative_to(previous):
            parser.error("output must be outside the previous workspace")
        shutil.copytree(previous, output, ignore=shutil.ignore_patterns(".git", ".env", ".env.*", "__pycache__", "*.pyc", "*.blend1", "*.log"))
        shutil.copyfile(previous / "character.blend", output / "source-character.blend")
    else:
        if args.previous is not None:
            parser.error("--previous is only used with photo-textured")
        output.mkdir(parents=True)
    shutil.copytree(case / "inputs", output / "input", dirs_exist_ok=True)
    prompt = "01-procedural.md" if args.stage == "procedural" else "02-photo-textured.md"
    shutil.copyfile(case / "prompts" / prompt, output / "PROMPT.md")
    print(f"Prepared {args.stage} workspace: {output}")
    print("Give PROMPT.md to your model using this directory as its workspace.")


if __name__ == "__main__":
    main()

