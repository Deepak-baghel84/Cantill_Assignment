from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# This import follows the project-root sys.path setup above. The noqa marker only
# suppresses Flake8 E402 (module-level import not at top of file); it is not a runtime error.
from src.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Company A procurement spend pipeline")
    parser.add_argument("--config", default="configs/company_a.yaml", help="Company YAML configuration")
    parser.add_argument("--zero-shot", action="store_true", help="Ignore gold labels for predictions; classify from taxonomy only")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        output = run_pipeline(args.config, force_zero_shot=args.zero_shot)
        summary = output.with_name(output.stem + "__run_summary.json")
        print(f"\nPipeline completed successfully.\nOutput: {output}\nSummary: {summary}")
        return 0
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
