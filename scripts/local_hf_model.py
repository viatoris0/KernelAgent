#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run KernelAgent workflows with local Hugging Face models (no endpoint)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from utils.providers.local_transformers_provider import LocalTransformersProvider
from triton_kernel_agent import TritonKernelAgent
from triton_kernel_agent.platform_config import get_platform


def _resolve_snapshot_path(
    model_id: str,
    cache_dir: str,
    download_if_missing: bool,
) -> str:
    try:
        return snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        if not download_if_missing:
            raise RuntimeError(
                f"Model '{model_id}' not found in local cache '{cache_dir}'. "
                "Use --download-if-missing to fetch it."
            )
        return snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use local Hugging Face models across KernelAgent workflows."
    )
    parser.add_argument("--model-id", required=True, help="HF model id to run")
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "huggingface" / "hub"),
        help="Local HF cache dir (default: ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download model if missing from local cache",
    )
    sub = parser.add_subparsers(dest="workflow", required=True)

    smoke = sub.add_parser("smoke", help="Quick local prompt smoke test")
    smoke.add_argument(
        "--prompt",
        default="Write a one-sentence hello from a local model.",
        help="Prompt for local smoke test",
    )
    smoke.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens for smoke test",
    )
    smoke.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for smoke test",
    )
    auto = sub.add_parser(
        "autoroute", help="Run Fuser.auto_agent using a local HF model for all stages"
    )
    auto.add_argument("--problem", required=True, help="Absolute path to problem file")
    auto.add_argument("--verify", action="store_true", help="Run final verification")
    auto.add_argument("--no-router-cache", action="store_true")
    auto.add_argument("--dispatch-jobs", type=int, default=2)
    auto.add_argument("--workers", type=int, default=4)
    auto.add_argument("--max-iters", type=int, default=5)
    auto.add_argument("--target-platform", default="cuda", choices=["cuda", "xpu"])
    auto.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to Fuser.auto_agent",
    )

    pipe = sub.add_parser(
        "pipeline", help="Run Fuser.pipeline with local HF model for extract/dispatch/compose"
    )
    pipe.add_argument("--problem", required=True, help="Absolute path to problem file")
    pipe.add_argument("--verify", action="store_true")
    pipe.add_argument("--dispatch-jobs", default="auto")
    pipe.add_argument("--workers", type=int, default=4)
    pipe.add_argument("--max-iters", type=int, default=5)
    pipe.add_argument("--target-platform", default="cuda", choices=["cuda", "xpu"])
    pipe.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to Fuser.pipeline",
    )

    direct = sub.add_parser("direct", help="Run TritonKernelAgent.generate_kernel directly")
    direct.add_argument(
        "--problem-description",
        required=True,
        help="Natural language kernel problem description",
    )
    direct.add_argument(
        "--test-file",
        default="",
        help="Optional test file path (if omitted, test is generated)",
    )
    direct.add_argument("--num-workers", type=int, default=4)
    direct.add_argument("--max-rounds", type=int, default=8)
    direct.add_argument("--target-platform", default="cuda", choices=["cuda", "xpu"])
    direct.add_argument("--test-timeout-s", type=int, default=30)

    return parser


def _run_subprocess(cmd: list[str], env: dict[str, str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, env=env, cwd=str(repo_root)).returncode


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    snapshot_path = _resolve_snapshot_path(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        download_if_missing=args.download_if_missing,
    )
    print(f"Using model_id={args.model_id}")
    print(f"Using snapshot={snapshot_path}")

    env = os.environ.copy()
    env["OPENAI_MODEL"] = args.model_id
    env["LOCAL_MODEL_PATH"] = snapshot_path

    if args.workflow == "smoke":
        provider = LocalTransformersProvider()
        if not provider.is_available():
            raise RuntimeError(
                "LocalTransformersProvider is unavailable. Install torch + transformers."
            )
        os.environ.update({"OPENAI_MODEL": args.model_id, "LOCAL_MODEL_PATH": snapshot_path})
        messages = [{"role": "user", "content": args.prompt}]
        response = provider.get_response(
            args.model_id,
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n=== MODEL RESPONSE ===")
        print(response.content)
        return 0

    if args.workflow == "autoroute":
        cmd = [
            sys.executable,
            "-m",
            "Fuser.auto_agent",
            "--problem",
            args.problem,
            "--router-model",
            args.model_id,
            "--ka-model",
            args.model_id,
            "--extract-model",
            args.model_id,
            "--dispatch-model",
            args.model_id,
            "--compose-model",
            args.model_id,
            "--dispatch-jobs",
            str(args.dispatch_jobs),
            "--workers",
            str(args.workers),
            "--max-iters",
            str(args.max_iters),
            "--target-platform",
            args.target_platform,
        ]
        if args.verify:
            cmd.append("--verify")
        if args.no_router_cache:
            cmd.append("--no-router-cache")
        cmd.extend(args.passthrough)
        return _run_subprocess(cmd, env)

    if args.workflow == "pipeline":
        cmd = [
            sys.executable,
            "-m",
            "Fuser.pipeline",
            "--problem",
            args.problem,
            "--extract-model",
            args.model_id,
            "--dispatch-model",
            args.model_id,
            "--compose-model",
            args.model_id,
            "--dispatch-jobs",
            str(args.dispatch_jobs),
            "--workers",
            str(args.workers),
            "--max-iters",
            str(args.max_iters),
            "--target-platform",
            args.target_platform,
        ]
        if args.verify:
            cmd.append("--verify")
        cmd.extend(args.passthrough)
        return _run_subprocess(cmd, env)

    if args.workflow == "direct":
        os.environ.update({"OPENAI_MODEL": args.model_id, "LOCAL_MODEL_PATH": snapshot_path})
        test_code = None
        if args.test_file:
            test_code = Path(args.test_file).read_text()
        agent = TritonKernelAgent(
            num_workers=args.num_workers,
            max_rounds=args.max_rounds,
            model_name=args.model_id,
            preferred_provider=LocalTransformersProvider,
            target_platform=get_platform(args.target_platform),
            test_timeout_s=args.test_timeout_s,
        )
        result = agent.generate_kernel(
            problem_description=args.problem_description,
            test_code=test_code,
        )
        print(result)
        return 0 if result.get("success") else 1

    parser.error(f"Unknown workflow: {args.workflow}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
