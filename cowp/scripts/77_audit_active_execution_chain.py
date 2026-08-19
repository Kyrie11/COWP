from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

ACTIVE_SHELLS = (
    "NEXT_EXECUTION_V16_8_24_CN.sh",
    "BENCHMARK_V16_8_24_FASTPATHS_CN.sh",
    "PREPARE_COWP_V16_8_24_FAST_DATA_CN.sh",
    "ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Static self-containment audit for the active v16.8.24 build/replay chain.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    scripts_dir = root / "cowp" / "scripts"
    existing_modules = {p.stem for p in scripts_dir.glob("*.py")}

    missing_shells: list[str] = []
    missing_modules: list[dict[str, str]] = []
    missing_configs: list[dict[str, str]] = []
    missing_local_shell_refs: list[dict[str, str]] = []
    shell_syntax_errors: list[dict[str, str]] = []
    module_syntax_errors: list[dict[str, str]] = []
    refs: dict[str, dict[str, list[str]]] = {}

    for rel in ACTIVE_SHELLS:
        p = root / rel
        if not p.is_file():
            missing_shells.append(rel)
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        mods = sorted(set(re.findall(r"cowp\.scripts\.([A-Za-z0-9_]+)", text)))
        cfgs = sorted(set(re.findall(r"configs/[A-Za-z0-9_./-]+\.(?:yaml|yml|json)", text)))
        shrefs = sorted(set(re.findall(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_][A-Za-z0-9_.-]*\.sh)(?![A-Za-z0-9_.-])", text)))
        refs[rel] = {"modules": mods, "configs": cfgs, "shells": shrefs}
        for mod in mods:
            if mod not in existing_modules:
                missing_modules.append({"from": rel, "module": f"cowp.scripts.{mod}"})
        for cfg in cfgs:
            if not (root / cfg).is_file():
                missing_configs.append({"from": rel, "path": cfg})
        for sh in shrefs:
            if sh == rel:
                continue
            if not (root / sh).is_file():
                missing_local_shell_refs.append({"from": rel, "path": sh})
        try:
            proc = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                shell_syntax_errors.append({"from": rel, "error": (proc.stderr or proc.stdout).strip()})
        except Exception as exc:
            shell_syntax_errors.append({"from": rel, "error": repr(exc)})

    # Parse every Python module referenced by the active shell chain. This catches
    # truncated/corrupt source files without importing optional heavy dependencies.
    for rel, info in refs.items():
        for mod in info["modules"]:
            mp = scripts_dir / f"{mod}.py"
            if not mp.is_file():
                continue
            try:
                ast.parse(mp.read_text(encoding="utf-8"), filename=str(mp))
            except Exception as exc:
                module_syntax_errors.append({"from": rel, "module": f"cowp.scripts.{mod}", "error": repr(exc)})

    # Explicitly guard the exact regression that broke v23.
    nonexistent_summary_ref = any(
        any(x["module"].endswith("44_summarize_label_build_profile") for x in missing_modules)
        for _ in [0]
    )
    checks = {
        "active_shells_present": not missing_shells,
        "active_python_modules_present": not missing_modules,
        "active_configs_present": not missing_configs,
        "active_local_shell_refs_present": not missing_local_shell_refs,
        "active_shell_syntax_valid": not shell_syntax_errors,
        "active_module_syntax_valid": not module_syntax_errors,
        "no_v23_summary_module_regression": not nonexistent_summary_ref,
    }
    payload = {
        "schema_version": "cowp_v16_8_24_active_execution_chain_v1",
        "repo_root": str(root),
        "checks": checks,
        "pass": all(checks.values()),
        "missing_shells": missing_shells,
        "missing_modules": missing_modules,
        "missing_configs": missing_configs,
        "missing_local_shell_refs": missing_local_shell_refs,
        "shell_syntax_errors": shell_syntax_errors,
        "module_syntax_errors": module_syntax_errors,
        "active_references": refs,
        "note": "This audits the supported v16.8.24 execution chain only. Historical/deprecated wrappers are not promotion entrypoints.",
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text, encoding="utf-8")
    print(text, end="")
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
