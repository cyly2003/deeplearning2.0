"""GPU-only environment verification for the unified multitask QSAR project."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PackageStatus:
    name: str
    ok: bool
    version: str | None = None
    error: str | None = None


def capture_import(name: str) -> tuple[PackageStatus, Any | None]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return (
            PackageStatus(
                name=name,
                ok=False,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            ),
            None,
        )
    return (
        PackageStatus(
            name=name,
            ok=True,
            version=getattr(module, "__version__", "unknown"),
        ),
        module,
    )


def capture_subprocess_import(name: str, preload_pil: bool = False) -> PackageStatus:
    preload = "from PIL import Image; " if preload_pil else ""
    command = (
        "import importlib; "
        f"{preload}"
        f"module = importlib.import_module('{name}'); "
        "print(getattr(module, '__version__', 'unknown'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return PackageStatus(
            name=name,
            ok=True,
            version=completed.stdout.strip() or "unknown",
        )
    error = completed.stderr.strip() or completed.stdout.strip() or "unknown import failure"
    return PackageStatus(name=name, ok=False, error=error)


def main() -> None:
    result: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        }
    }

    core_names = ["numpy", "pandas", "scipy", "sklearn", "rdkit"]
    core_statuses: list[PackageStatus] = []
    for name in core_names:
        status, _ = capture_import(name)
        core_statuses.append(status)
    result["core_scientific_stack"] = {
        "all_ok": all(item.ok for item in core_statuses),
        "packages": [asdict(item) for item in core_statuses],
    }

    torch_status, torch_module = capture_import("torch")
    torchaudio_status, _ = capture_import("torchaudio")
    torchmetrics_status, _ = capture_import("torchmetrics")

    torch_block: dict[str, Any] = {
        "torch": asdict(torch_status),
        "torchaudio": asdict(torchaudio_status),
        "torchmetrics": asdict(torchmetrics_status),
    }
    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        torch_block["cuda_runtime"] = torch_module.version.cuda
        torch_block["cuda_available"] = cuda_available
        torch_block["device_count"] = int(torch_module.cuda.device_count())
        torch_block["device_name"] = (
            torch_module.cuda.get_device_name(0) if cuda_available else None
        )
    result["torch_runtime"] = torch_block

    torchvision_direct_status = capture_subprocess_import("torchvision")
    torchvision_after_pil_status: PackageStatus | None = None
    pil_status: PackageStatus | None = None

    if not torchvision_direct_status.ok:
        pil_status, _ = capture_import("PIL.Image")
        torchvision_after_pil_status = capture_subprocess_import("torchvision", preload_pil=True)
    result["torchvision"] = {
        "direct_import": asdict(torchvision_direct_status),
        "pil_preload": asdict(pil_status) if pil_status is not None else None,
        "after_pil_preload": (
            asdict(torchvision_after_pil_status)
            if torchvision_after_pil_status is not None
            else None
        ),
        "diagnosis": (
            "direct import fails but succeeds after preloading PIL.Image; "
            "this points to a Windows DLL load-order issue around Pillow/_imaging."
            if (
                not torchvision_direct_status.ok
                and torchvision_after_pil_status is not None
                and torchvision_after_pil_status.ok
            )
            else None
        ),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
