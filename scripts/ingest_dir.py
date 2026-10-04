"""批量导入一个目录下的所有文档：python scripts/ingest_dir.py <目录>"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import get_services  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python scripts/ingest_dir.py <目录路径>")
        return
    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"目录不存在：{root}")
        return

    services = get_services()
    extension_set = set(services.settings.supported_extensions)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extension_set)
    if not files:
        print(f"目录中没有支持的文件（{'/'.join(sorted(extension_set))}）")
        return

    print(f"共发现 {len(files)} 个文件，开始导入…\n")
    succeeded = skipped = failed = 0
    for path in files:
        outcome = services.pipeline.ingest_path(path)
        icon = {"succeeded": "✅", "skipped": "⏭️", "failed": "❌"}.get(outcome.status, "•")
        print(f"{icon} {path.name}：{outcome.message}")
        succeeded += outcome.status == "succeeded"
        skipped += outcome.status == "skipped"
        failed += outcome.status == "failed"
    print(f"\n完成：成功 {succeeded}，跳过 {skipped}，失败 {failed}")


if __name__ == "__main__":
    main()
