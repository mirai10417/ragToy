import hashlib
import json
from pathlib import Path
from app.config import HASH_STORE_PATH


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_hashes() -> dict[str, str]:
    if HASH_STORE_PATH.exists():
        return json.loads(HASH_STORE_PATH.read_text(encoding="utf-8"))
    return {}


def save_hashes(hashes: dict[str, str]):
    HASH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_STORE_PATH.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")


def diff_files(
    current_files: list[Path],
    stored_hashes: dict[str, str],
) -> tuple[list[Path], list[str], dict[str, str]]:
    """
    Returns:
        changed      : 신규 또는 내용이 바뀐 파일 목록
        removed      : 디렉토리에서 사라진 파일명 목록
        current_hashes: 현재 파일 전체 해시 맵 (저장용)
    """
    current_hashes = {f.name: file_hash(f) for f in current_files}
    current_names = set(current_hashes)

    changed = [f for f in current_files if current_hashes[f.name] != stored_hashes.get(f.name)]
    removed = [name for name in stored_hashes if name not in current_names]

    return changed, removed, current_hashes
