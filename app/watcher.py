"""
data/raw/ 디렉토리를 감시하다가 PDF 파일이 추가·수정·삭제되면
해시를 확인하고 변경된 파일만 자동으로 재인덱싱합니다.

실행: venv\Scripts\python.exe -m app.watcher
"""
import time
import threading
import torch  # noqa: F401  (PaddleOCR DLL 충돌 방지)

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.config import RAW_DIR
from app.hash_store import load_hashes, file_hash
from pathlib import Path


# PDF가 복사·이동되는 동안 이벤트가 여러 번 발생하는 것을 막기 위한 디바운스 대기 시간(초)
DEBOUNCE_SECONDS = 3.0


class PDFHandler(FileSystemEventHandler):
    def __init__(self):
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # 파일 생성·수정·삭제 이벤트를 모두 동일하게 처리
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            self._schedule()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            self._schedule()

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            self._schedule()

    def on_moved(self, event):
        # PDF가 raw 폴더 안으로 이동된 경우
        if not event.is_directory and event.dest_path.lower().endswith(".pdf"):
            self._schedule()

    def _schedule(self):
        """연속 이벤트를 DEBOUNCE_SECONDS 후 한 번만 실행하도록 타이머 재설정."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._run_ingest)
            self._timer.start()

    def _run_ingest(self):
        current_files = list(Path(RAW_DIR).glob("*.pdf"))
        stored = load_hashes()

        # 실제로 해시가 다른 파일이 있을 때만 ingest_flow 실행
        changed = [f for f in current_files if file_hash(f) != stored.get(f.name)]
        removed = [name for name in stored if name not in {f.name for f in current_files}]

        if not changed and not removed:
            return

        print(f"\n[watcher] 변경 감지 — 변경 {len(changed)}개 / 삭제 {len(removed)}개")
        try:
            from app.ingest_flow import ingest_flow
            ingest_flow()
        except Exception as e:
            print(f"[watcher] 인덱싱 오류: {e}")


def start_watcher():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    handler = PDFHandler()
    observer = Observer()
    observer.schedule(handler, str(RAW_DIR), recursive=False)
    observer.start()
    print(f"[watcher] 감시 시작: {RAW_DIR}")
    print("[watcher] PDF 파일을 추가·수정·삭제하면 자동으로 인덱싱됩니다. 종료: Ctrl+C\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[watcher] 종료")

    observer.join()


if __name__ == "__main__":
    start_watcher()
