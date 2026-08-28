from pathlib import Path
import yaml
from .models import Chunk

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}, text
    metadata = yaml.safe_load(parts[1]) or {}
    return metadata, parts[2].lstrip()

def load_chunks(kb_dir: str | Path) -> list[Chunk]:
    kb_dir = Path(kb_dir)
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(raw)
        current_heading = metadata.get("title", path.stem)
        buffer: list[str] = []
        section_no = 0

        def flush():
            nonlocal section_no, buffer, current_heading
            text = "\n".join(buffer).strip()
            if text:
                section_no += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{path.stem}:{section_no}",
                        filename=path.name,
                        heading=current_heading,
                        text=text,
                        metadata=dict(metadata),
                    )
                )
            buffer = []

        for line in body.splitlines():
            if line.startswith("#"):
                flush()
                current_heading = line.lstrip("#").strip()
            else:
                buffer.append(line)
        flush()
    return chunks
