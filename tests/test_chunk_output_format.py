import json
from pathlib import Path

from src.ingest.chunking import chunk_code, chunk_text
from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor
from src.ingest.pre_processor.markdown_pre_processor import MarkdownPreProcessor


def create_sample_files(tmp_path):
    py_path = tmp_path / "s.py"
    py_path.write_text("""def foo():\n    return 1\n""")
    md_path = tmp_path / "s.md"
    md_path.write_text("""# Title\n\nBody text.""")
    return py_path, md_path


def test_output_format_json(tmp_path):
    py_path, md_path = create_sample_files(tmp_path)

    # Code JSON
    p = PythonPreProcessor()
    doc_code = p.process_file(str(py_path))
    json_str = chunk_code(doc_code, output_format="json")
    parsed = json.loads(json_str)
    assert isinstance(parsed, list) and parsed, "chunk_code json output empty"

    # Text JSON
    m = MarkdownPreProcessor()
    doc_text = m.process_file(str(md_path))
    json_str2 = chunk_text(doc_text, output_format="json")
    parsed2 = json.loads(json_str2)
    assert isinstance(parsed2, list) and parsed2, "chunk_text json output empty"
