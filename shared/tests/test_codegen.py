import importlib.util
from pathlib import Path

GEN = Path(__file__).resolve().parents[1] / "codegen" / "gen_c_headers.py"


def test_generated_headers_up_to_date() -> None:
    spec = importlib.util.spec_from_file_location("gen_c_headers", GEN)
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.main(["--check"]) == 0, "run: uv run python shared/codegen/gen_c_headers.py"
