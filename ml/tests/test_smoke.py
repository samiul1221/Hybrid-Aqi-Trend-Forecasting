import importlib

SUBPACKAGES = [
    "data",
    "sim",
    "features",
    "distill",
    "export",
    "eval",
    "models.edge.rf",
    "models.edge.gdc_cnn",
    "models.edge.baselines",
    "models.edge.model_b",
    "baselines.lgbm_teacher",
    "models.teacher",
]


def test_imports() -> None:
    for name in SUBPACKAGES:
        assert importlib.import_module(f"aqi_ml.{name}").__doc__
