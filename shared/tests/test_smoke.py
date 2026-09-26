import aqi_shared.aqi
import aqi_shared.schema_loader
import aqi_shared.time_utils


def test_imports() -> None:
    assert aqi_shared.aqi.__doc__
    assert aqi_shared.time_utils.__doc__
    assert aqi_shared.schema_loader.__doc__
