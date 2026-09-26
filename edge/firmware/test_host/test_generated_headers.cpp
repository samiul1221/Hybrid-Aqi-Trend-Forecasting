// Generated headers compile cleanly on the host and carry the spec's shapes ([S06.6], [S08]).
// The C AQI implementation itself (checked against shared/fixtures/aqi_vectors.json) lands in P10.
#include <cassert>
#include <cstdio>
#include <cstring>

#include "aqi/cpcb_tables.gen.h"
#include "features/feature_schema.gen.h"

static_assert(FS_N_CHANNELS == 23, "[S08.1] X[24x23]");
static_assert(FS_N_RF_A == 50, "[S08.3] approx_count 50");
static_assert(FS_N_OUTPUTS == 2 * FS_HORIZONS, "pollutant-major outputs");
static_assert(CPCB_N_BANDS == 6, "six CPCB categories");

int main() {
  for (int k = 1; k < CPCB_N_BANDS; ++k) {
    assert(CPCB_PM25_HI[k] > CPCB_PM25_HI[k - 1]);
    assert(CPCB_PM10_HI[k] > CPCB_PM10_HI[k - 1]);
    assert(CPCB_AQI_LO[k] == CPCB_AQI_HI[k - 1] + 1.0);
  }
  for (int i = 0; i < FS_N_CHANNELS; ++i) assert(FS_CHANNELS[i].name && FS_CHANNELS[i].name[0]);
  int n_b = 0;
  for (int i = 0; i < FS_N_CHANNELS; ++i) n_b += FS_CHANNELS[i].model_b;
  assert(n_b == FS_N_CHANNELS_B);
  for (int i = 0; i < FS_N_RF_B - FS_N_MODEL_B_EXTRA; ++i) assert(!FS_RF[FS_RF_B_FROM_A[i]].pm);
  assert(std::strncmp(FS_SCHEMA_HASH, "sha256:", 7) == 0 && std::strlen(FS_SCHEMA_HASH) == 71);
  std::puts("test_generated_headers: ok");
  return 0;
}
