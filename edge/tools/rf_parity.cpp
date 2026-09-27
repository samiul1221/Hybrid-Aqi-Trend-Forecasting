// Host parity runner: evaluates an RF blob with the firmware evaluator (rf_eval.c) on vectors.bin
// and writes predictions to out.bin. Used by aqi_ml.export.rf_blob.host_parity ([S13.2] parity_test).
// vectors.bin: u32 n | u16 n_features | u16 n_outputs | n x n_features float32
#include <cstdio>
#include <cstdint>
#include <vector>

#include "rf_eval.h"

static std::vector<uint8_t> slurp(const char *path) {
  std::vector<uint8_t> b;
  FILE *f = std::fopen(path, "rb");
  if (!f) return b;
  int c;
  while ((c = std::fgetc(f)) != EOF) b.push_back(static_cast<uint8_t>(c));
  std::fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  std::vector<uint8_t> blob = slurp(argv[1]), vec = slurp(argv[2]);
  rf_model_t m;
  int rc = rf_load(&m, blob.data(), blob.size(), nullptr);
  if (rc != RF_OK || vec.size() < 8) {
    std::printf("load error %d\n", rc);
    return 1;
  }
  uint32_t n = vec[0] | (vec[1] << 8) | (vec[2] << 16) | (static_cast<uint32_t>(vec[3]) << 24);
  uint16_t nf = static_cast<uint16_t>(vec[4] | (vec[5] << 8));
  uint16_t no = static_cast<uint16_t>(vec[6] | (vec[7] << 8));
  if (nf != m.n_features || no != m.n_outputs || vec.size() != 8 + static_cast<size_t>(n) * nf * 4) return 3;
  const float *x = reinterpret_cast<const float *>(vec.data() + 8);
  std::vector<float> y(static_cast<size_t>(n) * no);
  for (uint32_t i = 0; i < n; ++i) rf_predict(&m, x + static_cast<size_t>(i) * nf, y.data() + static_cast<size_t>(i) * no);
  FILE *f = std::fopen(argv[3], "wb");
  std::fwrite(y.data(), sizeof(float), y.size(), f);
  std::fclose(f);
  std::printf("ok %u\n", n);
  return 0;
}
