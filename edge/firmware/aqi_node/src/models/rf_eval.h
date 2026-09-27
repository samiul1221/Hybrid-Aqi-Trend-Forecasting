// Generic RandomForest evaluator for OTA-loadable model blobs ([S13.2], [S05] model_runtime).
// Pure C (no Arduino headers): the same file is compiled for the host parity check.
// Blob format v1 (little-endian), written by ml/src/aqi_ml/export/rf_blob.py (ADR 0009):
//   "AQRF" | u16 version | u16 n_outputs | u16 n_features | u16 reserved | char schema_hash[72]
//   per output: u16 n_trees; per tree: u32 n_nodes, then n_nodes x 9-byte nodes
//   node: u8 feature (255 = leaf) | f32 threshold (leaf: value) | u16 left | u16 right
//   split rule: x[feature] <= threshold -> left. Output = mean of the trees' leaf values.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RF_MAX_OUTPUTS 8
#define RF_LEAF 255u
#define RF_OK 0
#define RF_ERR_MAGIC -1
#define RF_ERR_VERSION -2
#define RF_ERR_BOUNDS -3
#define RF_ERR_SCHEMA -4

typedef struct {
  const uint8_t *blob;
  size_t size;
  uint16_t n_outputs;
  uint16_t n_features;
  const uint8_t *output[RF_MAX_OUTPUTS];  // start of each output's forest
  char schema_hash[72];
} rf_model_t;

// Validates the whole blob (sizes, child/feature indices) before use. expected_schema may be NULL.
int rf_load(rf_model_t *m, const uint8_t *blob, size_t size, const char *expected_schema);
// y[n_outputs] = forest predictions for feature vector x[n_features] (finite values expected).
void rf_predict(const rf_model_t *m, const float *x, float *y);

#ifdef __cplusplus
}
#endif
