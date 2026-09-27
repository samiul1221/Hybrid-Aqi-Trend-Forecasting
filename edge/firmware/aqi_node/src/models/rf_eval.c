// Generic RandomForest blob evaluator. Format and rules: rf_eval.h, ADR 0009.
#include "rf_eval.h"

#include <string.h>

#define HEADER_SIZE (4 + 2 + 2 + 2 + 2 + 72)
#define NODE_SIZE 9

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static float rdf(const uint8_t *p) {
  uint32_t u = rd32(p);
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

int rf_load(rf_model_t *m, const uint8_t *blob, size_t size, const char *expected_schema) {
  if (size < HEADER_SIZE || memcmp(blob, "AQRF", 4) != 0) return RF_ERR_MAGIC;
  if (rd16(blob + 4) != 1) return RF_ERR_VERSION;
  m->blob = blob;
  m->size = size;
  m->n_outputs = rd16(blob + 6);
  m->n_features = rd16(blob + 8);
  memcpy(m->schema_hash, blob + 12, 72);
  m->schema_hash[71] = '\0';
  if (m->n_outputs == 0 || m->n_outputs > RF_MAX_OUTPUTS || m->n_features >= RF_LEAF) return RF_ERR_BOUNDS;
  if (expected_schema && strcmp(expected_schema, m->schema_hash) != 0) return RF_ERR_SCHEMA;  // I05
  size_t off = HEADER_SIZE;
  for (uint16_t o = 0; o < m->n_outputs; ++o) {
    if (off + 2 > size) return RF_ERR_BOUNDS;
    m->output[o] = blob + off;
    uint16_t n_trees = rd16(blob + off);
    off += 2;
    if (n_trees == 0) return RF_ERR_BOUNDS;
    for (uint16_t t = 0; t < n_trees; ++t) {
      if (off + 4 > size) return RF_ERR_BOUNDS;
      uint32_t n_nodes = rd32(blob + off);
      off += 4;
      if (n_nodes == 0 || n_nodes > 65535u || off + (size_t)n_nodes * NODE_SIZE > size) return RF_ERR_BOUNDS;
      for (uint32_t i = 0; i < n_nodes; ++i) {
        const uint8_t *nd = blob + off + (size_t)i * NODE_SIZE;
        if (nd[0] == RF_LEAF) continue;
        uint16_t l = rd16(nd + 5), r = rd16(nd + 7);
        // children must come after the parent (sklearn layout): no cycles, so rf_predict terminates
        if (nd[0] >= m->n_features || l >= n_nodes || r >= n_nodes || l <= i || r <= i) return RF_ERR_BOUNDS;
      }
      off += (size_t)n_nodes * NODE_SIZE;
    }
  }
  return off == size ? RF_OK : RF_ERR_BOUNDS;
}

void rf_predict(const rf_model_t *m, const float *x, float *y) {
  for (uint16_t o = 0; o < m->n_outputs; ++o) {
    const uint8_t *p = m->output[o];
    uint16_t n_trees = rd16(p);
    p += 2;
    float sum = 0.0f;
    for (uint16_t t = 0; t < n_trees; ++t) {
      uint32_t n_nodes = rd32(p);
      const uint8_t *nodes = p + 4;
      uint32_t i = 0;
      for (;;) {  // terminates: rf_load checked every child index is greater than its parent's
        const uint8_t *nd = nodes + (size_t)i * NODE_SIZE;
        float v = rdf(nd + 1);
        if (nd[0] == RF_LEAF) {
          sum += v;
          break;
        }
        i = (x[nd[0]] <= v) ? rd16(nd + 5) : rd16(nd + 7);
      }
      p = nodes + (size_t)n_nodes * NODE_SIZE;
    }
    y[o] = sum / (float)n_trees;
  }
}
