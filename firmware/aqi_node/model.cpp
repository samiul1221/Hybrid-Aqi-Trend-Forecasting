// model.cpp - the GDC-CNN student on TensorFlow Lite Micro (FP32).
// student_model.h (made by notebook 04, copied here by `kaggle_run.py download`) holds the .tflite file as a C array.
// Input [1, 12, 46] = 12 two-hour patches x (2 x 23 features); output [1, 6] = change of ln(1 + PM) / scale.
#include "aqi_node.h"

#if __has_include("student_model.h")            // the trained student, made by notebook 04
#include "student_model.h"
#define HAVE_STUDENT 1
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#else
#define HAVE_STUDENT 0
#endif

#if HAVE_STUDENT
const int ARENA_BYTES = 48 * 1024;              // working memory of the interpreter; the real need is printed below
static uint8_t* arena = nullptr;                // taken from the heap at start-up (static RAM is too small)
static tflite::MicroInterpreter* interpreter = nullptr;

bool setupModel() {
  if (STUDENT_FEATURES != N_FEAT) {             // e.g. a Model B file (17 features)
    snprintf(modelStatus, sizeof(modelStatus), "model wants %d features", STUDENT_FEATURES);
    return false;
  }
  arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_8BIT);   // TFLite wants 16-byte alignment
  if (!arena) {
    strcpy(modelStatus, "model: no memory for arena");
    return false;
  }
  const tflite::Model* model = tflite::GetModel(student_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    strcpy(modelStatus, "model: wrong TFLite version");
    return false;
  }
  static tflite::AllOpsResolver resolver;       // every operator (the 8 MB flash has room)
  static tflite::MicroInterpreter instance(model, resolver, arena, ARENA_BYTES);
  if (instance.AllocateTensors() != kTfLiteOk) {
    strcpy(modelStatus, "model: arena too small");
    return false;
  }
  TfLiteTensor* in = instance.input(0);
  TfLiteTensor* out = instance.output(0);
  if (in->type != kTfLiteFloat32 || in->bytes != N_PATCH * PATCH_LEN * sizeof(float) ||
      out->bytes != N_OUT * sizeof(float)) {
    strcpy(modelStatus, "model: unexpected shapes");
    return false;
  }
  interpreter = &instance;
  arenaUsedBytes = instance.arena_used_bytes();
  snprintf(modelStatus, sizeof(modelStatus), "%s, %u KB arena", STUDENT_NAME, (unsigned)(arenaUsedBytes / 1024));
  Serial.printf("student %s ready: %u of %u arena bytes used\n", STUDENT_NAME, (unsigned)arenaUsedBytes, ARENA_BYTES);
  return true;
}

bool runModel(const float* input, float* output) {
  if (!interpreter) return false;
  memcpy(interpreter->input(0)->data.f, input, N_PATCH * PATCH_LEN * sizeof(float));
  if (interpreter->Invoke() != kTfLiteOk) return false;
  memcpy(output, interpreter->output(0)->data.f, N_OUT * sizeof(float));
  return true;
}

bool modelReady() { return interpreter != nullptr; }

#else   // no student_model.h yet: the node still measures, stores hours and shows everything except the forecast
bool setupModel() { return false; }
bool runModel(const float*, float*) { return false; }
bool modelReady() { return false; }
#endif
