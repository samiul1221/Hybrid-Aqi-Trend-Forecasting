// Host test smoke check: proves the host g++ build works. Real tests (AQI, IST, features,
// RF evaluator parity) are added per milestone and read shared/fixtures/.
#include <cassert>
#include <cstdio>

int main() {
  assert(1 + 1 == 2);
  std::puts("test_smoke: ok");
  return 0;
}
