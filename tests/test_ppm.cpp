#include <stdio.h>
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/ppm.h"

static std::string tmpPath(const char *name) {
  return std::string("test_ppm_") + name + ".ppm";
}

static void writeRaw(const std::string &path, const std::string &bytes) {
  FILE *f = nullptr;
  fopen_s(&f, path.c_str(), "wb");
  assert(f);
  fwrite(bytes.data(), 1, bytes.size(), f);
  fclose(f);
}

static bool throws(void (*fn)()) {
  try {
    fn();
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

static void testRoundTripExact() {
  Image img;
  img.width = 3;
  img.height = 2;
  const int bytes[18] = {10, 32, 255, 0,  1,  2,   128, 64, 200,
                         13, 9,  77,  250, 3, 99, 254, 11, 42};
  for (int k : bytes)
    img.data.push_back(static_cast<float>(k) / 255.0f);

  const std::string path = tmpPath("roundtrip");
  writePPM(path, img);
  Image back = readPPM(path);
  std::remove(path.c_str());

  assert(back.width == 3 && back.height == 2 && back.channels == 3);
  assert(back.data.size() == img.data.size());
  for (size_t i = 0; i < img.data.size(); i++)
    assert(back.data[i] == img.data[i]);
}

// (k + 0.6) / 255 must write byte k + 1. Truncation would write k, biasing
// every pixel half a level dark.
static void testRoundsToNearest() {
  Image img;
  img.width = 1;
  img.height = 1;
  img.data = {(3 + 0.6f) / 255.0f, (100 + 0.6f) / 255.0f,
              (254 + 0.6f) / 255.0f};

  const std::string path = tmpPath("rounding");
  writePPM(path, img);
  Image back = readPPM(path);
  std::remove(path.c_str());

  assert(back.data[0] == 4.0f / 255.0f);
  assert(back.data[1] == 101.0f / 255.0f);
  assert(back.data[2] == 1.0f);
}

static void testClamps() {
  Image img;
  img.width = 1;
  img.height = 1;
  img.data = {-0.5f, 1.5f, 100.0f};

  const std::string path = tmpPath("clamp");
  writePPM(path, img);
  Image back = readPPM(path);
  std::remove(path.c_str());

  assert(back.data[0] == 0.0f);
  assert(back.data[1] == 1.0f);
  assert(back.data[2] == 1.0f);
}

static void testHeaderComment() {
  const std::string path = tmpPath("comment");
  writeRaw(path, std::string("P6\n# made by hand\n2 1\n# another\n255\n") +
                     std::string("\x01\x02\x03\x04\x05\x06", 6));
  Image img = readPPM(path);
  std::remove(path.c_str());

  assert(img.width == 2 && img.height == 1);
  assert(img.data[0] == 1.0f / 255.0f);
  assert(img.data[5] == 6.0f / 255.0f);
}

static void readAsciiPpm() {
  const std::string path = tmpPath("ascii");
  writeRaw(path, "P3\n1 1\n255\n0 0 0\n");
  try {
    readPPM(path);
  } catch (...) {
    std::remove(path.c_str());
    throw;
  }
}

static void readTruncated() {
  const std::string path = tmpPath("truncated");
  writeRaw(path, std::string("P6\n2 2\n255\n") + std::string("\x01\x02\x03", 3));
  try {
    readPPM(path);
  } catch (...) {
    std::remove(path.c_str());
    throw;
  }
}

static void readWrongMaxval() {
  const std::string path = tmpPath("maxval");
  writeRaw(path, std::string("P6\n1 1\n65535\n") + std::string(6, '\0'));
  try {
    readPPM(path);
  } catch (...) {
    std::remove(path.c_str());
    throw;
  }
}

static void readMissingFile() { readPPM("test_ppm_does_not_exist.ppm"); }

static void writeOneChannel() {
  Image img;
  img.width = 1;
  img.height = 1;
  img.channels = 1;
  img.data = {0.5f};
  writePPM(tmpPath("one_channel"), img);
}

static void writeSizeMismatch() {
  Image img;
  img.width = 2;
  img.height = 2;
  img.data = {0.0f, 0.0f, 0.0f}; // needs 12
  writePPM(tmpPath("size_mismatch"), img);
}

int main() {
  testRoundTripExact();
  testRoundsToNearest();
  testClamps();
  testHeaderComment();

  assert(throws(readAsciiPpm));
  assert(throws(readTruncated));
  assert(throws(readWrongMaxval));
  assert(throws(readMissingFile));
  assert(throws(writeOneChannel));
  assert(throws(writeSizeMismatch));

  FILE *leftover = nullptr;
  fopen_s(&leftover, tmpPath("one_channel").c_str(), "rb");
  assert(leftover == nullptr);

  std::cout << "test_ppm PASS\n";
  return 0;
}
