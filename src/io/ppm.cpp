#include "io/ppm.h"

#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {

const int kMaxVal = 255;
const int kMaxDim = 16384;

using FilePtr =
    std::unique_ptr<FILE, int (*)(FILE *)>; // closes file on destruction

unsigned char toByte(float x) {
  x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
  return static_cast<unsigned char>(x * 255.0f + 0.5f);
}

int readHeaderInt(FILE *file) {
  int c = fgetc(file);

  for (;;) {
    if (c == '#') {
      while (c != '\n' && c != EOF) {
        c = fgetc(file);
      }
    } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      c = fgetc(file);
    } else {
      break;
    }
  }

  if (c < '0' || c > '9')
    throw std::runtime_error("invalid PPM header");

  int value = 0;
  while (c >= '0' && c <= '9') {
    if (value > kMaxDim)
      throw std::runtime_error("PPM header value too large");
    value = value * 10 + (c - '0');
    c = fgetc(file);
  }
  return value;
}

} // namespace

Image readPPM(const std::string &path) {
  FilePtr file(fopen(path.c_str(), "rb"), &fclose);
  if (!file)
    throw std::runtime_error("could not open " + path);

  if (fgetc(file.get()) != 'P' || fgetc(file.get()) != '6')
    throw std::runtime_error(path + ": not a binary PPM (expected P6)");

  Image image;
  image.width = readHeaderInt(file.get());
  image.height = readHeaderInt(file.get());
  const int max_val = readHeaderInt(file.get());

  if (image.width <= 0 || image.height <= 0 || image.width > kMaxDim ||
      image.height > kMaxDim)
    throw std::runtime_error(path + ": invalid PPM dimensions");
  if (max_val != kMaxVal)
    throw std::runtime_error(path + ": only maxval 255 is supported");

  const size_t count = static_cast<size_t>(image.width) * image.height * 3;
  std::vector<unsigned char> bytes(count);
  if (fread(bytes.data(), 1, count, file.get()) != count)
    throw std::runtime_error(path + ": truncated PPM pixel data");

  image.data.resize(count);
  for (size_t i = 0; i < count; i++)
    image.data[i] = static_cast<float>(bytes[i]) / static_cast<float>(kMaxVal);

  return image;
}

void writePPM(const std::string &path, const Image &image) {
  if (image.channels != 3)
    throw std::runtime_error("PPM requires a 3-channel image");
  if (image.width <= 0 || image.height <= 0)
    throw std::runtime_error("PPM requires positive dimensions");
  const size_t count = static_cast<size_t>(image.width) * image.height * 3;
  if (image.data.size() != count)
    throw std::runtime_error("image data size != width * height * channels");

  std::vector<unsigned char> bytes(count);
  for (size_t i = 0; i < count; i++)
    bytes[i] = toByte(image.data[i]);

  FilePtr file(fopen(path.c_str(), "wb"), &fclose);
  if (!file)
    throw std::runtime_error("could not open " + path);

  fprintf(file.get(), "P6\n%d %d\n%d\n", image.width, image.height, kMaxVal);
  if (fwrite(bytes.data(), 1, count, file.get()) != count)
    throw std::runtime_error("failed writing " + path);
}
