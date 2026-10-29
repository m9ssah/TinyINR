#pragma once

#include <string>
#include <vector>

struct Image {
  int width = 0;
  int height = 0;
  int channels = 3;

  std::vector<float> data; // row major HWC
};

Image readPPM(const std::string &path);

void writePPM(const std::string &path, const Image &image);
