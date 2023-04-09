# Compilation

Requirements: OpenCV4, C++20

```
cmake -S. -Bbuild && cmake --build build
```

# Usage

```
./CannyEdgeDetector [OPTIONS] [input]

Positionals:
  input TEXT                  Input image(s) path

Options:
  -h,--help                   Print this help message and exit
  -o,--output TEXT            Output image(s) path
  -s,--sigma FLOAT            Smoothing sigma percentage
  --highthrmul FLOAT          Value by which to multiply the Otsu threshold to get the high threshold
  --lowthrmul FLOAT           Value by which to multiply the Otsu threshold to get the low threshold
  -d,--debug                  Show intermediate images for debugging
  -m,--multiple               Do 3D edge detection
  --disable24                 Disable 24-connectivity
  --opencv                    Use OpenCV implementation
```