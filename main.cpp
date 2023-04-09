#include <iostream>
#include "CLI11.hpp"
#include "detector.hpp"

int main(int argc, char** argv) {
    CLI::App app("Canny Edge Detector");

    std::string input_file = "test.jpg";
    app.add_option("input", input_file, "Input image(s) path");
    std::string output_file = "test_edges.jpg";
    app.add_option("-o,--output", output_file, "Output image(s) path");
    float sig_percentage = 0.005;
    app.add_option("-s,--sigma", sig_percentage, "Smoothing sigma percentage");
    float high_thr_mul = 2;
    app.add_option("--highthrmul", high_thr_mul, "Value by which to multiply the Otsu threshold to get the high threshold");
    float low_thr_mul = 0.8;
    app.add_option("--lowthrmul", low_thr_mul, "Value by which to multiply the Otsu threshold to get the low threshold");
    bool debug = false;
    app.add_flag("-d,--debug", debug, "Show intermediate images for debugging");
    bool multiple = false;
    app.add_flag("-m,--multiple", multiple, "Do 3D edge detection");
    bool disable24 = false;
    app.add_flag("--disable24", disable24, "Disable 24-connectivity");
    bool opencv = false;
    app.add_flag("--opencv", opencv, "Use OpenCV implementation");

    CLI11_PARSE(app, argc, argv);

    std::cout << "Parameters" << std::endl;
    std::cout << " - Input path: " << input_file << std::endl;
    std::cout << " - Output path: " << output_file << std::endl;
    std::cout << " - Sigma percentage: " << sig_percentage << std::endl;
    std::cout << " - High threshold multiplier: " << high_thr_mul << std::endl;
    std::cout << " - Low threshold multiplier: " << low_thr_mul << std::endl;
    std::cout << " - Multiple flag: " << multiple << std::endl;
    std::cout << " - Disable 24-connectivity flag: " << disable24 << std::endl;
    std::cout << " - OpenCV flag: " << opencv << std::endl;
    std::cout << " - Debug flag: " << debug << std::endl;

    if (opencv)
        opencv_canny(input_file, output_file, sig_percentage, high_thr_mul, low_thr_mul, debug);
    else if (!multiple)
        detect_edges_2d(input_file, output_file, sig_percentage, high_thr_mul, low_thr_mul, debug);
    else
        detect_edges_3d(input_file, output_file, sig_percentage, high_thr_mul, low_thr_mul, disable24, debug);
}
