#include <iostream>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
#include <filesystem>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define FLOAT_EPSILON 0.000001

bool DEBUG = false;

const std::string SUPPORTED_FORMATS[] = {"jpg", "jpeg", "png", "bmp"};

void debug_show_img(const char* name, cv::Mat &img) {
    if (DEBUG) {
        cv::Mat cimg = img.clone();
        auto size = cimg.size();
        float scale = size.height / 500.0;
        cv::resize(cimg, cimg, cv::Size(), 1.0 / scale, 1.0 / scale);
        cv::normalize(cimg, cimg, 255.0, 10.0, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(name, cimg);
        cv::waitKey(0);
        cv::imwrite("debug.jpg", cimg);
    }
}

template<typename T>
void debug_print_array(T *arr, uint size) {
    if (DEBUG) {
        for (uint i = 0; i < size; ++i) {
            std::cout << arr[i] << " ";
        }
    }
    std::cout << std::endl;
}

bool is_format_supported(const std::string &format) {
    for (auto &supported_format : SUPPORTED_FORMATS) {
        if (format == supported_format || format == "." + supported_format) {
            return true;
        }
    }
    return false;
}

std::tuple<double, int> threshold_and_connect_image(cv::Mat &img, cv::Mat &out_edge_img, cv::Mat &out_low_thr, cv::Mat &out_labels, 
    float sig_percentage, float high_thr_mul = 1.0, float low_thr_mul = 0.5) {
        img.convertTo(img, CV_32FC1, 1/255.0);

        auto size = img.size();

        // ----- SMOOTHING -----
        if (sig_percentage > FLOAT_EPSILON) {
            float sig1 = std::min(size.height, size.width) * abs(sig_percentage);
            uint tsig1 = round(sig1 * 3.0);
            cv::Mat kernel = cv::getGaussianKernel(tsig1 + (tsig1 + 1) % 2, sig1, CV_32F);
            cv::sepFilter2D(img, img, -1, kernel, kernel.t());
            debug_show_img("Blurred image", img);
        }
        // ----- SMOOTHING -----

        // ----- DERIVATION -----
        cv::Mat grad_x, grad_y;

        cv::Mat ker_x1, ker_x2, ker_y1, ker_y2;
        // Gets separated Sobel or Scharr kernels (depending on ksize param)
        cv::getDerivKernels(ker_x1, ker_x2, 1, 0, 3, false, CV_32F);
        cv::getDerivKernels(ker_y1, ker_y2, 0, 1, 3, false, CV_32F);
        cv::sepFilter2D(img, grad_x, -1, ker_x1, ker_x2.t());
        cv::sepFilter2D(img, grad_y, -1, ker_y1, ker_y2.t());

        cv::Mat grad_mag, grad_angle;
        cv::cartToPolar(grad_x, grad_y, grad_mag, grad_angle);

        debug_show_img("Gradient Magnitude", grad_mag);
        // ----- DERIVATION -----

        // ----- NON-MAXIMUM SUPPRESSION -----
        float split = M_PI / 8.0;
        float start = -(split / 2.0);
        float splits[] = {start, start + split, start + 2 * split, start + 3 * split};
        float splits_cos[] = {std::cos(splits[0]), std::cos(splits[1]), std::cos(splits[2]), std::cos(splits[3])};
        float splits_sin[] = {std::sin(splits[0]), std::sin(splits[1]), std::sin(splits[2]), std::sin(splits[3])};
        cv::Mat grad_mag_nms = grad_mag.clone();
        for (uint i = 0; i < size.height; ++i) {
            for (uint j = 0; j < size.width; ++j) {
                float nx = grad_x.at<float>(i, j);
                float ny = grad_y.at<float>(i, j);
                float mag = grad_mag.at<float>(i, j);
                if (mag < FLOAT_EPSILON) {
                    grad_mag_nms.at<float>(i, j) = 0.0;
                }
                nx /= mag;
                ny /= mag;

                if (splits_sin[0] <= ny && ny < splits_sin[1]) {
                    // y = 0
                    float mag1 = j - 1 >= 0 ? grad_mag.at<float>(i, j - 1) : 0.0;
                    float mag2 = j + 1 < size.width ? grad_mag.at<float>(i, j + 1) : 0.0;
                    if (mag1 >= mag || mag2 >= mag) {
                        grad_mag_nms.at<float>(i, j) = 0.0;
                    }
                } else if (splits_cos[2] <= nx && nx <= splits_cos[3]) {
                    // x = 0
                    float mag1 = i - 1 >= 0 ? grad_mag.at<float>(i - 1, j) : 0.0;
                    float mag2 = i + 1 < size.height ? grad_mag.at<float>(i + 1, j) : 0.0;
                    if (mag1 >= mag || mag2 >= mag) {
                        grad_mag_nms.at<float>(i, j) = 0.0;
                    }
                } else if (nx > 0 && ny > 0 || nx < 0 && ny < 0) {
                    // y = x
                    float mag1 = i - 1 >= 0 && j - 1 >= 0 ? grad_mag.at<float>(i - 1, j - 1) : 0.0;
                    float mag2 = i + 1 < size.height && j + 1 < size.width ? grad_mag.at<float>(i + 1, j + 1) : 0.0;
                    if (mag1 >= mag || mag2 >= mag) {
                        grad_mag_nms.at<float>(i, j) = 0.0;
                    }
                } else {
                    // y = -x
                    float mag1 = i - 1 >= 0 && j + 1 < size.width ? grad_mag.at<float>(i - 1, j + 1) : 0.0;
                    float mag2 = i + 1 < size.height && j - 1 >= 0 ? grad_mag.at<float>(i + 1, j - 1) : 0.0;
                    if (mag1 >= mag || mag2 >= mag) {
                        grad_mag_nms.at<float>(i, j) = 0.0;
                    }
                }
            }
        }
        // ----- NON-MAXIMUM SUPPRESSION -----

        // ----- THRESHOLDING -----
        cv::normalize(grad_mag_nms, out_edge_img, 255.0, 0.0, cv::NORM_MINMAX, CV_8UC1);

        cv::Mat high_thr;
        double threshold = cv::threshold(out_edge_img, high_thr, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        if (DEBUG) {
            std::cout << "High threshold: " << threshold * high_thr_mul << ", Low threshold: " << threshold * low_thr_mul << std::endl;
        }

        cv::threshold(out_edge_img, out_low_thr, threshold * low_thr_mul, 255, cv::THRESH_BINARY);
        threshold *= high_thr_mul;
        // ----- THRESHOLDING -----

        int num_labels = cv::connectedComponents(out_low_thr, out_labels, 8, CV_32S);

        return std::tuple(threshold, num_labels);
}

void detect_edges_2d(std::string &in, std::string &out, float sig_percentage, 
    float high_thr_mul = 1.0, float low_thr_mul = 0.5, bool debug = false) {
        DEBUG = debug;
        cv::Mat img = cv::imread(in, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Could not read image: " << in << std::endl;
            return;
        }

        auto size = img.size();
        
        cv::Mat edge_img, low_thr, labels;
        const auto &[threshold, num_labels] = 
            threshold_and_connect_image(img, edge_img, low_thr, labels, sig_percentage, high_thr_mul, low_thr_mul);

        // ----- HYSTERESIS THRESHOLDING -----
        bool acc_segments[num_labels] = {false};
        // Check components if at least one pixel is above the high threshold
        for (uint i = 0; i < size.height; ++i) {
            for (uint j = 0; j < size.width; ++j) {
                int32_t label = labels.at<int32_t>(i, j);
                uint8_t mag = edge_img.at<uint8_t>(i, j);
                if (mag >= threshold) {
                    acc_segments[label] = true;
                }
            }
        }

        edge_img = low_thr;

        // Delete not accepted components
        for (uint i = 0; i < size.height; ++i) {
            for (uint j = 0; j < size.width; ++j) {
                int32_t label = labels.at<int32_t>(i, j);
                if (!acc_segments[label]) {
                    edge_img.at<uint8_t>(i, j) = 0;
                }
            }
        }
        // ----- HYSTERESIS THRESHOLDING -----

        debug_show_img("Final image", edge_img);

        cv::imwrite(out, edge_img);
}

void opencv_canny(std::string &in, std::string &out, float sig_percentage, int low_thr, int high_thr, bool debug = false) {
    DEBUG = debug;
    cv::Mat img = cv::imread(in, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not read image: " << in << std::endl;
        return;
    }
    if (sig_percentage > FLOAT_EPSILON) {
        auto size = img.size();
        float sig = std::min(size.height, size.width) * abs(sig_percentage);
        uint tsig = round(sig * 3.0);
        cv::GaussianBlur(img, img, cv::Size(tsig + (tsig + 1) % 2, tsig + (tsig + 1) % 2), sig, sig);
        debug_show_img("Blurred image", img);
    }
    cv::Mat edge_img;
    cv::Canny(img, edge_img, low_thr, high_thr, 3, true);
    debug_show_img("Final image", edge_img);
    cv::imwrite(out, edge_img);
}

class EdgeData {
public:
    std::string img_name;
    cv::Mat edge_img;
    cv::Mat low_thr;
    cv::Mat labels;
    double threshold;
    int num_labels;

    EdgeData(std::string img_name, cv::Mat edge_img, cv::Mat low_thr, cv::Mat labels, double threshold, int num_labels)
        : img_name(img_name), edge_img(edge_img), low_thr(low_thr), labels(labels), threshold(threshold), num_labels(num_labels) {}
};

std::vector<int32_t> get_labels_in_range(cv::Mat &labels, uint row, uint col) {
    auto size = labels.size();
    std::vector<int32_t> labels_in_range;
    for (int i_offset = -1; i_offset <= 1; ++i_offset) {
        for (int j_offset = -1; j_offset <= 1; ++j_offset) {
            int i = row + i_offset;
            int j = col + j_offset;
            if (i < 0 || i >= size.height || j < 0 || j >= size.width) {
                continue;
            }
            int32_t label = labels.at<int32_t>(i, j);
            if (std::find(labels_in_range.begin(), labels_in_range.end(), label) == labels_in_range.end()) {
                labels_in_range.push_back(label);
            }
        }
    }
    return labels_in_range;
}

inline void do_24_connectivity(cv::Mat &ol, cv::Mat &cl, cv::Mat &out_l, uint row, uint col) {
    const int offsets [16][2] = {{0, 2}, {1, 2}, {2, 2}, {2, 1}, {2, 0}, {2, -1}, {2, -2}, {1, -2}, {0, -2}, 
                         {-1, -2}, {-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2}, {-1, 2}};

    if (ol.at<uint8_t>(row, col) > 0) {
        auto size = cl.size();
        for (int i_offset = -1; i_offset <= 1; ++i_offset) {
            for (int j_offset = -1; j_offset <= 1; ++j_offset) {
                int i = row + i_offset;
                int j = col + j_offset;
                if (i < 0 || i >= size.height || j < 0 || j >= size.width) {
                    continue;
                }
                if (cl.at<uint8_t>(i, j) > 0) {
                    return;
                }
            }
        }
        for (const auto &[i_offset, j_offset] : offsets) {
            int i = row + i_offset;
            int j = col + j_offset;
            if (i < 0 || i >= size.height || j < 0 || j >= size.width) {
                continue;
            }
            if (cl.at<uint8_t>(i, j) > 0) {
                out_l.at<uint8_t>(row + (i_offset / 2), col + (j_offset / 2)) = 255;
            }
        }
    }
}

void detect_edges_3d(std::string &in_dir, std::string &out_dir, float sig_percentage, 
    float high_thr_mul = 1.0, float low_thr_mul = 0.5, bool disable_24_connectivity = false, bool debug = false) {
        DEBUG = debug;
        if (!std::filesystem::exists(in_dir)) {
            std::cerr << "Input directory does not exist: " << in_dir << std::endl;
            return;
        }
        if (out_dir.at(0) == ' ')
            out_dir = out_dir.substr(1); 
        if (!std::filesystem::exists(out_dir)) {
            std::cerr << "Output directory does not exist: " << out_dir << std::endl;
            return;
        }

        std::vector<EdgeData> layers;
        std::cout << "\rLoading images                " << std::flush;
        for (const auto &entry : std::filesystem::directory_iterator(in_dir)) {
            std::string img_name = entry.path().filename().string();
            std::string img_ext = entry.path().extension().string();
            if (!is_format_supported(img_ext)) {
                std::cerr << "Image " << img_name << " format not supported: " << img_ext << std::endl;
                continue;
            }
            std::string img_path = entry.path().string();
            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Could not read image: " << img_path << std::endl;
                continue;
            }

            auto size = img.size();
            
            cv::Mat edge_img, low_thr, labels;
            const auto &[threshold, num_labels] = 
                threshold_and_connect_image(img, edge_img, low_thr, labels, sig_percentage, high_thr_mul, low_thr_mul);

            layers.push_back(EdgeData{img_name, edge_img, low_thr, labels, threshold, num_labels});
        }

        uint num_layers = layers.size();

        std::sort(layers.begin(), layers.end(), [](const EdgeData &l1, const EdgeData &l2) {
            return l1.img_name.compare(l2.img_name) < 0;
        });

        // ----- HYSTERESIS THRESHOLDING -----
        // Update labels
        std::cout << "\rUpdating labels               " << std::flush;
        ulong cumul_labels = 0;
        for (auto &layer : layers) {
            auto size = layer.edge_img.size();
            for (uint i = 0; i < size.height; ++i) {
                for (uint j = 0; j < size.width; ++j) {
                    int32_t &label = layer.labels.at<int32_t>(i, j);
                    if (label != 0) {
                        label += cumul_labels;
                    }
                }
            }
            cumul_labels += layer.num_labels - 1;
        }

        // Merge labels of neighbouring layers
        bool **equivalence_table = new bool*[cumul_labels + 1];
        int32_t label_map[cumul_labels + 1] = {0};
        for (uint i = 0; i < cumul_labels + 1; ++i) {
            equivalence_table[i] = new bool[cumul_labels + 1];
            for (uint j = 0; j < cumul_labels + 1; ++j) {
                equivalence_table[i][i] = i == j;
            }
        }
        for (uint i = 0; i < num_layers; ++i) {
            auto &layer = layers[i];
            auto size = layer.edge_img.size();
            for (uint j = 0; j < size.height; ++j) {
                for (uint k = 0; k < size.width; ++k) {
                    int32_t label = layer.labels.at<int32_t>(j, k);
                    if (label == 0) {
                        continue;
                    }
                    if (i > 0) {
                        auto &prev_layer = layers[i - 1];
                        auto labels_in_range = get_labels_in_range(prev_layer.labels, j, k);
                        for (auto &prev_label : labels_in_range) {
                            equivalence_table[label][prev_label] |= label != 0 && prev_label != 0;
                            equivalence_table[prev_label][label] |= label != 0 && prev_label != 0;
                        }
                    }
                    if (i < num_layers - 1) {
                        auto &next_layer = layers[i + 1];
                        auto labels_in_range = get_labels_in_range(next_layer.labels, j, k);
                        for (auto &next_label : labels_in_range) {
                            equivalence_table[label][next_label] |= label != 0 && next_label != 0;
                            equivalence_table[next_label][label] |= label != 0 && next_label != 0;
                        }
                    }
                }
            }
        }
        for (uint i = 1; i < cumul_labels + 1; ++i) {
            for (uint j = 1; j < cumul_labels + 1; ++j) {
                if (equivalence_table[i][j]) {
                    label_map[i] = j;
                    break;
                }
            }
        }
        for (auto &layer : layers) {
            auto size = layer.edge_img.size();
            for (uint i = 0; i < size.height; ++i) {
                for (uint j = 0; j < size.width; ++j) {
                    int32_t &label = layer.labels.at<int32_t>(i, j);
                    label = label_map[label];
                }
            }
        }

        std::cout << "\rHysteresis thresholding       " << std::flush;
        bool acc_segments[cumul_labels + 1] = {false};
        // Check components if at least one pixel is above the high threshold
        for (auto &layer : layers) {
            auto size = layer.edge_img.size();
            for (uint i = 0; i < size.height; ++i) {
                for (uint j = 0; j < size.width; ++j) {
                    int32_t label = layer.labels.at<int32_t>(i, j);
                    uint8_t mag = layer.edge_img.at<uint8_t>(i, j);
                    if (mag >= layer.threshold) {
                        acc_segments[label] = true;
                    }
                }
            }
        }

        // Delete not accepted components
        for (auto &layer : layers) {
            auto size = layer.low_thr.size();
            for (uint i = 0; i < size.height; ++i) {
                for (uint j = 0; j < size.width; ++j) {
                    int32_t label = layer.labels.at<int32_t>(i, j);
                    if (!acc_segments[label]) {
                        layer.low_thr.at<uint8_t>(i, j) = 0;
                    }
                }
            }
        }
        for (uint i = 0; i < cumul_labels + 1; ++i) {
            delete[] equivalence_table[i];
        }
        delete[] equivalence_table;
        // ----- HYSTERESIS THRESHOLDING -----

        // ----- 24-CONNECTIVITY -----
        std::cout << "\rExecuting 24-connectivity     " << std::flush;
        std::vector<cv::Mat> edges_24;
        edges_24.resize(num_layers);
        for (uint i = 0; i < num_layers - 1; ++i) {
            auto &layer = layers[i];
            edges_24[i] = layer.low_thr.clone();
            auto size = layer.edge_img.size();
            if (!disable_24_connectivity) {
                for (uint j = 0; j < size.height; ++j) {
                    for (uint k = 0; k < size.width; ++k) {
                        auto &next_layer = layers[i + 1];
                        do_24_connectivity(layer.low_thr, next_layer.low_thr, edges_24[i], j, k);
                    }
                }
            }
        }
        edges_24[num_layers - 1] = layers[num_layers - 1].low_thr.clone();
        // ----- 24-CONNECTIVITY -----

        std::cout << "\rWriting output images         " << std::flush;
        for (uint i = 0; i < num_layers; ++i) {
            cv::imwrite(out_dir + "edges_" + layers[i].img_name, edges_24[i]);
        }
        std::cout << "\rDone!                         " << std::endl;
}