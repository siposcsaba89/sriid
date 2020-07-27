#include <iostream>
#include <opencv2/opencv.hpp>
#include <regex>

static void onIDSslider(int val, void* user_data)
{};

void calculateShadowEdgesRoi(const cv::Mat& img,
    const cv::Rect& roi_to_clear,
    cv::Mat& edge_image,
    int canny_thrs = 80);
void calculateShadowEdges(cv::Mat& ii_img,
    const cv::Mat& img,
    cv::Mat& edge_image,
    int canny_thrs_ii_img,
    int canny_thrs_img);

std::vector<std::string> split(const std::string& input, const std::string& regex);
//float c1 = -0.985644, c2 = 0.168834; snowy img


int main(int argc, const char* argv[])
{

    cv::Mat filter_gx(1, 3, CV_32F);
    filter_gx.at<float>(0) = -0.5f;
    filter_gx.at<float>(1) = 0.0f;
    filter_gx.at<float>(2) = 0.5f;
    cv::Mat filter_gy = filter_gx.t();

    const std::string keys =
        "{help h usage ? |      | print this message                                                        }"
        "{@image1        |      | image1 for compare                                                        }"
        "{roi            |      | roi to delete canny edges                                                 }"
        "{thrs1          |200   | Canny threshold for grayscale image                                       }"
        "{thrs2          |200   | Canny threshold for invariant image                                       }"
        "{c1             |0.5463| Illuminant invariant direction c1                                         }"
        "{c2             |-0.837| Illuminant invariant direction c2                                         }"
        "{calibrate      |false | Manually calibrate [c1, c2] direction                                     }"
        "{mode           |1     | 1 = Use roi to clear non shadow edges, 0 = try to figure out shadow edges }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string img1 = parser.get<std::string>(0);
    float img_canny_thrs = parser.get<float>("thrs1");
    float ii_img_canny_thrs = parser.get<float>("thrs2");
    float c1 = parser.get<float>("c1");
    float c2 = parser.get<float>("c2");
    bool calibrate = parser.get<bool>("calibrate");
    int mode = parser.get<int>("mode");

    cv::Mat img = cv::imread(img1);
    //cv::Mat img2;
    //cv::resize(img, img2, cv::Size(img.cols / 2, img.rows / 2));
    //img = img2;
    cv::imshow("Original", img);

    cv::Mat log_r_g(img.rows, img.cols, CV_32F), log_b_g(img.rows, img.cols, CV_32F);
    for (int j = 0; j < img.rows; ++j)
    {
        for (int i = 0; i < img.cols; ++i)
        {
            cv::Vec3b color = img.at<cv::Vec3b>(j, i);
            log_r_g.at<float>(j, i) = 0;
            log_b_g.at<float>(j, i) = 0;
            if (color[1] != 0)
            {
                float scale = color[1];
                if (color[2] != 0)
                    log_r_g.at<float>(j, i) = std::log(color[2] / scale );
                if (color[0] != 0)
                    log_b_g.at<float>(j, i) = std::log(color[0] / scale);
            }
        }
    }

    auto combineLogImages = [](const cv::Mat& log_r_g, const cv::Mat& log_b_g, cv::Mat& img_res, const float c1, const float c2)
    {
        for (int j = 0; j < img_res.rows; ++j)
        {
            for (int i = 0; i < img_res.cols; ++i)
            {
                img_res.at<float>(j, i) = std::abs(c1 * log_r_g.at<float>(j, i) - c2 * log_b_g.at<float>(j, i));
            }
        }
    };

    cv::Mat illuminant_invariant_img(log_r_g.size(), CV_32F);
    combineLogImages(log_r_g, log_b_g, illuminant_invariant_img, c1, c2);
    cv::Mat illuminant_invariant_img_exp;

    if (calibrate)
    {
        //create trackbar window
        const std::string winname = "Invariant direction slider";
        const std::string slider_name = "ids";
        cv::namedWindow(winname, cv::WINDOW_AUTOSIZE); // Create Window
        int val = 0, val_curr = -1;

        int max_val = 1000;
        cv::createTrackbar(slider_name, winname, &val, max_val, onIDSslider);

        bool calib_done = false;
        while (!calib_done)
        {
            if (val != val_curr)
            {
                val_curr = val;
                c1 = std::cos(val / (float)max_val * 2 * 3.141592f);
                c2 = std::sin(val / (float)max_val * 2 * 3.141592f);
                std::cout << c1 << "   " << c2 << std::endl;
                combineLogImages(log_r_g, log_b_g, illuminant_invariant_img, c1, c2);
                cv::exp(illuminant_invariant_img, illuminant_invariant_img_exp);
                cv::imshow(winname, illuminant_invariant_img_exp / 2);
            }
            int key = cv::waitKey(50);
            if (key == 27)
            {
                calib_done = true;
            }
        }
    }
    cv::Mat edges_to_remove;
    if (mode == 0)
    {
        calculateShadowEdges(illuminant_invariant_img, img, edges_to_remove, int(ii_img_canny_thrs), int(img_canny_thrs));
    }
    else if (mode == 1)
    {
        cv::Rect roi(0, 0, img.cols, img.rows);
        if (parser.has("roi"))
        {
            std::string roi_str = parser.get<std::string>("roi");
            auto rois = split(roi_str, "x");
            assert(rois.size() == 4);
            roi.x = int(std::stof(rois[0]) * roi.width);
            roi.y = int(std::stof(rois[1]) * roi.height);
            roi.width = int(std::stof(rois[2]) * roi.width);
            roi.height = int(std::stof(rois[3]) * roi.height);
        }
        calculateShadowEdgesRoi(img, roi, edges_to_remove, int(img_canny_thrs));
    }
    cv::imshow("Edges to remove", edges_to_remove);
    cv::waitKey(1);
    //split color channels and calculate log (color/gradient magnitude) images
    std::vector<cv::Mat> bgr_channels, bgr_channels_log, log_grad, log_dx, log_dy;
    cv::split(img, bgr_channels);
    bgr_channels_log.resize(bgr_channels.size());
    log_grad.resize(bgr_channels.size());
    log_dx.resize(bgr_channels.size());
    log_dy.resize(bgr_channels.size());
    for (size_t i = 0; i < bgr_channels.size(); ++i)
    {
        cv::Mat float_img;
        bgr_channels[i].convertTo(float_img, CV_32F);
        bgr_channels_log[i].create(float_img.size(), CV_32FC1);
        for (int l = 0; l < float_img.rows; ++l)
        {
            for (int k = 0; k < float_img.cols; ++k)
            {
                float int_val = float_img.at<float>(l, k);
                if (int_val < std::numeric_limits<float>::epsilon())
                    int_val = 0.0001f;
                bgr_channels_log[i].at<float>(l, k) = std::log(int_val);
            }
        }
        cv::filter2D(bgr_channels_log[i], log_dx[i], CV_32F, filter_gx);
        cv::filter2D(bgr_channels_log[i], log_dy[i], CV_32F, filter_gy);
        cv::magnitude(log_dx[i], log_dy[i], log_grad[i]);
    }

    //calculate gradient of the illuminant invariant image
    cv::Mat grad_iii;
    {
        cv::Mat dx, dy;
        cv::filter2D(illuminant_invariant_img, dx, CV_32F, filter_gx);
        cv::filter2D(illuminant_invariant_img, dy, CV_32F, filter_gy);
        cv::magnitude(dx, dy, grad_iii);
    }
    //solving for color image
    std::vector<cv::Mat> re_log_channels(bgr_channels.size());
    std::vector<cv::Mat> re_channels(bgr_channels.size());
    //iterate over image channel and remove color edges belonging to shadows, restore original image channels
    for (size_t k = 0; k < bgr_channels.size(); ++k)
    {
        cv::Mat S_x(log_dx[k].size(), CV_32FC1);
        cv::Mat S_y(log_dx[k].size(), CV_32FC1);

        for (int j = 0; j < log_dx[k].rows; ++j)
        {
            for (int i = 0; i < log_dx[k].cols; ++i)
            {
                auto& s_x = S_x.at<float>(j, i);
                auto& s_y = S_y.at<float>(j, i);
                float edge_to_remove = edges_to_remove.at<uchar>(j, i);

                if (edge_to_remove > 100) //shadow edge
                {
                    s_x = 0;
                    s_y = 0;
                }
                else
                {
                    //std::cout << ii_grad << std::endl;
                    s_x = log_dx[k].at<float>(j, i);
                    s_y = log_dy[k].at<float>(j, i);
                }
            }
        }

        //calculate laplacian of the edge map, shadows already removed
        cv::Mat S_x_dx, S_y_dy;
        cv::filter2D(S_x, S_x_dx, CV_32F, filter_gx);
        cv::filter2D(S_y, S_y_dy, CV_32F, filter_gy);
        cv::Mat S_lap = S_x_dx + S_y_dy;

        cv::Mat& u = re_log_channels[k];

        //set boundary to some common value
        u.create(log_dx[k].size(), CV_32FC1);
        u.setTo(std::log(111));

        //lots of iteration required to restore the original image from the laplacian
        for (int it = 0; it < 20000; ++it)
        {
            for (int j = 2; j < log_dx[k].rows - 2; ++j)
            {
                for (int i = 2; i < log_dx[k].cols - 2; ++i)
                {
                    float lap_ = 
                        0.25f * (
                            u.at<float>(j, i - 2) + u.at<float>(j, i + 2) +
                        u.at<float>(j - 2, i) + u.at<float>(j + 2, i) -
                        4.0f * u.at<float>(j, i));
        
        
                    u.at<float>(j, i) +=
                        lap_ - S_lap.at<float>(j, i);
                }
            }
        }

        re_channels[k].create(log_dx[k].size(), CV_8UC1);
        for (int j = 0; j < log_dx[k].rows; ++j)
        {
            for (int i = 0; i < log_dx[k].cols; ++i)
            {
                re_channels[k].at<uint8_t>(j, i) = uint8_t(std::min(std::exp(u.at<float>(j, i)) * 1.f, 255.0f));
            }
        }
    }
    cv::Mat reconstructed_img;
    cv::merge(re_channels, reconstructed_img);

    cv::imshow("reconstructed_img", reconstructed_img);

    cv::waitKey(0);
    return 0;
}


std::vector<std::string> split(const std::string& input, const std::string& regex)
{
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator
        first{ input.begin(), input.end(), re, -1 },
        last;
    return { first, last };
}

void calculateShadowEdgesRoi(const cv::Mat& img,
    const cv::Rect& roi_to_clear,
    cv::Mat& edge_image,
    int canny_thrs)
{
    cv::Mat img_gray;
    if (img.channels() == 1)
    {
        img_gray = img;
    }
    else
    {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    }
    cv::Canny(img_gray, edge_image, canny_thrs, canny_thrs);
    edge_image(roi_to_clear).setTo(0);
    cv::dilate(edge_image, edge_image, cv::Mat());
    cv::dilate(edge_image, edge_image, cv::Mat());
}

void calculateShadowEdges(cv::Mat& ii_img,
    const cv::Mat& img,
    cv::Mat& edge_image,
    int canny_thrs_ii_img,
    int canny_thrs_img)
{
    cv::Mat img_gray;
    if (img.channels() == 1)
        img_gray = img;
    else
    {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    }
    edge_image.create(img_gray.size(), CV_8UC1);
    edge_image.setTo(0);
    cv::Mat blurred_img;

    cv::GaussianBlur(ii_img, blurred_img, cv::Size(5, 5), 2.0);
    cv::Mat u8_iimg(img.size(), CV_8UC1), tmp;
    cv::exp(blurred_img, blurred_img);
    tmp = (blurred_img * 255.0f);
    for (int j = 0; j < img.rows; ++j)
    {
        for (int i = 0; i < img.cols; ++i)
        {
            u8_iimg.at<uint8_t>(j, i) = uint8_t(std::abs(tmp.at<float>(j, i)));
        }
    }

    cv::Mat ii_edges;
    cv::Canny(u8_iimg, ii_edges, canny_thrs_ii_img, canny_thrs_ii_img);

    cv::Mat img_edges;
    cv::Canny(img_gray, img_edges, canny_thrs_img, canny_thrs_img);

    for (int j = 0; j < img.rows; ++j)
    {
        for (int i = 0; i < img.cols; ++i)
        {
            if (img_edges.at<uint8_t>(j, i) > 100 && ii_edges.at<uint8_t>(j, i) < 100)
            {
                edge_image.at<uint8_t>(j, i) = 255;
            }
        }
    }
    cv::dilate(edge_image, edge_image, cv::Mat());
    cv::dilate(edge_image, edge_image, cv::Mat());
}

