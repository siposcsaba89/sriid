#include <iostream>
#include <opencv2/opencv.hpp>


int slider_val = 0;

static void onIDSslider(int val, void* user_data)
{
    slider_val = val;
};



void calculateShadowEdges(cv::Mat& ii_img, const cv::Mat& img, cv::Mat& edge_image)
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
            u8_iimg.at<uint8_t>(j, i) = std::abs(tmp.at<float>(j, i));
        }
    }

    cv::Mat ii_edges;
    cv::Canny(u8_iimg, ii_edges, 1, 150);

    cv::Mat img_edges;

    cv::Canny(img_gray, img_edges, 50, 100);

    

    //img_edges(cv::Rect(0, 0, img.cols, img.rows / 2)).setTo(0);
    //tmp.convertTo(u8_iimg, CV_8U);

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


    cv::dilate(img_edges, img_edges, cv::Mat());
    cv::dilate(img_edges, img_edges, cv::Mat());

    cv::imshow("ii_edges", ii_edges);
    cv::imshow("image_edges", img_edges);
    //cv::imshow("ii_imgsdasdsadasd", u8_iimg);
    cv::imshow("img_gray", img_gray);
    cv::imshow("shadow_edge_image", edge_image);
    cv::waitKey(0);
}


cv::Mat RL_deconvolution(cv::Mat observed, cv::Mat psf, int iterations)
{
    using namespace cv;
    Scalar grey;

    // Uniform grey starting estimation
    grey = Scalar(128.f);
    Mat latent_est = Mat(observed.size(), CV_32F);
    latent_est.setTo(127.0f);
    cv::randu(latent_est, 0.0f, log(255.0f));


    // Flip the point spread function (NOT the inverse)
    Mat psf_hat = Mat(psf.size(), CV_32FC1);
    int psf_row_max = psf.rows - 1;
    int psf_col_max = psf.cols - 1;
    for (int row = 0; row <= psf_row_max; row++) {
        for (int col = 0; col <= psf_col_max; col++) {
            psf_hat.at<float>(psf_row_max - row, psf_col_max - col) =
                psf.at<float>(row, col);
        }
    }
    std::cout << psf_hat << std::endl;

    Mat est_conv;
    Mat relative_blur;
    Mat error_est;

    // Iterate
    for (int i = 0; i < iterations; i++) {

        filter2D(latent_est, est_conv, -1, psf);

        // Element-wise division
        relative_blur = observed.mul(1.0 / est_conv);

        filter2D(relative_blur, error_est, -1, psf_hat);

        // Element-wise multiplication
        latent_est = latent_est.mul(error_est);
    }

    return latent_est;
}


int main(int argc, const char* argv[])
{

    cv::Mat filter_gx(1, 3, CV_32F);
    filter_gx.at<float>(0) = -0.5f;
    filter_gx.at<float>(1) = 0.0f;
    filter_gx.at<float>(2) = 0.5f;
    cv::Mat filter_gy = filter_gx.t();
    std::cout << filter_gx << std::endl;
    std::cout << filter_gy << std::endl;


    const std::string keys =
        "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1 for compare   }"
        //"{ts timestamp   |      | use time stamp       }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string img1 = parser.get<std::string>(0);


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
                img_res.at<float>(j, i) = c1 * log_r_g.at<float>(j, i) - c2 * log_b_g.at<float>(j, i);
            }
        }
    };

    //create trackbar window
    const std::string winname = "Invariant direction slider";
    const std::string slider_name = "ids";
    cv::namedWindow(winname, cv::WINDOW_AUTOSIZE); // Create Window
    int val = 0, val_curr = -1;


    float c1 = 0.627692, c2 = 0.778462;
       
    cv::Mat illuminant_invariant_img(log_r_g.size(), CV_32F);
    combineLogImages(log_r_g, log_b_g, illuminant_invariant_img, c1, c2);
    cv::Mat illuminant_invariant_img_exp;
    //
    //cv::Mat edges, illuminant_invariant_img_u8;
    //illuminant_invariant_img.convertTo(illuminant_invariant_img_u8, CV_8UC1, 256);
    //cv::GaussianBlur(illuminant_invariant_img_u8, illuminant_invariant_img_u8, cv::Size(3, 3), 6.0);
    //cv::GaussianBlur(illuminant_invariant_img_u8, illuminant_invariant_img_u8, cv::Size(3, 3), 6.0);
    //cv::GaussianBlur(illuminant_invariant_img, illuminant_invariant_img, cv::Size(3, 3), 2.0);
    //cv::Canny(illuminant_invariant_img_u8, edges, 1, 16, 3);
    //cv::dilate(edges, edges, cv::Mat());
    //cv::dilate(edges, edges, cv::Mat());
    //cv::dilate(edges, edges, cv::Mat());
    //cv::imshow("edges", edges);
    //cv::waitKey(0);
    //cv::exp(illuminant_invariant_img, illuminant_invariant_img_exp);
    //cv::imshow(winname, illuminant_invariant_img_exp / 4);
#if 0
    int max_val = 1000;
    cv::createTrackbar(slider_name, winname, &val, max_val, onIDSslider);
    
    cv::imshow("log_r_g", log_r_g);
    cv::imshow("log_b_g", log_b_g);

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
            cv::imshow(winname, illuminant_invariant_img_exp / 4);
        }
        int key = cv::waitKey(50);
        if (key == 27)
        {
            calib_done = true;
        }
    
    }
#endif

    cv::Mat edges_to_remove;
    calculateShadowEdges(illuminant_invariant_img, img, edges_to_remove);
#if 1
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
        //for (int l = 0; l < float_img.rows; ++l)
        //{
        //    for (int k = 0; k < float_img.cols; ++k)
        //    {
        //        float mag = log_grad[i].at<float>(l, k);
        //        if (mag < 0.20f)
        //        {
        //            log_dx[i].at<float>(l, k) = 0;
        //            log_dy[i].at<float>(l, k) = 0;
        //        }
        //    }
        //}

    }

    //calculate gradient of the illuminant invariant image
    cv::Mat grad_iii;
    {
        cv::Mat dx, dy;
        cv::filter2D(illuminant_invariant_img, dx, CV_32F, filter_gx);
        cv::filter2D(illuminant_invariant_img, dy, CV_32F, filter_gy);
        cv::magnitude(dx, dy, grad_iii);
    }
    cv::imshow("grad_iii", grad_iii);
    float thrs1 = 0.1f; //mondjuk
    float thrs2 = 0.2f;
    //solving for color image
    std::vector<cv::Mat> re_log_channels(bgr_channels.size());
    std::vector<cv::Mat> re_channels(bgr_channels.size());
#if 1
    for (size_t k = 0; k < bgr_channels.size(); ++k)
    {
        cv::Mat S_x(log_dx[k].size(), CV_32FC1);
        cv::Mat S_y(log_dx[k].size(), CV_32FC1);
        cv::Mat shadow_edges(log_dx[k].size(), CV_32FC1);
        shadow_edges.setTo(0);
        cv::imshow("log_grad[k]", log_grad[k]);

        for (int j = 0; j < log_dx[k].rows; ++j)
        {
            for (int i = 0; i < log_dx[k].cols; ++i)
            {
                auto& s_x = S_x.at<float>(j, i);
                auto& s_y = S_y.at<float>(j, i);
                float log_channel_grad_mag = log_grad[k].at<float>(j, i);
                //float ii_grad = grad_iii.at<float>(j, i);
                float ii_grad = edges_to_remove.at<uchar>(j, i);

                if (/*log_channel_grad_mag > thrs1 && */ ii_grad > 100) //shadow edge
                //if (ii_grad < thrs2) //shadow edge
                //if (log_channel_grad_mag > thrs1) //shadow edge
                {
                    s_x = 0;
                    s_y = 0;
                    shadow_edges.at<float>(j, i) = 1.0f;
                }
                else
                {
                    //std::cout << ii_grad << std::endl;
                    s_x = log_dx[k].at<float>(j, i);
                    s_y = log_dy[k].at<float>(j, i);
                }
            }
        }

        cv::imshow("shadow edges", shadow_edges);
        cv::waitKey(1);
        cv::Mat S_x_dx, S_y_dy;
        cv::filter2D(S_x, S_x_dx, CV_32F, filter_gx);
        cv::filter2D(S_y, S_y_dy, CV_32F, filter_gy);

        cv::Mat S_lap = S_x_dx + S_y_dy;

        cv::imshow("S_lap", S_lap);
        //cv::waitKey(0);
        cv::Mat& u = re_log_channels[k];

        u.create(log_dx[k].size(), CV_32FC1);
        u.setTo(std::log(111));

        //for (int i = 0; i < img.cols; ++i)
        //{
        //    u.at<float>(0, i) = bgr_channels_log[k].at<float>(0, i);
        //    u.at<float>(1, i) = bgr_channels_log[k].at<float>(1, i);
        //}
        //
        //for (int i = 0; i < img.cols; ++i)
        //{
        //    u.at<float>(img.rows - 1, i) = bgr_channels_log[k].at<float>(img.rows - 1, i);
        //    u.at<float>(img.rows - 2, i) = bgr_channels_log[k].at<float>(img.rows - 2, i);
        //}
        //
        //for (int i = 0; i < img.rows; ++i)
        //{
        //    u.at<float>(i, 0) = bgr_channels_log[k].at<float>(i, 0);
        //    u.at<float>(i, 1) = bgr_channels_log[k].at<float>(i, 1);
        //}
        //
        //for (int i = 0; i < img.rows; ++i)
        //{
        //    u.at<float>(i, img.cols - 1) = bgr_channels_log[k].at<float>(i, img.cols - 1);
        //    u.at<float>(i, img.cols - 2) = bgr_channels_log[k].at<float>(i, img.cols - 2);
        //}


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
        cv::imshow("re_channels[k]", re_channels[k]);
        cv::waitKey(1);
    }
    cv::Mat reconstructed_img;
    cv::merge(re_channels, reconstructed_img);

    std::vector<cv::Mat> tmp(bgr_channels_log.size());
    cv::exp(bgr_channels_log[0], tmp[0]);
    cv::exp(bgr_channels_log[1], tmp[1]);
    cv::exp(bgr_channels_log[2], tmp[2]);
    cv::Mat t_c;
    cv::merge(tmp, t_c);
    cv::imshow("exp chanell", t_c / 255.0f);
#endif
#if 0
    {
        int channel = 0;
        //test GS iteration


        cv::Mat grad_x, grad_y;
        cv::Mat grad_xx, grad_yy;


        cv::filter2D(bgr_channels_log[channel], grad_x, CV_32F, filter_gx);
        cv::filter2D(grad_x, grad_xx, CV_32F, filter_gx);
        cv::filter2D(bgr_channels_log[channel], grad_y, CV_32F,filter_gy);
        cv::filter2D(grad_y, grad_yy, CV_32F, filter_gy);
        cv::Mat lap2 = (grad_xx + grad_yy);
        cv::imshow("lap2", lap2);
        
        cv::Mat img(lap2.size(), CV_32F);
        img.setTo(0);

        for (int i = 0; i < img.cols; ++i)
        {
            img.at<float>(0, i) = bgr_channels_log[channel].at<float>(0, i);
            img.at<float>(1, i) = bgr_channels_log[channel].at<float>(1, i);
        }
        
        for (int i = 0; i < img.cols; ++i)
        {
            img.at<float>(img.rows -1, i) = bgr_channels_log[channel].at<float>(img.rows - 1, i);
            img.at<float>(img.rows -2, i) = bgr_channels_log[channel].at<float>(img.rows - 2, i);
        }
        
        for (int i = 0; i < img.rows; ++i)
        {
            img.at<float>(i, 0) = bgr_channels_log[channel].at<float>(i, 0);
            img.at<float>(i, 1) = bgr_channels_log[channel].at<float>(i, 1);
        }
        
        for (int i = 0; i < img.rows; ++i)
        {
            img.at<float>(i, img.cols - 1) = bgr_channels_log[channel].at<float>(i, img.cols - 1);
            img.at<float>(i, img.cols - 2) = bgr_channels_log[channel].at<float>(i, img.cols - 2);
        }


        for (int it = 0; it < 30000; ++it)
        {
            for (int j = 2; j < img.rows - 2; ++j)
            {
                for (int i = 2; i < img.cols - 2; ++i)
                {
                    float lap_ = 
                        img.at<float>(j, i - 2) + img.at<float>(j, i + 2) +
                        img.at<float>(j - 2, i) + img.at<float>(j + 2, i) -
                        //img.at<float>(j - 1, i - 1) + img.at<float>(j - 1, i + 1) +
                        //img.at<float>(j + 1, i - 1) + img.at<float>(j + 1, i + 1) -
                        4.0f * img.at<float>(j, i);

                    img.at<float>(j, i) +=
                        0.25f * lap_ - lap2.at<float>(j, i);
                }
            }
        }

        cv::Mat lap_ker(3, 3, CV_32F);
        lap_ker.at<float>(0, 0) = 0;
        lap_ker.at<float>(0, 1) = 1;
        lap_ker.at<float>(0, 2) = 0;
        lap_ker.at<float>(1, 0) = 1;
        lap_ker.at<float>(1, 1) = -4;
        lap_ker.at<float>(1, 2) = 1;
        lap_ker.at<float>(2, 0) = 0;
        lap_ker.at<float>(2, 1) = 1;
        lap_ker.at<float>(2, 2) = 0;
        //cv::Mat rl_res = RL_deconvolution(lap, lap_ker, 50);
        cv::Mat exp_img;
        cv::exp(img, exp_img);
        cv::imshow("exp_img", exp_img  / 255.0f);
        //cv::Mat rl_res_exp;
        //cv::exp(rl_res, rl_res_exp);
        //cv::imshow("rl_res", rl_res_exp / 255.0f);
        cv::Mat orig_exp;
        cv::exp(bgr_channels_log[channel], orig_exp);
        cv::imshow("orig_exp", orig_exp / 255.0f);

    }

#endif
    cv::imshow("log_r_g", log_r_g);
    cv::imshow("log_b_g", log_b_g);

    cv::imshow("reconstructed_img", reconstructed_img);

    cv::waitKey(0);
#endif
    return 0;
}

