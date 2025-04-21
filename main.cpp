#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <filesystem>
#include <opencv2/core/utils/logger.hpp>

struct FeatureInfo
{
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

std::vector<cv::DMatch> MatchDescriptors(FeatureInfo const &first, FeatureInfo const &second){
   
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);
	std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> goodMatches;
    matcher->match(first.descriptors, second.descriptors, matches);
    double minDist = 100;
	for (auto match : matches)
	{
		if (match.distance < minDist)
			minDist = match.distance;
	}
    for (auto match : matches){
        if (match.distance < 4*minDist){
            goodMatches.push_back(match);
        }
    }
    std::cout << "Количество найденных точек для 1-го изображения: " << first.keypoints.size();
    std::cout << ", для второго: " << second.keypoints.size() << std::endl;
    std::cout << "Количество хороших точек: " << goodMatches.size() << std::endl;
    
    return goodMatches;

};

void drawKeypointsAndLines (cv::Mat &first_img, cv::Mat &second_img, FeatureInfo const &first, 
    FeatureInfo const &second, std::vector<cv::DMatch> &goodMatches, std::string window_name){

    cv::Mat img;
    cv::Mat img_2;
    cv::Mat result;

    cv::drawKeypoints(first_img, first.keypoints, img);
    cv::drawKeypoints(second_img, second.keypoints, img_2);
    cv::drawMatches(img, first.keypoints, img_2, second.keypoints, goodMatches, result);

    cv::namedWindow(window_name, cv::WINDOW_KEEPRATIO);
    cv::imshow(window_name, result);
    cv::resizeWindow(window_name, 600, 600);
    cv::waitKey();
    
};

void calculateAndPrintStatistics(std::vector<double> &distances) {

    std::sort(distances.begin(), distances.end());
    double sum = 0;
    for (auto distance : distances) {
        sum += distance;
    }
    double mean = sum / distances.size();

    double median;
    if (distances.size() % 2 == 0) {
        median = (distances[int(distances.size() / 2)] + distances[int(distances.size() / 2)-1]) / 2;
    } else {
        median = distances[int(distances.size() / 2)];
    }

    std::cout << "Среднее значение: " << mean << ", медианное значение: " << median << std::endl;
    std::cout << std::endl;

}

void calculateEpipolarDistances (std::vector<cv::KeyPoint> &keypoints_first, 
    std::vector<cv::KeyPoint> &keypoints_second, std::vector<cv::DMatch> &goodMatches){
    
    std::vector<cv::Point2f> keypoints_first_point2f;
    std::vector<cv::Point2f> keypoints_second_point2f;

    for (int i = 0; i < goodMatches.size(); ++i){

        int first_img_keypoint_index = goodMatches[i].queryIdx;
        int second_img_keypoint_index = goodMatches[i].trainIdx;

        keypoints_first_point2f.push_back(keypoints_first[first_img_keypoint_index].pt);
        keypoints_second_point2f.push_back(keypoints_second[second_img_keypoint_index].pt);
    }

    std::vector<cv::Point3f> lines_first, lines_second;
    std::vector<double> distances;

    cv::Mat fundamental_matrix = cv::findFundamentalMat(keypoints_first_point2f, keypoints_second_point2f,cv::USAC_MAGSAC);
    if (fundamental_matrix.rows == 3 && fundamental_matrix.cols == 3){

        cv::computeCorrespondEpilines(keypoints_first_point2f, 1, fundamental_matrix, lines_first);
        cv::computeCorrespondEpilines(keypoints_second_point2f, 2, fundamental_matrix.t(), lines_second);

        for (int i = 0; i<keypoints_first_point2f.size(); ++i){
            double distance_first = std::abs(
                lines_first[i].x * keypoints_first_point2f[i].x +
                lines_first[i].y * keypoints_first_point2f[i].y +
                lines_first[i].z) /
                std::sqrt(std::pow(lines_first[i].x, 2) + 
                          std::pow(lines_first[i].y, 2));
            
            double distance_second = std::abs(
                lines_second[i].x * keypoints_second_point2f[i].x +
                lines_second[i].y * keypoints_second_point2f[i].y +
                lines_second[i].z) /
                std::sqrt(std::pow(lines_second[i].x, 2) + 
                          std::pow(lines_second[i].y, 2));
            
            distances.push_back(distance_first + distance_second);
        }
        calculateAndPrintStatistics(distances);
    } else {
        std::cout << "Error, uncorrect fundamental matrix" << std::endl;
    }

}

void detectAndComputeAllDescriptors (cv::Mat &first_img, cv::Mat &second_img){

    FeatureInfo brisk_frameInfo_first, brisk_frameInfo_second;
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    brisk->detect(first_img, brisk_frameInfo_first.keypoints);
    brisk->detect(second_img, brisk_frameInfo_second.keypoints);
    clock_t start_brisk = clock();
    brisk->compute(first_img, brisk_frameInfo_first.keypoints, brisk_frameInfo_first.descriptors);
    brisk->compute(second_img, brisk_frameInfo_second.keypoints, brisk_frameInfo_second.descriptors);
    clock_t end_brisk = clock();
    std::cout << "BRISK working time: " <<  (end_brisk-start_brisk) << std::endl;
    std::vector<cv::DMatch> brisk_matches = MatchDescriptors(brisk_frameInfo_first, brisk_frameInfo_second);
    drawKeypointsAndLines(first_img, second_img, brisk_frameInfo_first, brisk_frameInfo_second, brisk_matches, "BRISK");
    calculateEpipolarDistances(brisk_frameInfo_first.keypoints, brisk_frameInfo_second.keypoints, brisk_matches);


    FeatureInfo sift_frameInfo_first, sift_frameInfo_second;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detect(first_img, sift_frameInfo_first.keypoints);
    sift->detect(second_img, sift_frameInfo_second.keypoints);
    clock_t start_sift = clock();
    sift->compute(first_img, sift_frameInfo_first.keypoints, sift_frameInfo_first.descriptors);
    sift->compute(second_img, sift_frameInfo_second.keypoints, sift_frameInfo_second.descriptors);
    clock_t end_sift = clock();
    std::cout << "SIFT working time: " <<  (end_sift-start_sift) << std::endl;
    std::vector<cv::DMatch> sift_matches = MatchDescriptors(sift_frameInfo_first, sift_frameInfo_second);
    drawKeypointsAndLines(first_img, second_img, sift_frameInfo_first, sift_frameInfo_second, sift_matches, "SIFT");
    calculateEpipolarDistances(sift_frameInfo_first.keypoints, sift_frameInfo_second.keypoints, sift_matches);
    

    FeatureInfo orb_frameInfo_first, orb_frameInfo_second;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detect(first_img, orb_frameInfo_first.keypoints);
    orb->detect(second_img, orb_frameInfo_second.keypoints);
    clock_t start_orb = clock();
    orb->compute(first_img, orb_frameInfo_first.keypoints, orb_frameInfo_first.descriptors);
    orb->compute(second_img, orb_frameInfo_second.keypoints, orb_frameInfo_second.descriptors);
    clock_t end_orb = clock();
    std::cout << "ORB working time: " <<  (end_orb-start_orb) << std::endl;
    std::vector<cv::DMatch> orb_matches = MatchDescriptors(orb_frameInfo_first, orb_frameInfo_second);
    drawKeypointsAndLines(first_img, second_img, orb_frameInfo_first, orb_frameInfo_second, orb_matches, "ORB");
    calculateEpipolarDistances(orb_frameInfo_first.keypoints, orb_frameInfo_second.keypoints, orb_matches);

};

int main(){

    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    for (int i = 1; i < 6 ; ++i){
        std::stringstream first_file;
        std:: stringstream second_file;

        first_file << "./1/" << i << ".png";
        second_file << "./2/" << i << ".png";

        cv::Mat first_img = cv::imread(first_file.str());
        cv::Mat second_img = cv::imread(second_file.str());

        cv::cvtColor(first_img, first_img, cv::COLOR_BGR2GRAY);
        cv::cvtColor(second_img, second_img, cv::COLOR_BGR2GRAY);

        detectAndComputeAllDescriptors(first_img, second_img);
    }

}