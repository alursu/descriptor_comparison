#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
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
        if (match.distance < 2*minDist){
            goodMatches.push_back(match);
        }
    }
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
    cv::imshow(window_name, result);
    cv::waitKey();
    
};

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


    FeatureInfo surf_frameInfo_first, surf_frameInfo_second;
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    surf->detect(first_img, surf_frameInfo_first.keypoints);
    surf->detect(second_img, surf_frameInfo_second.keypoints);
    clock_t start_surf = clock();
    surf->compute(first_img, surf_frameInfo_first.keypoints, surf_frameInfo_first.descriptors);
    surf->compute(second_img, surf_frameInfo_second.keypoints, surf_frameInfo_second.descriptors);
    clock_t end_surf = clock();
    std::cout << "SURF working time: " <<  (end_surf-start_surf) << std::endl;
    std::vector<cv::DMatch> surf_matches = MatchDescriptors(surf_frameInfo_first, surf_frameInfo_second);
    drawKeypointsAndLines(first_img, second_img, surf_frameInfo_first, surf_frameInfo_second, surf_matches, "SURF");

};

int main(){

    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    for (int i = 1; i <=10 ; ++i){
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