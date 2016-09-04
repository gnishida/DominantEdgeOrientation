/****************************************************************************************************************************************
 *
 * testdataフォルダの画像を全て読み込み、エッジ画像をedgesフォルダへ、エッジラインをresultsフォルダへ、
 * dominant edge orientationに基づいてwarpした画像をwarpedフォルダへ、
 * 正しいorientationに基づいてwarpした画像をtrue_warpedフォルダへ保存する。

 * Updates:
 * 9/4/2016		Canny edge detectorの結果について、gradient orientationが水平／垂直のedgeのみ残す
 *
 * @author Gen Nishida
 * @date 9/2/2016
 *
 ***************************************************************************************************************************************/


#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "HoughTransform.h"
#include "EdgeDetection.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include "CVUtils.h"

int main() {
	float thresholdRatio = 0.0;
	int topN = 2;
	int edge_detector = 0;	// 0 -- Canny(50, 120) / 1 -- autoCanny(0.33) / 2 -- autoCanny2
	bool smooth_accum = false;
	float angle_threshold = 15.0f;
	bool remove_diagonal_edges = false;
	bool use_magnitude_as_weight = true;

	std::vector<float> true_hori = { -1, 0, 1.2f, 0, 0, -0.2f, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, -3.4f, -1, 3, 3, 1, 2.5f, -1 };
	std::vector<float> true_vert = { 88, 87, 90, 88, 90, 89, 88, 90, 88, 89, 90, 91, 86, 90, 94, 89, 87, 91.7f, 90, 98, 94, 90, 90.5f, 83 };
	float error = 0.0f;

	boost::filesystem::path dir("../testdata/");
	boost::filesystem::path dir_edges("../edges/");
	boost::filesystem::path dir_out("../results/");
	boost::filesystem::path dir_warp("../warped/");
	boost::filesystem::path dir_true("../true_warped/");
	int cnt = 0;
	for (auto it = boost::filesystem::directory_iterator(dir); it != boost::filesystem::directory_iterator(); ++it, ++cnt) {
		if (boost::filesystem::is_directory(it->path())) continue;

		// read an image and detect edges
		cv::Mat img = cv::imread(dir.string() + it->path().filename().string());

		// save the edge images
		{
			cv::Mat grayImg;
			cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
			//cv::blur(grayImg, grayImg, cv::Size(5, 5));
			cv::blur(grayImg, grayImg, cv::Size(3, 3));
			cv::Mat edgeImg;
			if (edge_detector == 0) {
				cv::Canny(grayImg, edgeImg, 50, 120);
			}
			else if (edge_detector == 1) {
				ed::autoCanny(grayImg, edgeImg);
			}
			else if (edge_detector == 2) {
				ed::autoCanny2(grayImg, edgeImg);
			}

			if (remove_diagonal_edges) {
				ed::removeDiagonalEdges(grayImg, edgeImg, angle_threshold);
			}

			cv::imwrite(dir_edges.string() + it->path().filename().string(), edgeImg);
		}

		// find the dominant edge orientations
		std::pair<float, float> orientation = ed::detectHorizontalAndVerticalDominantOrientation(img, angle_threshold, remove_diagonal_edges, use_magnitude_as_weight, smooth_accum, thresholdRatio);
		error += (orientation.first - true_hori[cnt]) * (orientation.first - true_hori[cnt]) + (orientation.second - true_vert[cnt]) * (orientation.second - true_vert[cnt]);
		std::cout << it->path().filename().string() << ": " << orientation.first << " (T: " << true_hori[cnt] << "), " << orientation.second << " (T: " << true_vert[cnt] << ")" << std::endl;
		
		// warp the facade image
		cv::Mat warped;
		ed::warp(img, orientation.first, orientation.second, warped);
		cv::imwrite(dir_warp.string() + it->path().filename().string(), warped);
		ed::warp(img, true_hori[cnt], true_vert[cnt], warped);
		cv::imwrite(dir_true.string() + it->path().filename().string(), warped);

		// detect horizontal and vertical edges
		std::vector<std::tuple<glm::vec2, glm::vec2, int, float>> edges = ed::detectHorizontalAndVerticalEdges(img, angle_threshold, edge_detector, remove_diagonal_edges, use_magnitude_as_weight, smooth_accum, topN);

		// define the line width for drawing edges
		int line_width = std::max(1, (img.rows + 40) / 80);

		// get the max votes
		float h_max_votes = 0.0f;
		float v_max_votes = 0.0f;
		for (int i = 0; i < std::min(2, (int)edges.size()); ++i) {
			glm::vec2 p1, p2;
			int type;
			float votes;
			std::tie(p1, p2, type, votes) = edges[i];
			if (type == 0) {
				h_max_votes = votes;
			}
			else {
				v_max_votes = votes;
			}
		}

		// craete an image that contains only the edge lines
		cv::Mat result(img.size(), CV_8UC4, cv::Scalar(255, 255, 255, 0));
		for (int i = edges.size() - 1; i >= 0; --i) {
			glm::vec2 p1, p2;
			int edge_type;
			float votes;
			std::tie(p1, p2, edge_type, votes) = edges[i];
			//std::cout << edge_type << ", " << votes << ", (" << p1.x << "," << p1.y << "), (" << p2.x << ", " << p2.y << ")" << std::endl;

			if (edge_type == 0) {
				if (votes > h_max_votes * thresholdRatio) {
					cv::line(result, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 0, 255, votes / h_max_votes * 255), line_width, cv::LINE_8);
				}
			}
			else {
				if (votes > v_max_votes * thresholdRatio) {
					cv::line(result, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(255, 0, 0, votes / v_max_votes * 255), line_width, cv::LINE_8);
				}
			}
		}

		// alpha blend the images
		cvutils::blend(result, img, result);
		cv::imwrite(dir_out.string() + it->path().filename().string(), result);
	}

	std::cout << "Dominant orientationi estimation error: " << sqrt(error / true_hori.size()) << std::endl;

	return 0;
}