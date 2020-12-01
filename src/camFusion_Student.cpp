
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));
    // create constant RED (BGR) color for Lidar points
    cv::Scalar currColor = cv::Scalar(0, 0, 255);
    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        // Use next line to generate random new colors
        // cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (auto &match : kptMatches)
    {
        const auto &currKeyPoint = kptsCurr[match.trainIdx].pt;
        if (boundingBox.roi.contains(currKeyPoint))
        {
            boundingBox.kptMatches.emplace_back(match);
        }
    }

    double sum = 0;
    // std::cout << "Distance of Matches: \n";
    // Remove outlier matches based on the euclidean distance between them in relation to all the matches in the bounding box.
    for (auto &it : boundingBox.kptMatches)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it.trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it.queryIdx);
        double dist = cv::norm(kpCurr.pt - kpPrev.pt);
        sum += dist;

        // std::cout << dist << "\n";
    }
    double mean = sum / boundingBox.kptMatches.size();
    double ratio = 1.5;
    for (auto it = boundingBox.kptMatches.begin(); it < boundingBox.kptMatches.end();)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);
        double dist = cv::norm(kpCurr.pt - kpPrev.pt);

        if (dist >= mean * ratio)
        {
            boundingBox.kptMatches.erase(it);
        }
        else
        {
            it++;
        }
    }
}
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; it1++)
    {
        // get current keypoint and its matched partner in the prev frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++)
        {
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     //eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.empty())
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    // std::cout << "Distance Ratios: " << std::endl;
    for (auto &dist : distRatios)
    {
        // std::cout << dist << " ";
    }
    // std::cout << std::endl;

    long medIndex = floor(distRatios.size() / 2.0);
    // compute median dist. ratio to remove outlier influence
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

    // std::cout << "medDistRatio = " << medDistRatio << std::endl;

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    
    bool fPrint = false;
    if (fPrint)
    {
        std::cout << "Camera TTC = " << TTC << "s.\n";
    }
    fPrint = false;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr removeLidarOutlier(const std::vector<LidarPoint> &lidarPoints, float clusterTolerance)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &p : lidarPoints)
    {
        cloud->push_back(pcl::PointXYZ((float)p.x, (float)p.y, (float)p.z));
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setSearchMethod(tree);
    ec.setMinClusterSize(50);
    ec.setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    if (cluster_indices.empty())
        return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &cluster : cluster_indices)
        for (size_t idx : cluster.indices)
            result->points.push_back(cloud->points.at(idx));
    return result;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1 / frameRate; // time between two measurements in seconds
    double laneWidth = 4.0;    // assmed width of ego lane
    float clusterTolerance = 0.05;

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;

    // Apply euclidean clustering to remove outliers
    auto clusteredPrevPts = removeLidarOutlier(lidarPointsPrev, clusterTolerance);
    auto clusteredCurrPts = removeLidarOutlier(lidarPointsCurr, clusterTolerance);

    // find closest distance to Lidar points within ego lane
    for (auto &it : clusteredPrevPts->points)
    {
        if (fabs(it.y) <= laneWidth / 2.0)
        {
            minXPrev = minXPrev > it.x ? it.x : minXPrev;
        }
    }

    for (auto &it : clusteredCurrPts->points)
    {
        if (fabs(it.y) <= laneWidth / 2.0)
        {
            minXCurr = minXCurr > it.x ? it.x : minXCurr;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);

    bool fPrint = false;
    if (fPrint)
    {
        std::cout << "Lidar TTC = " << TTC << "s.\n";
    }
    fPrint = false;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    for (const auto &prevBox : prevFrame.boundingBoxes)
    {
        std::map<int, int> m;
        for (const auto &currBox : currFrame.boundingBoxes)
        {
            // queryIdx refers to keypoints1 and trainIdx refers to keypoints2
            for (const auto &match : matches)
            {
                const auto &prevKeyPoint = prevFrame.keypoints[match.queryIdx].pt;
                if (prevBox.roi.contains(prevKeyPoint))
                {
                    const auto &currKeyPoint = currFrame.keypoints[match.trainIdx].pt;
                    if (currBox.roi.contains(currKeyPoint))
                    {
                        if (0 == m.count(currBox.boxID))
                        {
                            m[currBox.boxID] = 1;
                        }
                        else
                        {
                            m[currBox.boxID]++;
                        }
                    }
                }

            } // eof iterating all matches

        } // eof iterating all current bounding boxes
        auto bestMatch = std::max_element(m.begin(), m.end(), [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) { return p1.second < p2.second; });

        bbBestMatches[prevBox.boxID] = bestMatch->first;

        bool fPrint = false;
        if (fPrint)
        {
            std::cout << "ID Matching: " << prevBox.boxID << " => " << bestMatch->first << "\n";
        }
        fPrint = false;

    } // eof iterating all previous bounding boxes
}