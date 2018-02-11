#include "GraphCutSegmentation.h"

void GraphCutSegmentation::calcColorVariance(const cv::Mat& origImg)
{
	cv::Scalar tmp { cv::mean(origImg) };
	cv::Vec3d avgColor{tmp[0], tmp[1], tmp[2]};

	std::for_each(origImg.begin<cv::Vec3b>(), origImg.end<cv::Vec3b>(),
		[&avgColor, this](const cv::Vec3b& pix) {
			cv::Vec3d diff = static_cast<cv::Vec3d>(pix) - avgColor;
			cv::pow(diff, 2, diff);
			sigmaSqr += diff;
		}
	);
	sigmaSqr /= static_cast<int>(origImg.total());
}

void GraphCutSegmentation::initComponent(const cv::Mat& origImg, const cv::Mat& seedMask)
{
	calcColorVariance(origImg);
	cv::Mat data_points;
	origImg.convertTo(data_points, CV_32FC3);
	data_points = data_points.reshape(0, origImg.total());

	cv::kmeans(data_points,
			   nCluster,
			   cluster_idx,
			   cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
			   1,
			   cv::KMEANS_RANDOM_CENTERS);

	std::vector<int> obj_hist(nCluster + 1u), bkg_hist(nCluster + 1u);
	bkgRelativeHistogram.resize(nCluster);
	objRelativeHistogram.resize(nCluster);

	for (auto it = seedMask.begin<char>(); it != seedMask.end<char>(); it++)
	{
		auto curPos = it.pos();
		switch (auto curCluster = cluster_idx.at<int>(curPos.y * origImg.cols + curPos.x, 0); *it) 
		{
		case OBJECT:
			obj_hist[curCluster]++;
			obj_hist[nCluster]++;
			break;

		case BACKGROUND:
			bkg_hist[curCluster]++;
			bkg_hist[nCluster]++;
			break;

		default:
			break;
	
		}
	}

	for (auto i = 0u; i < nCluster; i++)
	{
		bkgRelativeHistogram[i] = double(bkg_hist[i]) / bkg_hist[nCluster];
		objRelativeHistogram[i] = double(obj_hist[i]) / obj_hist[nCluster];
	}
}

void GraphCutSegmentation::buildGraph(const cv::Mat& origImg, const cv::Mat& seedMask)
{
	g->add_node(origImg.total());
	cv::Rect imgRect(cv::Point(), origImg.size());

	for (auto it = origImg.begin<cv::Vec3b>(); it != origImg.end<cv::Vec3b>(); it++) 
	{
		auto curPos = it.pos();
		int node = curPos.y * static_cast<int>(imgWidth) + curPos.x;

		double tmpSumNLink = 0.0;

		// Relation to neighbors
		for (auto& k : neighbor)
		{
			cv::Point neighborPos = curPos + k;

			if (imgRect.contains(neighborPos))
			{
				auto neighborNode = convertPixelToNode(neighborPos);
				auto tmpNWeight = calcNWeight(curPos, neighborPos, origImg);
				tmpSumNLink += 2 * tmpNWeight;

				g->add_edge(node, neighborNode,
							tmpNWeight,
							tmpNWeight);
			}
		}
		K = std::max(tmpSumNLink, K);
	}

	K += 1.0;

	for (auto it = seedMask.begin<char>(); it != seedMask.end<char>(); it++) 
	{
		cv::Point curPos = it.pos();
		int node = curPos.y * static_cast<int>(imgWidth) + curPos.x;
		// Relation to source and sink
		g->add_tweights(
			node,
			calcTWeight(curPos, *it),
			calcTWeight(curPos, *it, false));
	
	}
}

double GraphCutSegmentation::calcTWeight(const cv::Point& pix, int pixType, bool toSource)
{
	switch (pixType)
	{
	case OBJECT:
		return (toSource ? K : 0);
		break;

	case BACKGROUND:
		return (toSource ? 0 : K);
		break;

	default:
		return (toSource ? lambda * Pr_bkg(pix) : lambda * Pr_obj(pix));
		break;
	}
}

double GraphCutSegmentation::calcNWeight(const cv::Point& pix1, const cv::Point& pix2, const cv::Mat& origImg)
{
	const auto& r1 = origImg.at<cv::Vec3b>(pix1),
		 		r2 = origImg.at<cv::Vec3b>(pix2);
	double intensityDiff{ 0.0 };
	for (int i = 0; i < dim; i++)
	{
		double tmpDiff = static_cast<double>(r1[i]) - r2[i];
		intensityDiff += (tmpDiff * tmpDiff / (2 * sigmaSqr[i]));
	}
	auto dist = pix2 - pix1;
	// cv::norm is much slower than manual Euclidean distance implementation
	return std::exp(-intensityDiff) / std::sqrt(dist.x * dist.x + dist.y * dist.y);
}

double GraphCutSegmentation::Pr_bkg(const cv::Point& pix)
{
	return -std::log(bkgRelativeHistogram[cluster_idx.at<int>(convertPixelToNode(pix), 0)]);
}

double GraphCutSegmentation::Pr_obj(const cv::Point& pix)
{
	return -std::log(objRelativeHistogram[cluster_idx.at<int>(convertPixelToNode(pix), 0)]);
}

void GraphCutSegmentation::cutGraph(cv::Mat& outputMask)
{
	outputMask.create(cv::Size(imgWidth, imgHeight), CV_8U);
	// double flow = 0.0;
	// flow = g->maxflow(!runFirstTime, NULL);
	// runFirstTime = false;
	std::cout << double(g->maxflow()) << std::endl;
	auto node = 0u;
	std::for_each(outputMask.begin<uchar>(), outputMask.end<uchar>(),
		[this, &node](uchar& pix){
			pix = g->what_segment(node++) == GraphType::SOURCE ? static_cast<uint8_t>(~0u) : 0u;
		}
	);
}

void GraphCutSegmentation::segment(const cv::Mat& img, const cv::Mat& seedMask, cv::Mat& outputMask)
{
	g.reset(new GraphType(img.total(), img.total() * (neighbor.size() + 2)));
	imgWidth = img.cols;
	imgHeight = img.rows;

	initComponent(img, seedMask);

	buildGraph(img, seedMask);

	cutGraph(outputMask);
}

void GraphCutSegmentation::updateSeeds(const std::vector<cv::Point>& newSeeds, PixelType pixType, cv::Mat& outputMask)
{
	for (const auto& p : newSeeds)
	{
		g->mark_node(convertPixelToNode(p));
		g->add_tweights(
			convertPixelToNode(p),
			calcTWeight(p, pixType),
			calcTWeight(p, pixType, false));
	}
	cutGraph(outputMask);
}

GraphCutSegmentation::GraphCutSegmentation()
{
	createDefault();
}

GraphCutSegmentation::~GraphCutSegmentation()
{
	cleanGarbage();
}
