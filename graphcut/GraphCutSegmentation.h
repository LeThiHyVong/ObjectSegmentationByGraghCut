#ifndef GRAPHCUT_SEGMENTATION_H_
#define GRAPHCUT_SEGMENTATION_H_

#include "../max_flow/graph.h"
#include "../stdc++.h"

/**
 * @todo Use Boost implementation of Boykov maxflow algorithm, and compare to push-relabel or edmond-karp algorithm
 * @todo Refactor 
 * @todo Doxygen comment style
 * @todo Iterate image for each pixel with custom callback process each pixel
 * @ref http://www.boost.org/doc/libs/1_60_0/libs/graph/doc/boykov_kolmogorov_max_flow.html
 */
class GraphCutSegmentation
{
	typedef Graph<double, double, double> GraphType;

  public:
	//! The enumeration type represent state of each pixel
	enum PixelType
	{
		BACKGROUND = -1,
		UNKNOWN = 0,
		OBJECT = 1
	};

	//! The default no parameter constructor
	GraphCutSegmentation();

	//! Default destructor
	~GraphCutSegmentation();

	//! Set the number of color clusters
	/**
	 * \param[in] _cluster The custom number of cluster
	*/
	void setNCluster(int _cluster);

	//! Set the number of color dimension
	/**
	 * \param[in] _dim The custom number of color dimensions, usually it the number of image channels
	*/
	void setNDimension(int _dim);

	//! Calculate the image's color variance for each dimension
	/**
	 * \param[in] origImg the original image to process
	*/
	void calcColorVariance(const cv::Mat &origImg);

	//! Set the relationship coefficience betwwen regional and boundary term
	/**
	 * \param[in] _lambda a non-negative decimal number
	*/
	void setRegionBoundaryRelation(double _lambda);

	//! Initialize component for processing
	/**
	 * \param[in] origImg the image need separating
	 * \param[in] seedMask the sample points provided by user
	*/
	void initComponent(const cv::Mat &origImg, const cv::Mat &seedMask);

	//! Build the directed graph (flow) 
	/**
	 * \param[in] origImg the image need separating
	 * \param[in] seedMask the sample points provided by user
	*/
	void buildGraph(const cv::Mat &origImg, const cv::Mat &seedMask);

	//! Find the minimum cut to find optimal segmentation 
	/**
	 * \param[out] outMask output mask represent the segmentation result
	*/
	void cutGraph(cv::Mat &outMask);

	//! Whole segmentation process, just a combination of #initComponent, #buildGraph and #cutGraph 
	/**
	 * \param[in] img the image need separating
	 * \param[in] seedMask the sample points provided by user
	 * \param[out] outputMask output mask represent the segmentation result
	*/
	void segment(const cv::Mat &img, const cv::Mat &seedMask, cv::Mat &outputMask);

	//! @todo incremental edit
	void updateSeeds(const std::vector<cv::Point> &newSeeds, PixelType pixType, cv::Mat &outputMask);

	//! create initial state
	void createDefault();

	//! collect memory garbage
	void cleanGarbage();

  private:
	//! Relational coordinate of 8-neighbor relationship
	const std::vector<cv::Point> neighbor{
		{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};

	//! the flow
	std::unique_ptr<GraphType> g;
	
	uint32_t imgWidth, imgHeight;

	double K;

	uint32_t nCluster;
	uint8_t dim;
	cv::Vec3d sigmaSqr;
	double lambda;
	// bool runFirstTime; not used yet

	cv::Mat cluster_idx;

	std::vector<double> bkgRelativeHistogram;
	std::vector<double> objRelativeHistogram;

	void initParam();

	// From terminal to internal
	double calcTWeight(const cv::Point &pix, int pixType, bool toSource = true);

	// Between internal
	double calcNWeight(const cv::Point &pix1, const cv::Point &pix2, const cv::Mat &origImg); //B_pq

	uint32_t convertPixelToNode(const cv::Point &);

	cv::Point convertNodeToPixel(uint32_t node);

	double Pr_bkg(const cv::Point &);

	double Pr_obj(const cv::Point &);

	uint32_t getNumNodes(const cv::Mat &img);

	uint32_t getNumEdges(const cv::Mat &img);

	template<typename ...T>
	void iterateImg(const cv::Mat& img, std::function<void(int, int, T& ...)> pixRelatedProc,T& ... args) {
		for (int r = 0; r < img.rows; r++)
			for (int c = 0; c < img.cols; c++)
				pixRelatedProc(r, c, args...);
	}
};

inline uint32_t GraphCutSegmentation::getNumEdges(const cv::Mat &img)
{
	return img.cols * img.rows * (neighbor.size() + 2u);
}

inline uint32_t GraphCutSegmentation::getNumNodes(const cv::Mat &img)
{
	return img.cols * img.rows;
}

inline uint32_t GraphCutSegmentation::convertPixelToNode(const cv::Point &pix)
{
	return pix.y * imgWidth + pix.x;
}

inline cv::Point GraphCutSegmentation::convertNodeToPixel(uint32_t node)
{
	return cv::Point(node % imgWidth, node / imgWidth);
}

inline void GraphCutSegmentation::setNCluster(int _cluster)
{
	nCluster = _cluster;
}

inline void GraphCutSegmentation::setNDimension(int _dim)
{
	dim = _dim;
}

inline void GraphCutSegmentation::setRegionBoundaryRelation(double _lambda)
{
	lambda = _lambda;
}

inline void GraphCutSegmentation::initParam()
{
	setNCluster(20);
	setNDimension(3);
	setRegionBoundaryRelation(.5);
	sigmaSqr = {0.0, 0.0, 0.0};
	K = 0.0;
	// runFirstTime = true;
}

inline void GraphCutSegmentation::createDefault()
{
	initParam();
}

inline void GraphCutSegmentation::cleanGarbage()
{
	auto tmpPtr = g.release();
	if (tmpPtr != NULL)
	{
		tmpPtr->reset();
		delete tmpPtr;
	}

	//changedNode->Reset();
	//delete changedNode;
	//changedNode = NULL;
}

#endif /* FUSION_FRAMEWORK_H_ */
