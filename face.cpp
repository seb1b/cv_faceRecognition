#include "face.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

class FACE::FACEPimpl {
public:
	map<string, cv::Vec3b> centerColor;
};


/// Constructor
FACE::FACE() {
	
	pimpl = std::shared_ptr<FACEPimpl>(new FACEPimpl());
}

/// Destructor
FACE::~FACE() {
}

/// Start the training.  This resets/initializes the model.
void FACE::startTraining() {
}

/// Add a new person.
///
/// @param img:  250x250 pixel image containing a scaled and aligned face
/// @param name: name of the person who corresponds to img
void FACE::train(const cv::Mat3b& img, const string& name) {
    
    int minHessian = 400;
    
    SurfFeatureDetector detector( minHessian );
    
    std::vector<KeyPoint> keypoints_1;
    
    detector.detect( img, keypoints_1 );
    
    //-- Draw keypoints
    Mat img_keypoints_1;
    
    drawKeypoints( img, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    
    //-- Show detected (drawn) keypoints
    imshow("Keypoints ", img_keypoints_1 );
    waitKey(600);
	
	pimpl->centerColor[name] = img(124,124);
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void FACE::finishTraining() {
}

/// Verify if img corresponds to the provided name.  The result is a floating point
/// value directly proportional to the probability of being correct.
///
/// @param img:  test image
/// @param name: possible name of the person corresponding to img and that we want to verify
/// @return:    probability of human likelihood
double FACE::verify(const cv::Mat3b& img, const string& name) {
	
	return rand()%256;
	return 1.-abs(pimpl->centerColor[name][0]-img(124,124)[0])-abs(pimpl->centerColor[name][1]-img(124,124)[1])-abs(pimpl->centerColor[name][2]-img(124,124)[2]);
}

