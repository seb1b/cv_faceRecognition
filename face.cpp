#include "face.h"

using namespace std;

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

