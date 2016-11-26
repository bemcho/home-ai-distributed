#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv/cv.hpp>

#include "caf/all.hpp"


using namespace cv;
using namespace std;


using std::endl;
using std::string;

using namespace caf;

const string window_name("HomeAI");
const string face_cascade_name("../resources/cascade_frontalface.xml");

CascadeClassifier faceDetect;
Mat frame, frame_gray;

int main() {
    VideoCapture capture;

    faceDetect.load(face_cascade_name);
    namedWindow(window_name,WINDOW_OPENGL);
    for (int i = 0; i < 50; i++) {
        capture = VideoCapture(i);
        if (!capture.isOpened()) {
            capture.release();
            cout << "--(!)Error opening video capture\nYou do have camera plugged in, right?" << endl;
            if (i == 49)
                return -1;

            continue;
        } else {
            cout << "--(!)Camera found on " << i << " device index.";
            break;
        }
    }

    capture.set(CAP_PROP_FRAME_WIDTH, 10000);
    capture.set(CAP_PROP_FRAME_HEIGHT, 10000);

    capture.set(CAP_PROP_FRAME_WIDTH,
                (capture.get(CAP_PROP_FRAME_WIDTH) / 2) <= 1280 ? 1280 : capture.get(CAP_PROP_FRAME_WIDTH) / 2);
    capture.set(CAP_PROP_FRAME_HEIGHT,
                (capture.get(CAP_PROP_FRAME_HEIGHT) / 2) <= 720 ? 720 : capture.get(CAP_PROP_FRAME_HEIGHT) / 2);

    while (true) {

        // create another actor that calls 'hello_world(mirror_actor)';
        capture >> frame;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        if (frame.empty()) {
            cout << " --(!) No captured frame -- Break!" << endl;
            break;
        }

        vector<Rect> detects;
        faceDetect.detectMultiScale(frame_gray,detects);

        for(const auto& rect : detects)
        {
            rectangle(frame,rect,CV_RGB(0,255,0),1);
        }

        imshow(window_name,frame);
        //-- bail out if escape was pressed
        if (waitKey(1) == 27) {
            break;
        }

    }
    capture.release();
    exit(0);
}
