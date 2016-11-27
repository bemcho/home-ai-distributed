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
    VideoCapture capture; //= VideoCapture("/home/bemcho/Movies/wtrust.mkv");

    faceDetect.load(face_cascade_name);

    namedWindow(window_name, WINDOW_OPENGL);
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

    int long long fc = 1;
    vector<Rect> detectsFront, detectsProfile;
    while (true) {

        capture >> frame;
        fc++;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        if (frame.empty()) {
            cout << " --(!) No captured frame -- Break!" << endl;
            break;
        }

        /**don't call heavy algorithms 30 or more times per second
        *call them every 10-th second
        *just this check gains more than 40% less cpu usage(on my machine debian 8 with I7)
        * without fc counter was 56% from all  cores with the counter is 13 %
        **/
        if (fc % 5 == 0) {
            detectsFront.clear();
            faceDetect.detectMultiScale(frame_gray, detectsFront, 1.1, 7, 0 | CASCADE_SCALE_IMAGE);
        }

        for (const auto &rect : detectsFront) {
            rectangle(frame, rect, CV_RGB(0, 125, 255), 2);
        }

        imshow(window_name, frame);
        //-- bail out if escape was pressed
        if (waitKey(1) == 27) {
            break;
        }

    }
    capture.release();
    exit(0);
}
