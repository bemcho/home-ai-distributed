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

template <class Inspector>
typename std::enable_if<Inspector::reads_state,
        typename Inspector::result_type>::type
inspect(Inspector& f, cv::Rect& rect) {
    return f(meta::type_name("Rect"), rect.x, rect.y, rect.width, rect.height);
    }

template <class Inspector>
typename std::enable_if<Inspector::writes_state,
        typename Inspector::result_type>::type
inspect(Inspector& f, cv::Rect& rect) {
    int x;
    int y;
    int width;
    int height;
    // write back to rect at scope exit
    auto g = caf::detail::make_scope_guard([&] {
        rect.x = x;
        rect.y = y;
        rect.width = width;
        rect.height = height;
    });
    return f(meta::type_name("Rect"), x,y,width,height);
}

template <class Inspector>
typename std::enable_if<Inspector::reads_state,
        typename Inspector::result_type>::type
inspect(Inspector& f, cv::Mat& mat) {
    return f(meta::type_name("Mat"), mat.data, mat.rows, mat.cols);
}

template <class Inspector>
typename std::enable_if<Inspector::writes_state,
        typename Inspector::result_type>::type
inspect(Inspector& f, cv::Mat& mat) {
    uchar* data;
    int rows;
    int cols;
    // write back to mat at scope exit
    auto g = caf::detail::make_scope_guard([&] {
        mat.data = data;
        mat.rows = rows;
        mat.cols = cols;
    });
    return f(meta::type_name("Mat"), mat.data, mat.rows, mat.cols);
}

behavior mirror(event_based_actor *self) {
    // return the (initial) actor behavior
    return {
            // a handler for messages containing a single string
            // that replies with a string
            [=](const Mat& frame_gray) -> vector<Rect> {
                // prints "Hello World!" via aout (thread-safe cout wrapper)
                vector<Rect> localDetects;
                faceDetect.detectMultiScale(frame_gray, localDetects);                // reply "!dlroW olleH"
                return localDetects;
            }
    };
}

void hello_world(event_based_actor *self, const actor &buddy,const Mat& frame_gray) {
    // send "Hello World!" to our buddy ...
    self->request(buddy, std::chrono::seconds(10), frame_gray).then(
            // ... wait up to 10s for a response ...
            [&](const vector<Rect> detects) {
                // ... and print it
                for (const auto &rect : detects) {
                    rectangle(frame, rect, CV_RGB(0, 255, 0), 1);
                    putText(frame, "Human", Point(rect.x, rect.y - 20), CV_FONT_NORMAL, 1.0, CV_RGB(0, 255, 0), 1);

                }
                imshow(window_name, frame);
            }
    );
}

int main() {
    VideoCapture capture;

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

    long fc = 6;
    // our CAF environment
    actor_system_config cfg;
    actor_system system{cfg};
    // create a new actor that calls 'mirror()'
    auto mirror_actor = system.spawn(mirror);
    // create another actor that calls 'hello_world(mirror_actor)';
    system.spawn(hello_world, mirror_actor);
    // system will wait until both actors are destroyed before leaving main

    while (true) {

        // create another actor that calls 'hello_world(mirror_actor)';
        system.spawn(hello_world, mirror_actor);
        capture >> frame;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        if (frame.empty()) {
            cout << " --(!) No captured frame -- Break!" << endl;
            break;
        }


        fc++;
        //-- bail out if escape was pressed
        if (waitKey(1) == 27) {
            break;
        }

    }
    exit(0);
}
