/*
 * openRatSLAM
 *
 * main_lv - ROS interface bindings for the local view cells
 * MODIFIED FOR EVENT CAMERA INPUT with SAE and full integration
 */

#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <boost/property_tree/ini_parser.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dvs_msgs/EventArray.h>
#include <ratslam_ros/ViewTemplate.h>
#include "ratslam/local_view_match.h"
#include "utils/utils.h"

#if HAVE_IRRLICHT
#include "graphics/local_view_scene.h"
ratslam::LocalViewScene *lvs = NULL;
bool use_graphics;
#endif


using namespace ratslam;
ratslam::LocalViewMatch * lv = NULL;

// ---- Global variables for our SAE ----
ros::Publisher pub_vt;
cv::Mat sae_canvas; // The canvas for our Surface of Active Events
bool sae_initialized = false;

void event_callback(const dvs_msgs::EventArray::ConstPtr& events)
{
  // ---- SAE INITIALIZATION ----
  if (!sae_initialized)
  {
    sae_canvas = cv::Mat(events->height, events->width, CV_8UC1, cv::Scalar(0));
    sae_initialized = true;
    ROS_INFO_STREAM("SAE canvas initialized to " << events->width << "x" << events->height);
  }

  // ---- SAE ALGORITHM ----
  // 1. DECAY: Slower decay to create a stronger image from fast movement
  sae_canvas *= 0.98;

  // 2. ACCUMULATE: Paint the new events onto the canvas
  for (const auto& event : events->events)
  {
    // Boundary check to prevent crashes from out-of-bounds event coordinates
    if (event.x < sae_canvas.cols && event.y < sae_canvas.rows)
    {
        sae_canvas.at<uint8_t>(event.y, event.x) = 255;
    }
  }

  // ---- SAE VISUALIZATION ----
  cv::imshow("SAE", sae_canvas);
  cv::waitKey(1);

  // ---- THE CRITICAL FIX: RESIZE THE IMAGE ----
  // Create a new Mat to hold the resized image. RatSLAM core expects 64x64.
  cv::Mat resized_sae;
  cv::resize(sae_canvas, resized_sae, cv::Size(64, 64));

  // ---- INTEGRATION WITH RATSLAM CORE ----
  // Store the ID of the current template BEFORE processing the new image
  unsigned int prev_id = lv->get_current_vt();

  // Feed the correctly-sized SAE into the LocalViewMatch object
  lv->on_image(resized_sae.data, true, resized_sae.cols, resized_sae.rows);

  // Get the ID of the template AFTER processing
  unsigned int new_id = lv->get_current_vt();

  // ---- PUBLISH ONLY IF A NEW TEMPLATE IS CREATED ----
  // If the ID has changed, it means the system has recognized a new view.
  if (prev_id != new_id)
  {
    ROS_INFO("Action taken! Published View Template ID: %d", new_id);
    static ratslam_ros::ViewTemplate vt_output;
    vt_output.header.stamp = ros::Time::now();
    vt_output.header.seq++;
    vt_output.current_id = new_id;
    vt_output.relative_rad = lv->get_relative_rad();
    pub_vt.publish(vt_output);
  }


#ifdef HAVE_IRRLICHT
  if (use_graphics)
  {
    // Re-enable the original graphics visualization
    lvs->draw_all();
  }
#endif
}

int main(int argc, char * argv[])
{
  ROS_INFO_STREAM(argv[0] << " - openRatSLAM Copyright (C) 2012 David Ball and Scott Heath");
  ROS_INFO_STREAM("RatSLAM algorithm by Michael Milford and Gordon Wyeth");
  ROS_INFO_STREAM("Distributed under the GNU GPL v3, see the included license file.");

  if (argc < 2)
  {
    ROS_FATAL_STREAM("USAGE: " << argv[0] << " <config_file>");
    exit(-1);
  }
  std::string topic_root = "";

  boost::property_tree::ptree settings, ratslam_settings, general_settings;
  read_ini(argv[1], settings);

  get_setting_child(general_settings, settings, "general", true);
  get_setting_from_ptree(topic_root, general_settings, "topic_root", (std::string)"");
  get_setting_child(ratslam_settings, settings, "ratslam", true);
  lv = new ratslam::LocalViewMatch(ratslam_settings);

  if (!ros::isInitialized())
  {
    ros::init(argc, argv, "RatSLAMViewTemplate");
  }
  ros::NodeHandle node;

  pub_vt = node.advertise<ratslam_ros::ViewTemplate>(topic_root + "/LocalView/Template", 1);

  ros::Subscriber sub = node.subscribe(topic_root + "/dvs/events", 10, event_callback);
  ROS_INFO_STREAM("LV: Subscribed to " << topic_root << "/dvs/events");

#ifdef HAVE_IRRLICHT
    boost::property_tree::ptree draw_settings;
    get_setting_child(draw_settings, settings, "draw", true);
    get_setting_from_ptree(use_graphics, draw_settings, "enable", true);
    if (use_graphics)
      lvs = new ratslam::LocalViewScene(draw_settings, lv);
#endif

  ros::spin();

  return 0;
}


