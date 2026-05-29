# SLAMPROFILING
#
# SaDVIO SLAM profiling Message (indicators of SLAM performance)
#
# Compare {slam_profiling.txt}

std_msgs/Header header

string dataset_id       # Name of dataset
uint32 nframes          # Number of frames processed until now
uint32 nkeyframes       # Number of keyframes processed until now
float32 img_process_dt  # Average processing time per image (currently non-functional)
float32 avg_matches_frame # Average number of matches between frames
float32 avg_Mathes_time   # Average number of matches ???
