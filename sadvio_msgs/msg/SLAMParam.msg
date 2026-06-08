# SLAMPARAM
#
# SaDVIO Configuration Message
#
# Compare {slamParameters.h}

std_msgs/Header header

string dataset_path    # !< Path to the dataset
string dataset_id      # !< Id of the dataset
string slam_mode       # !< SLAM mode (mono, bimono, monovio...)
bool multithreading         # !< Allow to run front-end and back-end on different threads (unstable...)
bool enable_visu            # !< Allow visualization
string optimizer       # !< Optimizer type (ReprojectionError, AngularError...)
int32 contrast_enhancer       # !< integer to choose the contrast enhancement algorithm
float32 clahe_clip            # !< Clip of CLAHE (useful only if it is chosen for contrast enhancement)
float32 downsampling          # !< Float to reduce the size of the image (0,5 = half the size of the img)
int32 marginalization         # !< 0 no marginalization, 1 marginalization
bool sparsification         # !< 0 no sparsification, 1 sparsification
string pose_estimator  # !< Type of pose estimator
string rel_pose_estimator  # !< Type of relative pose estimator (NFR or ESKF) [UNUSED]
float32 eskf_r                    # !< Measurement covariance R of ESKF (pixels) [UNUSED]
string tracker         # !< Type of tracking (matcher or klt)
int32 min_kf_number           # !< Minimum KF for optimization
int32 max_kf_number           # !< Size maximum of the sliding windown
int32 fixed_frame_number      # !< Number of fixed frame for gauge fixing
float32 min_lmk_number        # !< Below this number of landmark, a KF is voted
float32 min_movement_parallax # !< Below this parallax, no motion is considered
float32 max_movement_parallax # !< Over this parallax, a KF is voted
bool mesh3d                 # !< 0 no 3D mesh, 1 3D mesh
float64 zncc_tsh             # !< Threshold on ZNCC for triangle filtering
float64 max_length_tsh       # !< Threshold on maximum length for triangle filtering

SLAMFeatures[] features_handled