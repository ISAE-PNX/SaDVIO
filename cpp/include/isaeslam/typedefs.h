#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <vector>
#include <memory>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <unordered_map>


namespace isae {
    class AFeature;
    class ALandmark;

template<typename T>
using vec_shared = std::vector<std::shared_ptr<T>>;

template<class T> using AlignedVector = std::vector<T,Eigen::aligned_allocator<T>>;
typedef AlignedVector<Eigen::Vector3d> vec3d;

typedef Eigen::Matrix< double, 6, 1 >	Vector6d;

using feature_pair = std::pair<std::shared_ptr<isae::AFeature>,std::shared_ptr<isae::AFeature>>;
using vec_match = std::vector<feature_pair>;
using typed_vec_match = std::unordered_map<std::string, vec_match>;

/// A vector For heterogeneous features
typedef std::unordered_map<std::string, std::vector<std::shared_ptr<isae::AFeature> >> typed_vec_features;
typedef std::unordered_map<std::string, std::vector<std::shared_ptr<isae::ALandmark> >> typed_vec_landmarks;
}

#endif //TYPEDEFS_H
