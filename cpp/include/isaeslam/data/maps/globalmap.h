#ifndef GLOBALMAP_H
#define GLOBALMAP_H

#include "isaeslam/data/maps/amap.h"

namespace isae {

class GlobalMap : public AMap {
  public:
    GlobalMap() = default;
    void addFrame(std::shared_ptr<Frame> &frame) override;

  protected:
    void pushLandmarks(std::shared_ptr<Frame> &frame) override;

};

} // namespace isae

#endif // GLOBALMAP_H