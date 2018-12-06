#ifndef DOLPHIN_FRAME_FRAME_HANDLER_H
#define DOLPHIN_FRAME_FRAME_HANDLER_H

#include <ci>

namespace dol {


  // Position handler
  struct RoiPos {
    const Pos16 * roiPos_;
    int roiNb_;

    CI_HODE RoiPos(const Pos16 *roiPos, int roiNb)
        : roiPos_(roiPos), roiNb_(roiNb) {}

    CI_HODE Pos16 pos(int wfsId, int roiId) const {
      return roiPos_[wfsId * roiNb_ + roiId];
    }
  };

  struct PixPosPyr {
    const Pos16 * pixPos_;

    CI_HODE PixPosPyr(const Pos16 *pixPos) : pixPos_(pixPos) {}

    CI_HODE Pos16 pos(int pixId) const { return pixPos_[pixId]; }
  };

  struct PixPosSH {
    int roiSize_;

    CI_HODE PixPosSH(int roiSize) : roiSize_(roiSize) {}

    CI_HODE Pos16 pos(int pixId) const {
      return Pos16(pixId % roiSize_, pixId / roiSize_);
    }
  };

  template <typename PixPos>
  struct WFSPos {
    RoiPos roiPos_;
    PixPos pixPos_;

    CI_HODE WFSPos(Pos16 const *roiPos, int roiNb, PixPos pixPos)
        : roiPos_(roiPos, roiNb), pixPos_(pixPos) {}

    CI_HODE Pos16 pos(int wfsId, int roiId, int pixId) const {
      return roiPos_.pos(wfsId, roiId) + pixPos_.pos(pixId);
    }
  };

// Frame handler

template <typename PixPos>
struct Frame {
  ci::FastDeviceBuffer<FrameT> frameBuffer_;

  WFSPos<PixPos> wfsPos_;

  int wfsNb_;
  int frameSize_;

  CI_HODE Frame(ci::FastDeviceBuffer<FrameT> frameBuffer, WFSPos<PixPos> wfsPos, int wfsNb, int frameSize) :
    frameBuffer_(frameBuffer),
    wfsPos_(wfsPos), wfsNb_(wfsNb),
    frameSize_(frameSize)
  {}

  CI_HODE Pos16 wfsOffset(int wfsId) const {
    return Pos16(0, frameSize_ * wfsId);
  }

  CI_HODE Pos16 posAt(int wfsId, int roiId, int pixId) const {
    return wfsPos_.pos(wfsId, roiId, pixId) + wfsOffset(wfsId);
  }

  CI_HODE DDataT pixel(Pos16 pos) const {
    FrameT const *__restrict__ framePix =
        framePtr_ + pos.x + pos.y * frameSize_;
    return static_cast<DDataT>(readPixel(framePix));
  }

  CI_HODE DDataT pixel(int wfsId, int roiId, int pixId) const {
    return pixel(posAt(wfsId, roiId, pixId));
  }

  CI_DEVICE void reset() const {
    if(ID < frameSize_ * frameSize_ * wfsNb_)
      dol::deviceStore(frameBuffer_.ptr(), pk::guard::getLock<FrameT>());
  }
};

}  // namespace dol

#endif  // DOLPHIN_FRAME_FRAME_HANDLER_H
