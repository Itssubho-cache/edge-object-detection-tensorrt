#include <cstdint>

// Structure to store frame timing info
struct FrameTimestamp {
    uint64_t pts;  // Presentation Timestamp
    uint64_t dts;  // Decode Timestamp
};

FrameTimestamp generateTimestamps(uint64_t frameIndex, uint64_t fps) {
    FrameTimestamp ts;
    ts.pts = (frameIndex * 1000000) / fps;
    ts.dts = ts.pts; // For now, same as PTS

    return ts;
}

// TODO:
// - Integrate with GStreamer buffers
// - Handle jitter and reordering
// - Support RTP timestamp mapping
