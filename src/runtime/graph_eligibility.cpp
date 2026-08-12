#include "runtime/graph_eligibility.h"

namespace imp {

const char* graph_demotion_reason_name(GraphDemotionReason r) {
    switch (r) {
        case GraphDemotionReason::None:
            return "none";
        case GraphDemotionReason::ConfigNever:
            return "runtime.cuda_graphs=never";
        case GraphDemotionReason::Gemma4NoGraphs:
            return "gemma4.no_graphs=true";
        case GraphDemotionReason::CalibrationActive:
            return "calibration_active";
        case GraphDemotionReason::StreamingKvConfigured:
            return "streaming_kv_configured";
        case GraphDemotionReason::ExpertsOnHost:
            return "experts_on_host";
        case GraphDemotionReason::PinnedSampleBufUnavailable:
            return "pinned_sample_buf_unavailable";
        case GraphDemotionReason::StreamingKvKvPressure:
            return "streaming_kv_kv_pressure";
    }
    return "?";
}

bool graph_demotion_is_mid_run(GraphDemotionReason r) {
    return r == GraphDemotionReason::StreamingKvKvPressure;
}

}  // namespace imp
