#include "runtime/request.h"

namespace imp {

const char* request_status_name(RequestStatus status) {
    switch (status) {
        case RequestStatus::PENDING:     return "PENDING";
        case RequestStatus::PREFILLING:  return "PREFILLING";
        case RequestStatus::DECODING:    return "DECODING";
        case RequestStatus::FINISHED:    return "FINISHED";
        case RequestStatus::CANCELLED:   return "CANCELLED";
        default:                         return "UNKNOWN";
    }
}

} // namespace imp
