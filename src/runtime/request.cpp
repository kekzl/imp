#include "runtime/request.h"

namespace imp {

const char* request_status_name(RequestStatus status) {
    using enum RequestStatus;
    switch (status) {
        case PENDING:
            return "PENDING";
        case PREFILLING:
            return "PREFILLING";
        case DECODING:
            return "DECODING";
        case FINISHED:
            return "FINISHED";
        case CANCELLED:
            return "CANCELLED";
        default:
            return "UNKNOWN";
    }
}

}  // namespace imp
