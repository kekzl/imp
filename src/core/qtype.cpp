#include "core/qtype.h"

namespace imp {

size_t qtype_elem_bytes(QType q) {
    switch (q) {
        case QType::F32:
            return 4;
        case QType::F16:
            return 2;
        case QType::BF16:
            return 2;
        case QType::FP8_E4M3:
            return 1;
        case QType::FP8_E5M2:
            return 1;
        case QType::INT8:
            return 1;
        case QType::INT4:
            return 1;  // 2 elems/byte (caller handles packing)
        case QType::INT32:
            return 4;
        case QType::FP4_E2M1:
            return 1;  // 2 elems/byte
        default:
            return 0;
    }
}

size_t qtype_row_bytes(QType q, int64_t cols) {
    switch (q) {
        case QType::Q6_K:
            return static_cast<size_t>(cols / 256) * 210;
        case QType::Q8_0:
            return static_cast<size_t>(cols / 32) * 34;
        case QType::Q4_0:
            return static_cast<size_t>(cols / 32) * 18;
        case QType::Q8_1:
            return static_cast<size_t>(cols / 32) * 36;
        case QType::Q4_1:
            return static_cast<size_t>(cols / 32) * 20;
        case QType::Q5_0:
            return static_cast<size_t>(cols / 32) * 22;
        case QType::Q5_1:
            return static_cast<size_t>(cols / 32) * 24;
        case QType::Q2_K:
            return static_cast<size_t>(cols / 256) * 84;
        case QType::Q3_K:
            return static_cast<size_t>(cols / 256) * 110;
        case QType::Q4_K:
            return static_cast<size_t>(cols / 256) * 144;
        case QType::Q5_K:
            return static_cast<size_t>(cols / 256) * 176;
        case QType::Q8_K:
            return static_cast<size_t>(cols / 256) * 292;
        case QType::IQ4_NL:
            return static_cast<size_t>(cols / 32) * 18;
        case QType::IQ4_XS:
            return static_cast<size_t>(cols / 256) * 136;
        case QType::F16:
        case QType::BF16:
            return static_cast<size_t>(cols) * 2;
        case QType::F32:
            return static_cast<size_t>(cols) * 4;
        case QType::INT4:
        case QType::FP4_E2M1:
        case QType::MXFP4:
            return static_cast<size_t>((cols + 1) / 2);
        case QType::NVFP4:
        case QType::MXFP4_KV:
            return static_cast<size_t>((cols + 1) / 2);  // packed; scales separate
        case QType::FP8_E4M3:
        case QType::FP8_E5M2:
        case QType::INT8:
            return static_cast<size_t>(cols);
        case QType::INT32:
            return static_cast<size_t>(cols) * 4;
        default:
            return static_cast<size_t>(cols) * 2;  // safe fallback
    }
}

const char* qtype_name(QType q) {
    switch (q) {
        case QType::F32:
            return "F32";
        case QType::F16:
            return "F16";
        case QType::Q4_0:
            return "Q4_0";
        case QType::Q4_1:
            return "Q4_1";
        case QType::Q5_0:
            return "Q5_0";
        case QType::Q5_1:
            return "Q5_1";
        case QType::Q8_0:
            return "Q8_0";
        case QType::Q8_1:
            return "Q8_1";
        case QType::Q2_K:
            return "Q2_K";
        case QType::Q3_K:
            return "Q3_K";
        case QType::Q4_K:
            return "Q4_K";
        case QType::Q5_K:
            return "Q5_K";
        case QType::Q6_K:
            return "Q6_K";
        case QType::Q8_K:
            return "Q8_K";
        case QType::IQ4_NL:
            return "IQ4_NL";
        case QType::IQ4_XS:
            return "IQ4_XS";
        case QType::BF16:
            return "BF16";
        case QType::MXFP4:
            return "MXFP4";
        case QType::NONE:
            return "NONE";
        case QType::FP8_E4M3:
            return "FP8_E4M3";
        case QType::FP8_E5M2:
            return "FP8_E5M2";
        case QType::INT8:
            return "INT8";
        case QType::INT4:
            return "INT4";
        case QType::INT32:
            return "INT32";
        case QType::FP4_E2M1:
            return "FP4_E2M1";
        case QType::NVFP4:
            return "NVFP4";
        case QType::MXFP4_KV:
            return "MXFP4_KV";
    }
    return "UNKNOWN";
}

}  // namespace imp
