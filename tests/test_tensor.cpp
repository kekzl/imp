#include <gtest/gtest.h>
#include "core/tensor.h"

namespace imp {
namespace {

TEST(TensorTest, DefaultConstruction) {
    Tensor t;
    EXPECT_EQ(t.data, nullptr);
    EXPECT_EQ(t.ndim, 0);
    EXPECT_EQ(t.qtype, QType::NONE);
    EXPECT_EQ(t.scales, nullptr);
    EXPECT_EQ(t.tensor_scale, 1.0f);  // default = multiplicative identity
    EXPECT_FALSE(t.on_device);
}

TEST(TensorTest, Numel) {
    int64_t shape[] = {2, 3, 4};
    Tensor t(nullptr, QType::F32, 3, shape, false);
    EXPECT_EQ(t.numel(), 24);
}

TEST(TensorTest, Nbytes) {
    int64_t shape[] = {2, 3, 4};
    Tensor t(nullptr, QType::F32, 3, shape, false);
    EXPECT_EQ(t.nbytes(), 24 * sizeof(float));
}

TEST(TensorTest, NbytesFP16) {
    int64_t shape[] = {8, 16};
    Tensor t(nullptr, QType::F16, 2, shape, false);
    EXPECT_EQ(t.nbytes(), 8 * 16 * 2);  // FP16 = 2 bytes
}

TEST(TensorTest, Strides) {
    int64_t shape[] = {2, 3, 4};
    Tensor t(nullptr, QType::F32, 3, shape, false);
    t.compute_strides();
    EXPECT_EQ(t.stride[0], 12);  // 3 * 4
    EXPECT_EQ(t.stride[1], 4);   // 4
    EXPECT_EQ(t.stride[2], 1);   // 1
}

TEST(TensorTest, IsContiguous) {
    int64_t shape[] = {2, 3, 4};
    Tensor t(nullptr, QType::F32, 3, shape, false);
    t.compute_strides();
    EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, Reshape) {
    int64_t shape[] = {2, 3, 4};
    Tensor t(nullptr, QType::F32, 3, shape, false);
    t.compute_strides();

    int64_t new_shape[] = {6, 4};
    Tensor reshaped = t.reshape(2, new_shape);
    EXPECT_EQ(reshaped.ndim, 2);
    EXPECT_EQ(reshaped.shape[0], 6);
    EXPECT_EQ(reshaped.shape[1], 4);
    EXPECT_EQ(reshaped.numel(), 24);
}

TEST(TensorTest, ToString) {
    int64_t shape[] = {2, 3};
    Tensor t(nullptr, QType::F16, 2, shape, true);
    std::string s = t.to_string();
    EXPECT_FALSE(s.empty());
    // Should contain shape and dtype info
    EXPECT_NE(s.find("2"), std::string::npos);
    EXPECT_NE(s.find("3"), std::string::npos);
}

TEST(TensorTest, DTypeSizeAndName) {
    EXPECT_EQ(dtype_size(QType::F32), 4u);
    EXPECT_EQ(dtype_size(QType::F16), 2u);
    EXPECT_EQ(dtype_size(QType::BF16), 2u);
    EXPECT_EQ(dtype_size(QType::INT8), 1u);
    EXPECT_EQ(dtype_size(QType::INT32), 4u);

    EXPECT_STREQ(dtype_name(QType::F32), "F32");
    EXPECT_STREQ(dtype_name(QType::F16), "F16");
}

TEST(TensorTest, ScalarTensor) {
    int64_t shape[] = {1};
    Tensor t(nullptr, QType::F32, 1, shape, false);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.nbytes(), sizeof(float));
}

}  // namespace
}  // namespace imp
