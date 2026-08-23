// tools/common/json_out.h - the writer behind `--json` (#1583).
//
// The escaping is what makes the contract real: `--prompt --json` puts model
// output into a JSON string, and model output contains quotes, backslashes and
// newlines. A writer that emits those raw produces a document jq rejects,
// which is the failure the mode exists to remove.

#include <gtest/gtest.h>

#include "common/json_out.h"

using imp_tools::JsonOut;

TEST(JsonOutTest, EmptyObject) { EXPECT_EQ(JsonOut().str(), "{}"); }

TEST(JsonOutTest, KeysAreCommaSeparatedInInsertionOrder) {
    JsonOut j;
    j.str("mode", "bench").intg("reps", 3);
    EXPECT_EQ(j.str(), "{\"mode\":\"bench\",\"reps\":3}");
}

TEST(JsonOutTest, NumbersCarryTheRequestedPrecision) {
    JsonOut j;
    j.num("a", 1.0 / 3.0, 2).num("b", 1.0 / 3.0, 4);
    EXPECT_EQ(j.str(), "{\"a\":0.33,\"b\":0.3333}");
}

TEST(JsonOutTest, BooleansAreUnquoted) {
    JsonOut j;
    j.boolean("measured", true).boolean("skipped", false);
    EXPECT_EQ(j.str(), "{\"measured\":true,\"skipped\":false}");
}

TEST(JsonOutTest, EscapesQuoteBackslashAndControlChars) {
    JsonOut j;
    j.str("text", "he said \"hi\"\\ \n\t\r");
    EXPECT_EQ(j.str(), "{\"text\":\"he said \\\"hi\\\"\\\\ \\n\\t\\r\"}");
}

TEST(JsonOutTest, EscapesOtherControlCharsAsUnicode) {
    JsonOut j;
    j.str("text", std::string("a\x01\x1f", 3));
    EXPECT_EQ(j.str(), "{\"text\":\"a\\u0001\\u001f\"}");
}

TEST(JsonOutTest, PassesUtf8Through) {
    // Multi-byte UTF-8 is valid inside a JSON string and must NOT be escaped
    // byte-by-byte: Ã¤ would be two characters, not "ä".
    JsonOut j;
    j.str("text", "gr\xc3\xbc\xc3\x9fe \xe4\xb8\xad");
    EXPECT_EQ(j.str(), "{\"text\":\"gr\xc3\xbc\xc3\x9fe \xe4\xb8\xad\"}");
}

TEST(JsonOutTest, KeysAreEscapedToo) {
    JsonOut j;
    j.intg("a\"b", 1);
    EXPECT_EQ(j.str(), "{\"a\\\"b\":1}");
}

TEST(JsonOutTest, NestedObjectIsEmbeddedVerbatim) {
    JsonOut inner;
    inner.intg("x", 1);
    JsonOut outer;
    outer.str("name", "n").obj("detail", inner);
    EXPECT_EQ(outer.str(), "{\"name\":\"n\",\"detail\":{\"x\":1}}");
}
