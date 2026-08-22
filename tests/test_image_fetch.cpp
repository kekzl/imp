// Destination classification for remote `image_url` fetching (#1610).
//
// The fetch was an SSRF primitive: host taken verbatim from an unauthenticated
// request body, redirects followed, no allowlist, no body cap, no read timeout.
// The default is now off, and this file pins the part that decides where the
// server is allowed to connect when it is on.
//
// CPU lane on purpose. Everything here is address arithmetic plus getaddrinfo
// on literals; nothing opens a socket, so CI - which has no GPU and no network
// guarantee - runs it on every pull request.

#include <gtest/gtest.h>

#include "image_fetch.h"

#include <string>

namespace {

using imp_server::classify_host;
using imp_server::classify_ip_literal;
using imp_server::DestinationVerdict;

TEST(SsrfClassifier, RejectsLoopback) {
    EXPECT_EQ(classify_ip_literal("127.0.0.1"), DestinationVerdict::Loopback);
    // The whole /8, not just .1 - 127.1 and 127.0.0.53 are the same machine.
    EXPECT_EQ(classify_ip_literal("127.255.255.254"), DestinationVerdict::Loopback);
    EXPECT_EQ(classify_ip_literal("127.0.0.53"), DestinationVerdict::Loopback);
    EXPECT_EQ(classify_ip_literal("::1"), DestinationVerdict::Loopback);
}

TEST(SsrfClassifier, RejectsCloudMetadata) {
    // The single most valuable target of an SSRF: instance credentials.
    EXPECT_EQ(classify_ip_literal("169.254.169.254"), DestinationVerdict::LinkLocal);
    EXPECT_EQ(classify_ip_literal("169.254.0.1"), DestinationVerdict::LinkLocal);
    EXPECT_EQ(classify_ip_literal("fe80::1"), DestinationVerdict::LinkLocal);
}

TEST(SsrfClassifier, RejectsPrivateRanges) {
    EXPECT_EQ(classify_ip_literal("10.0.0.1"), DestinationVerdict::PrivateRange);
    EXPECT_EQ(classify_ip_literal("172.16.0.1"), DestinationVerdict::PrivateRange);
    EXPECT_EQ(classify_ip_literal("172.31.255.255"), DestinationVerdict::PrivateRange);
    EXPECT_EQ(classify_ip_literal("192.168.1.1"), DestinationVerdict::PrivateRange);
    // The docker-compose network this repo ships lives here.
    EXPECT_EQ(classify_ip_literal("172.17.0.2"), DestinationVerdict::PrivateRange);
    // RFC6598, which is neither RFC1918 nor public and is easy to forget.
    EXPECT_EQ(classify_ip_literal("100.64.0.1"), DestinationVerdict::PrivateRange);
    EXPECT_EQ(classify_ip_literal("fd00::1"), DestinationVerdict::PrivateRange);
}

TEST(SsrfClassifier, RejectsTheBoundariesOfEachRangeCorrectly) {
    // 172.15 and 172.32 are public; only 172.16-31 is not. An off-by-one here
    // either opens the network or breaks a legitimate host.
    EXPECT_EQ(classify_ip_literal("172.15.0.1"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("172.32.0.1"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("100.63.255.255"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("100.128.0.1"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("11.0.0.1"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("126.255.255.255"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("128.0.0.1"), DestinationVerdict::Allowed);
}

TEST(SsrfClassifier, RejectsV4MappedLoopbackHoweverItIsSpelled) {
    // ::ffff:127.0.0.1 reaches 127.0.0.1. A classifier that only looks at the
    // v6 prefix calls this a public address.
    EXPECT_EQ(classify_ip_literal("::ffff:127.0.0.1"), DestinationVerdict::Loopback);
    EXPECT_EQ(classify_ip_literal("::ffff:169.254.169.254"), DestinationVerdict::LinkLocal);
    EXPECT_EQ(classify_ip_literal("::ffff:10.0.0.1"), DestinationVerdict::PrivateRange);
    // And the same addresses written as hex quads.
    EXPECT_EQ(classify_ip_literal("::ffff:7f00:1"), DestinationVerdict::Loopback);
}

TEST(SsrfClassifier, RejectsUnspecifiedAndMulticast) {
    // 0.0.0.0 routes to localhost on Linux.
    EXPECT_EQ(classify_ip_literal("0.0.0.0"), DestinationVerdict::Unspecified);
    EXPECT_EQ(classify_ip_literal("::"), DestinationVerdict::Unspecified);
    EXPECT_EQ(classify_ip_literal("224.0.0.1"), DestinationVerdict::Multicast);
    EXPECT_EQ(classify_ip_literal("255.255.255.255"), DestinationVerdict::Reserved);
}

TEST(SsrfClassifier, AllowsOrdinaryPublicAddresses) {
    // Negative control: the check must not refuse the thing it exists to allow.
    EXPECT_EQ(classify_ip_literal("8.8.8.8"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("1.1.1.1"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("93.184.216.34"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_ip_literal("2606:4700:4700::1111"), DestinationVerdict::Allowed);
}

TEST(SsrfClassifier, NonAddressesAreNotAddresses) {
    EXPECT_EQ(classify_ip_literal(""), DestinationVerdict::NotAnIpOrUnresolvable);
    EXPECT_EQ(classify_ip_literal("example.com"), DestinationVerdict::NotAnIpOrUnresolvable);
    EXPECT_EQ(classify_ip_literal("999.1.1.1"), DestinationVerdict::NotAnIpOrUnresolvable);
}

TEST(SsrfClassifier, HostFormAcceptsLiteralsAndBrackets) {
    // classify_host resolves, but a literal resolves to itself with no DNS.
    EXPECT_EQ(classify_host("127.0.0.1"), DestinationVerdict::Loopback);
    EXPECT_EQ(classify_host("[::1]"), DestinationVerdict::Loopback);
    EXPECT_EQ(classify_host("169.254.169.254"), DestinationVerdict::LinkLocal);
    EXPECT_EQ(classify_host("8.8.8.8"), DestinationVerdict::Allowed);
    EXPECT_EQ(classify_host(""), DestinationVerdict::NotAnIpOrUnresolvable);
    // "localhost" is the name every SSRF payload reaches for, and it resolves
    // without leaving the machine, so this stays hermetic.
    EXPECT_EQ(classify_host("localhost"), DestinationVerdict::Loopback);
}

// The default is what actually carries this defence: with the flag off, no
// URL reaches the network at all, whatever it names.
TEST(SsrfFetch, RemoteFetchIsOffByDefault) {
    auto r = imp_server::fetch_remote_image("http://127.0.0.1:9/probe", /*allow_remote=*/false);
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.detail.find("--allow-remote-images"), std::string::npos) << r.detail;
    EXPECT_TRUE(r.bytes.empty());
}

TEST(SsrfFetch, PrivateDestinationsAreRefusedEvenWhenEnabled) {
    // Port 9 (discard) is closed on this host; the point is that the refusal
    // happens before any connect, so the result must be the classifier's, not
    // a connection error.
    for (const char* u : {"http://127.0.0.1:9/x", "http://169.254.169.254/latest/meta-data/",
                          "http://10.0.0.1/x", "http://[::1]:9/x", "http://localhost:9/x"}) {
        auto r = imp_server::fetch_remote_image(u, /*allow_remote=*/true);
        EXPECT_FALSE(r.ok) << u;
        EXPECT_NE(r.detail.find("destination rejected"), std::string::npos) << u << ": " << r.detail;
    }
}

TEST(SsrfFetch, UserinfoDoesNotHideTheRealHost) {
    // http://allowed.example@127.0.0.1/ connects to 127.0.0.1. A parser that
    // splits on the first '/' and keeps everything before it as the host reads
    // this as "allowed.example@127.0.0.1" and hands that to the client.
    auto r = imp_server::fetch_remote_image("http://example.com@127.0.0.1:9/x", /*allow_remote=*/true);
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.detail.find("destination rejected"), std::string::npos) << r.detail;
}

TEST(SsrfFetch, NonHttpSchemesAreNotFetched) {
    auto r = imp_server::fetch_remote_image("file:///etc/passwd", /*allow_remote=*/true);
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.detail.find("scheme"), std::string::npos) << r.detail;
}

}  // namespace
