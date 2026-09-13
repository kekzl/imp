#pragma once

// Remote image_url fetch with destination checks that make handing a request-supplied URL to an
// HTTP client safe (#1610). Opt-in only (--allow-remote-images, default off): a remote URL turns
// the server into an HTTP client an unauthenticated caller aims at loopback, the container
// network, or the cloud metadata endpoint.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp_server {

// Largest image body accepted from a remote URL. The peer chooses the size
// otherwise, and the buffer is host RAM.
constexpr size_t kMaxRemoteImageBytes = 32ull * 1024 * 1024;

// Read timeout for the fetch. A connection timeout alone does not bound a
// slow-drip server, which holds the worker thread for as long as it likes.
constexpr int kRemoteImageReadTimeoutSec = 10;

// Why a destination was rejected. Kept separate from the caller-visible error
// string on purpose: the caller returns one uniform message, so the endpoint
// cannot be used to tell "port open" from "connection refused".
enum class DestinationVerdict {
    Allowed,
    NotAnIpOrUnresolvable,
    Loopback,      // 127.0.0.0/8, ::1
    LinkLocal,     // 169.254.0.0/16 (cloud metadata), fe80::/10
    PrivateRange,  // RFC1918, RFC4193 ULA, RFC6598 CGNAT
    Unspecified,   // 0.0.0.0, ::
    Multicast,
    Reserved,  // everything else IANA does not route publicly
};

// classify_ip_literal: pure, allocation-free IP classification (v4/v6 literal only, no host/port).
// Unit-tested for the accidental-bypass cases: 0x7f.1, ::ffff:127.0.0.1, 100.64/10.
DestinationVerdict classify_ip_literal(const std::string& ip);

// classify_host: a host is allowed only when ALL resolved addresses are public - one public +
// one loopback record is a rebinding primitive, not a partial success. Residual risk: check and
// connect are separate DNS resolutions (rebinding window); httplib has no connect-time hook to close it.
DestinationVerdict classify_host(const std::string& host);

struct FetchResult {
    bool ok = false;
    std::vector<uint8_t> bytes;
    // For the log only. Never returned to the client, see the uniform-error
    // note above.
    std::string detail;
};

// fetch_remote_image: destination classified, redirects NOT followed (no per-hop check hook in
// httplib), body capped, read timeout set. allow_remote=false returns ok=false, no network touched.
FetchResult fetch_remote_image(const std::string& url, bool allow_remote);

}  // namespace imp_server
