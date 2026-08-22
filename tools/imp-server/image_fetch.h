#pragma once

// Remote `image_url` fetching, with the destination checks that make it safe to
// hand a request-supplied URL to an HTTP client (#1610).
//
// The fetch itself is opt-in (`--allow-remote-images`, default off). A data URI
// is the path every real client uses and needs none of this; a remote URL turns
// the server into an HTTP client that an unauthenticated caller aims, which on
// this host means loopback, the container network, and the cloud metadata
// endpoint are all one request body away.

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

// Classify one textual IP address (v4 or v6, no host name, no port).
// Pure and allocation-free apart from the parse: this is the part that gets
// unit-tested, because the interesting cases are the ones nobody reaches by
// accident (0x7f.1, ::ffff:127.0.0.1, 100.64/10).
DestinationVerdict classify_ip_literal(const std::string& ip);

// Resolve `host` and classify every address it maps to. A host is allowed only
// when it resolves to at least one address and ALL of them are public: a name
// with one public and one loopback record is a rebinding primitive, not a
// partial success.
//
// NOTE the residual risk, which is not fixable at this layer: the check and the
// connection are two separate resolutions, so a name whose records change in
// between still reaches a private address (DNS rebinding). Closing that needs a
// connect-time callback, which httplib does not expose. The opt-in default is
// what actually carries this.
DestinationVerdict classify_host(const std::string& host);

struct FetchResult {
    bool ok = false;
    std::vector<uint8_t> bytes;
    // For the log only. Never returned to the client, see the uniform-error
    // note above.
    std::string detail;
};

// Fetch an http/https image URL with every bound this header describes:
// destination classified, redirects NOT followed (each hop would need its own
// check and httplib gives no hook for one), body capped, read timeout set.
//
// `allow_remote` false returns ok=false without touching the network.
FetchResult fetch_remote_image(const std::string& url, bool allow_remote);

}  // namespace imp_server
