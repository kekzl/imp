#include "image_fetch.h"

#include "core/logging.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstring>

#include <httplib.h>

namespace imp_server {

namespace {

DestinationVerdict classify_v4(uint32_t addr_host_order) {
    const uint32_t a = addr_host_order;
    const uint8_t b0 = static_cast<uint8_t>(a >> 24);
    const uint8_t b1 = static_cast<uint8_t>(a >> 16);

    if (a == 0)
        return DestinationVerdict::Unspecified;
    if (b0 == 127)
        return DestinationVerdict::Loopback;
    if (b0 == 169 && b1 == 254)
        return DestinationVerdict::LinkLocal;  // includes 169.254.169.254
    if (b0 == 10)
        return DestinationVerdict::PrivateRange;
    if (b0 == 172 && b1 >= 16 && b1 <= 31)
        return DestinationVerdict::PrivateRange;
    if (b0 == 192 && b1 == 168)
        return DestinationVerdict::PrivateRange;
    if (b0 == 100 && b1 >= 64 && b1 <= 127)
        return DestinationVerdict::PrivateRange;  // RFC6598 CGNAT
    if (b0 >= 224 && b0 <= 239)
        return DestinationVerdict::Multicast;
    if (b0 >= 240)
        return DestinationVerdict::Reserved;  // 240/4, incl. 255.255.255.255
    // 192.0.0.0/24, 192.0.2.0/24, 198.18.0.0/15, 198.51.100.0/24, 203.0.113.0/24
    if (b0 == 192 && b1 == 0)
        return DestinationVerdict::Reserved;
    if (b0 == 198 && (b1 == 18 || b1 == 19 || b1 == 51))
        return DestinationVerdict::Reserved;
    if (b0 == 203 && b1 == 0)
        return DestinationVerdict::Reserved;
    return DestinationVerdict::Allowed;
}

DestinationVerdict classify_v6(const in6_addr& a) {
    const uint8_t* b = a.s6_addr;

    // An IPv4-mapped or IPv4-compatible v6 address reaches the v4 host it
    // names. ::ffff:127.0.0.1 is loopback however it is spelled.
    static const uint8_t v4mapped_prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    if (std::memcmp(b, v4mapped_prefix, 12) == 0) {
        uint32_t v4 = (static_cast<uint32_t>(b[12]) << 24) | (static_cast<uint32_t>(b[13]) << 16) |
                      (static_cast<uint32_t>(b[14]) << 8) | static_cast<uint32_t>(b[15]);
        return classify_v4(v4);
    }
    bool first12_zero = true;
    for (int i = 0; i < 12; i++)
        if (b[i] != 0)
            first12_zero = false;
    if (first12_zero) {
        uint32_t v4 = (static_cast<uint32_t>(b[12]) << 24) | (static_cast<uint32_t>(b[13]) << 16) |
                      (static_cast<uint32_t>(b[14]) << 8) | static_cast<uint32_t>(b[15]);
        if (v4 == 1)
            return DestinationVerdict::Loopback;  // ::1
        if (v4 == 0)
            return DestinationVerdict::Unspecified;
        return DestinationVerdict::Reserved;  // ::a.b.c.d, deprecated
    }

    if ((b[0] & 0xfe) == 0xfc)
        return DestinationVerdict::PrivateRange;  // fc00::/7 ULA
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80)
        return DestinationVerdict::LinkLocal;  // fe80::/10
    if (b[0] == 0xff)
        return DestinationVerdict::Multicast;
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x00 && (b[3] & 0xf0) == 0x00)
        return DestinationVerdict::Reserved;  // 2001:0::/24 teredo/benchmarking block
    return DestinationVerdict::Allowed;
}

const char* verdict_name(DestinationVerdict v) {
    switch (v) {
        case DestinationVerdict::Allowed:
            return "allowed";
        case DestinationVerdict::NotAnIpOrUnresolvable:
            return "unresolvable";
        case DestinationVerdict::Loopback:
            return "loopback";
        case DestinationVerdict::LinkLocal:
            return "link-local";
        case DestinationVerdict::PrivateRange:
            return "private";
        case DestinationVerdict::Unspecified:
            return "unspecified";
        case DestinationVerdict::Multicast:
            return "multicast";
        case DestinationVerdict::Reserved:
            return "reserved";
    }
    return "unknown";
}

}  // namespace

DestinationVerdict classify_ip_literal(const std::string& ip) {
    in_addr v4{};
    if (inet_pton(AF_INET, ip.c_str(), &v4) == 1)
        return classify_v4(ntohl(v4.s_addr));
    in6_addr v6{};
    if (inet_pton(AF_INET6, ip.c_str(), &v6) == 1)
        return classify_v6(v6);
    return DestinationVerdict::NotAnIpOrUnresolvable;
}

DestinationVerdict classify_host(const std::string& host) {
    // A bracketed v6 literal arrives as [::1]; strip before resolving.
    std::string h = host;
    if (h.size() >= 2 && h.front() == '[' && h.back() == ']')
        h = h.substr(1, h.size() - 2);
    if (h.empty())
        return DestinationVerdict::NotAnIpOrUnresolvable;

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(h.c_str(), nullptr, &hints, &res) != 0 || !res)
        return DestinationVerdict::NotAnIpOrUnresolvable;

    // ALL records must be public. One public and one loopback record is a
    // rebinding primitive, not a partial success.
    DestinationVerdict worst = DestinationVerdict::NotAnIpOrUnresolvable;
    bool any = false;
    for (addrinfo* p = res; p; p = p->ai_next) {
        DestinationVerdict v = DestinationVerdict::NotAnIpOrUnresolvable;
        if (p->ai_family == AF_INET) {
            auto* sa = reinterpret_cast<sockaddr_in*>(p->ai_addr);
            v = classify_v4(ntohl(sa->sin_addr.s_addr));
        } else if (p->ai_family == AF_INET6) {
            auto* sa = reinterpret_cast<sockaddr_in6*>(p->ai_addr);
            v = classify_v6(sa->sin6_addr);
        } else {
            continue;
        }
        any = true;
        if (v != DestinationVerdict::Allowed) {
            freeaddrinfo(res);
            return v;
        }
        worst = v;
    }
    freeaddrinfo(res);
    return any ? worst : DestinationVerdict::NotAnIpOrUnresolvable;
}

FetchResult fetch_remote_image(const std::string& url, bool allow_remote) {
    FetchResult out;
    if (!allow_remote) {
        out.detail = "remote image_url fetching is off (--allow-remote-images)";
        return out;
    }

    const bool is_https = (url.rfind("https://", 0) == 0);
    if (!is_https && url.rfind("http://", 0) != 0) {
        out.detail = "scheme is not http/https";
        return out;
    }
    std::string rest = url.substr(is_https ? 8 : 7);
    // Strip userinfo: http://allowed.example@127.0.0.1/ connects to the host
    // AFTER the '@', which is not the one a naive reader sees.
    auto at = rest.find('@');
    auto first_slash_for_at = rest.find('/');
    if (at != std::string::npos && (first_slash_for_at == std::string::npos || at < first_slash_for_at))
        rest = rest.substr(at + 1);

    auto slash = rest.find('/');
    std::string authority = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
    std::string path_str = (slash != std::string::npos) ? rest.substr(slash) : "/";

    // Split host from port, keeping a bracketed v6 literal intact.
    std::string host = authority;
    if (!authority.empty() && authority.front() == '[') {
        auto close = authority.find(']');
        if (close == std::string::npos) {
            out.detail = "malformed IPv6 authority";
            return out;
        }
        host = authority.substr(0, close + 1);
    } else {
        auto colon = authority.find(':');
        if (colon != std::string::npos)
            host = authority.substr(0, colon);
    }

    DestinationVerdict verdict = classify_host(host);
    if (verdict != DestinationVerdict::Allowed) {
        out.detail = std::string("destination rejected: ") + verdict_name(verdict);
        IMP_LOG_WARN("image_url refused: %s resolves to a %s address", host.c_str(), verdict_name(verdict));
        return out;
    }

    auto run = [&](auto& cli) {
        // Redirects OFF. Following one re-runs the whole decision on a host
        // this function never saw, and httplib offers no per-hop hook.
        cli.set_follow_location(false);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(kRemoteImageReadTimeoutSec);
        cli.set_write_timeout(kRemoteImageReadTimeoutSec);

        std::string body;
        bool too_large = false;
        auto res = cli.Get(path_str, httplib::Headers{}, [&](const char* data, size_t len) {
            if (body.size() + len > kMaxRemoteImageBytes) {
                too_large = true;
                return false;  // abort the transfer
            }
            body.append(data, len);
            return true;
        });
        if (too_large) {
            out.detail = "image body over the cap";
            return;
        }
        if (!res) {
            out.detail = "request failed";
            return;
        }
        if (res->status != 200) {
            out.detail = "status " + std::to_string(res->status);
            return;
        }
        out.bytes.assign(body.begin(), body.end());
        out.ok = !out.bytes.empty();
        if (!out.ok)
            out.detail = "empty body";
    };

    if (is_https) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(authority);
        run(cli);
#else
        out.detail = "https image_url needs an imp built with OpenSSL";
#endif
    } else {
        httplib::Client cli(authority);
        run(cli);
    }
    return out;
}

}  // namespace imp_server
