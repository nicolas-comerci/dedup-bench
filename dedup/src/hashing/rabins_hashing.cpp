#include "rabins_hashing.hpp"

#include <cstdlib>
#include <cstring>

Rabins_Hashing::Rabins_Hashing() {
    poly = FINGERPRINT_PT;
    calcT();
}

void Rabins_Hashing::init(int window_size) {

    this->window_size = window_size;
    circbuf = static_cast<unsigned char *>(
        std::malloc(window_size * sizeof(unsigned char)));
    fingerprint = 0;
    circbuf_pos = -1;
    std::memset(circbuf, 0, window_size * sizeof(unsigned char));
}

uint64_t Rabins_Hashing::polymod(uint64_t nh, uint64_t nl, uint64_t d) {
    int i;
    int k = fls64(d) - 1;
    d <<= 63 - k;

    if (nh) {
        if (nh & MSB64) nh ^= d;  // XXX unreachable? (on 32 bit platform?)
        for (i = 62; i >= 0; i--)
            if (nh & ((uint64_t)1) << i) {
                nh ^= d >> (63 - i);
                nl ^= d << (i + 1);
            }
    }
    for (i = 63; i >= k; i--) {
        if (nl & INT64(1) << i) nl ^= d >> (63 - i);
    }
    return nl;
}

void Rabins_Hashing::polymult(uint64_t *php, uint64_t *plp, uint64_t x,
                              uint64_t y) {
    int i;
    uint64_t ph = 0, pl = 0;
    if (x & 1) pl = y;
    for (i = 1; i < 64; i++)
        if (x & (INT64(1) << i)) {
            ph ^= y >> (64 - i);
            pl ^= y << i;
        }
    if (php) *php = ph;
    if (plp) *plp = pl;
}

uint64_t Rabins_Hashing::polymmult(uint64_t x, uint64_t y, uint64_t d) {
    uint64_t h, l;
    polymult(&h, &l, x, y);
    return polymod(h, l, d);
}

void Rabins_Hashing::calcT() {

    unsigned int i;
    int xshift = fls64(poly) - 1;
    shift = xshift - 8;

    uint64_t T1 = polymod(0, INT64(1) << xshift, poly);
    for (i = 0; i < 256; i++) {
        T[i] = polymmult(i, T1, poly) | ((uint64_t)i << xshift);
    }

    uint64_t sizeshift = 1;
    for (i = 1; i < window_size; i++) {
        sizeshift = append8(sizeshift, 0);
    }

    for (i = 0; i < 256; i++) {
        U[i] = polymmult(i, sizeshift, poly);
    }
}

uint64_t Rabins_Hashing::slide8(unsigned char m) {

    circbuf_pos++;
    if (circbuf_pos >= window_size) {
        circbuf_pos = 0;
    }
    unsigned char om = circbuf[circbuf_pos];
    circbuf[circbuf_pos] = m;
    return fingerprint = append8(fingerprint ^ U[om], m);
}

uint64_t Rabins_Hashing::append8(uint64_t p, unsigned char m) {

    return ((p << 8) | m) ^ T[p >> shift];
}
