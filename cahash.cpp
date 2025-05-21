#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

// --------------------------------------------------
// Minimal SHA-256 implementation (public domain)
// --------------------------------------------------
struct SHA256_CTX {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
};
static const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define ROTRIGHT(a,b) (((a)>>(b))|((a)<<(32-(b))))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x)     (ROTRIGHT(x,2)^ROTRIGHT(x,13)^ROTRIGHT(x,22))
#define EP1(x)     (ROTRIGHT(x,6)^ROTRIGHT(x,11)^ROTRIGHT(x,25))
#define SIG0(x)    (ROTRIGHT(x,7)^ROTRIGHT(x,18)^((x)>>3))
#define SIG1(x)    (ROTRIGHT(x,17)^ROTRIGHT(x,19)^((x)>>10))

void sha256_transform(SHA256_CTX *ctx, const uint8_t d[]) {
    uint32_t m[64], a,b,c,d1,e,f,g,h,t1,t2;
    for(int i=0,j=0;i<16;++i,j+=4)
        m[i] = (d[j]<<24)|(d[j+1]<<16)|(d[j+2]<<8)|d[j+3];
    for(int i=16;i<64;++i) m[i] = SIG1(m[i-2]) + m[i-7] + SIG0(m[i-15]) + m[i-16];
    a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d1=ctx->state[3];
    e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];
    for(int i=0;i<64;++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d1+t1;
        d1=c; c=b; b=a; a=t1+t2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d1;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}
void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen=0; ctx->bitlen=0;
    ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
}
void sha256_update(SHA256_CTX *ctx, const uint8_t *d, size_t n) {
    for(size_t i=0;i<n;++i) {
        ctx->data[ctx->datalen++] = d[i];
        if(ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}
void sha256_final(SHA256_CTX *ctx, uint8_t hash[]) {
    size_t i = ctx->datalen;
    if(i < 56) {
        ctx->data[i++] = 0x80;
        while(i < 56) ctx->data[i++] = 0;
    } else {
        ctx->data[i++] = 0x80;
        while(i < 64) ctx->data[i++] = 0;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }
    ctx->bitlen += ctx->datalen * 8;
    for(int j = 7; j >= 0; --j) ctx->data[56 + (7 - j)] = (ctx->bitlen >> (j * 8)) & 0xFF;
    sha256_transform(ctx, ctx->data);
    for(i = 0; i < 32; ++i) hash[i] = (ctx->state[i/4] >> ((3 - (i % 4)) * 8)) & 0xFF;
}

// --------------------------------------------------
// CA2D + CAHash with AVX2/FMA + OpenMP + tiling
// --------------------------------------------------
class CA2D {
public:
    CA2D(int N, double αr, double βr, double c, double d, unsigned seed = 0)
      : N(N), cr(c), dr(d), αs(αr), βs(βr) {
        grid = (double*)_mm_malloc(N*N*sizeof(double), 64);
        tmp  = (double*)_mm_malloc(N*N*sizeof(double), 64);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(-0.5, 0.5);
        for(int i = 0; i < N*N; ++i) grid[i] = u(rng);
    }
    void absorb(const uint8_t* d, size_t n) {
        int M = N*N;
        for(size_t i = 0; i < n; ++i) grid[i % M] += double(d[i]) / 255.0 - 0.5;
    }
    __attribute__((target("avx2,fma")))
    void step() {
        const __m256d αv = _mm256_set1_pd(αs);
        const __m256d βv = _mm256_set1_pd(βs);
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d cv = _mm256_set1_pd(cr);
        const int tile = 16;
        #pragma omp parallel for collapse(2) schedule(static)
        for(int ti = 0; ti < N; ti += tile) {
            for(int tj = 0; tj < N; tj += tile) {
                for(int i = ti; i < ti + tile; ++i) {
                    int row = i * N;
                    for(int j = tj; j < tj + tile; j += 4) {
                        __m256d r00 = _mm256_load_pd(&grid[row + j]);
                        int im1 = ((i - 1 + N) % N) * N;
                        int ip1 = ((i + 1) % N) * N;
                        __m256d rN = _mm256_load_pd(&grid[im1 + j]);
                        __m256d rS = _mm256_load_pd(&grid[ip1 + j]);
                        __m256d rE = _mm256_loadu_pd(&grid[row + ((j + 1) % N)]);
                        __m256d rW = _mm256_loadu_pd(&grid[row + ((j - 1 + N) % N)]);
                        __m256d sum = _mm256_add_pd(_mm256_add_pd(rN, rS), _mm256_add_pd(rE, rW));
                        __m256d raw = _mm256_fmadd_pd(αv, sum, _mm256_mul_pd(βv, _mm256_mul_pd(r00, r00)));
                        __m256d absr = _mm256_andnot_pd(_mm256_set1_pd(-0.0), raw);
                        __m256d sq   = _mm256_mul_pd(absr, absr);
                        __m256d den  = _mm256_add_pd(one, _mm256_mul_pd(cv, sq));
                        __m256d nxt  = _mm256_div_pd(raw, den);
                        _mm256_store_pd(&tmp[row + j], nxt);
                    }
                }
            }
        }
        std::swap(grid, tmp);
    }
    void run(int s) { for(int i = 0; i < s; ++i) step(); }
    std::vector<double> real_flat() const { return std::vector<double>(grid, grid + N*N); }
    ~CA2D() { _mm_free(grid); _mm_free(tmp); }
private:
    int N;
    double cr, dr;
    double αs, βs;
    double *grid, *tmp;
};

std::array<uint8_t,32> ca_hash(const std::vector<uint8_t>& data) {
    CA2D ca(64, 0.3, 0.1, 0.5, 2.0, 0);
    ca.absorb(data.data(), data.size());
    ca.run(8);
    auto post = ca.real_flat();
    double mn = *std::min_element(post.begin(), post.end());
    double mx = *std::max_element(post.begin(), post.end());
    double r = mx - mn + 1e-12;
    std::vector<uint8_t> blk;
    blk.reserve(post.size());
    for(double v : post) blk.push_back(uint8_t((v - mn) / r * 255));
    SHA256_CTX ctx; sha256_init(&ctx);
    sha256_update(&ctx, blk.data(), blk.size());
    std::array<uint8_t,32> out; sha256_final(&ctx, out.data());
    return out;
}

static std::string hexstr(const uint8_t* d, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string s; s.reserve(2*n);
    for(size_t i = 0; i < n; ++i) {
        s.push_back(h[d[i] >> 4]);
        s.push_back(h[d[i] & 0xF]);
    }
    return s;
}

int main() {
    std::vector<uint8_t> msg = {'h','e','l','l','o'};
    auto dg = ca_hash(msg);
    std::cout << "CAHash(\"hello\") = " << hexstr(dg.data(), dg.size()) << "\n";
    std::vector<uint8_t> data(1024*1024);
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<uint8_t> ud(0,255);
    for(auto& b : data) b = ud(rng);
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 50; ++i) ca_hash(data);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double mb = double(data.size()) * 50 / 1048576.0;
    std::cout << "Throughput: " << (mb / secs) << " MB/s\n";
    return 0;
}
