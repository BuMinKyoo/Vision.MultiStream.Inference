// preprocess.cu
//
// Phase 4 Step 14: YOLOv8 전처리(letterbox 리사이즈 + 정규화 + HWC→CHW)를 CUDA 커널로 이전.
// 지금까지 C# YoloPreprocessor.Preprocess(byte[] bgr, w, h) 가 CPU(unsafe/Span)로 하던 일을
// 그대로 GPU 로 옮긴다. 목표는 "동일 출력, 더 빠른 처리" — 결과 텐서가 C# CPU 버전과
// 부동소수점 수준에서 일치해야 한다(bilinear/클램프/half-pixel 매핑을 그대로 복제).
//
// 경계(이번 단계):
//   - 입력 : 호스트의 BGR byte 버퍼 (width*height*3, OpenCV Mat 기본 포맷)
//   - 출력 : 호스트의 [1,3,640,640] float 텐서 (CHW, R/G/B 평면, 0~1 정규화)
//   - 내부 : H2D 업로드 → fill(114/255) → resize 커널 → D2H 다운로드
//   scale/newW/newH/padX/padY 는 C# 가 계산해 넘긴다(반올림 규칙 중복/불일치 방지).
//
// [최적화] device 버퍼(dSrc/dDst)를 프레임 간 상주시켜 매 프레임 cudaMalloc/cudaFree 를 없앤다.
//   - dDst 는 항상 3*640*640 float 로 고정 → 최초 1회만 할당.
//   - dSrc 는 프레임 해상도에 따라 크기가 다르므로 "필요 용량보다 작을 때만" 재할당(그 외 재사용).
//   버퍼는 프로세스 수명 동안 유지(종료 시 OS 회수). 여러 스트림이 같은 진입점을 호출할 수 있어
//   g_mtx 로 직렬화한다(호출자도 SerializedFrameDetector 로 직렬화하지만 방어적).
// NOTE(다음 단계): H2D/D2H 는 여전히 pageable 메모리 전송이다. pinned(고정) 호스트 버퍼 +
//   cudaMemcpyAsync + 추론까지 GPU 상주(Zero-Copy)로 전송 자체를 줄이는 게 이후 목표.

#include <cuda_runtime.h>
#include <mutex>

#define VCUDA_API extern "C" __declspec(dllexport)

namespace {

// ── 상주 device 버퍼 ──
std::mutex     g_mtx;
unsigned char* g_dSrc    = nullptr; // BGR 원본(가변 크기). g_dSrcCap 보다 크게 필요할 때만 재할당.
size_t         g_dSrcCap = 0;
float*         g_dDst    = nullptr; // 출력 텐서(고정 3*640*640). 최초 1회만 할당.

constexpr int   kSize          = 640;             // YOLOv8 고정 입력 변
constexpr int   kChannelStride = kSize * kSize;   // 한 채널(640×640) 원소 수
constexpr int   kTensorLen     = 3 * kChannelStride;
constexpr float kInv255        = 1.0f / 255.0f;
constexpr float kPadValue      = 114.0f / 255.0f; // letterbox 회색 패딩(정규화값)

// C# YoloPreprocessor.Bilinear 과 동일. byte 4이웃을 wx,wy 로 이중선형 보간.
// (v01-v00) 은 int 승격 후 float 곱 → C# 의 int 승격 결과와 동일.
__device__ __forceinline__ float bilinear(
    unsigned char v00, unsigned char v01, unsigned char v10, unsigned char v11,
    float wx, float wy)
{
    float top    = (float)v00 + ((float)v01 - (float)v00) * wx;
    float bottom = (float)v10 + ((float)v11 - (float)v10) * wx;
    return top + (bottom - top) * wy;
}

// 출력 텐서 전체를 letterbox 패딩값으로 채운다(cudaMemset 은 float 패턴 불가 → 커널로).
// C# 의 buffer.AsSpan().Fill(PadValueNormalized) 대응.
__global__ void fillKernel(float* dst, int len, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        dst[i] = value;
    }
}

// 스레드 1개 = 출력 픽셀 1개(ox,oy). 이미지 영역(newW×newH)만 담당하고
// 패딩 테두리는 fillKernel 이 미리 채운 값이 그대로 남는다.
__global__ void resizeKernel(
    const unsigned char* __restrict__ src, int width, int height,
    float* __restrict__ dst,
    int newW, int newH, int padX, int padY, float scale)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= newW || oy >= newH)
    {
        return;
    }

    int srcStride = width * 3; // BGR

    // 출력 행 oy → 원본 y (half-pixel center 매핑). C# 로직과 동일 순서/클램프.
    float srcYf = (oy + 0.5f) / scale - 0.5f;
    if (srcYf < 0.0f) { srcYf = 0.0f; }
    int y0 = (int)srcYf;
    if (y0 > height - 1) { y0 = height - 1; }
    int y1 = (y0 + 1 < height) ? y0 + 1 : y0;
    float wy = srcYf - y0;

    float srcXf = (ox + 0.5f) / scale - 0.5f;
    if (srcXf < 0.0f) { srcXf = 0.0f; }
    int x0 = (int)srcXf;
    if (x0 > width - 1) { x0 = width - 1; }
    int x1 = (x0 + 1 < width) ? x0 + 1 : x0;
    float wx = srcXf - x0;

    const unsigned char* row0 = src + y0 * srcStride;
    const unsigned char* row1 = src + y1 * srcStride;
    int c0 = x0 * 3; // 좌측 픽셀 BGR 시작
    int c1 = x1 * 3; // 우측 픽셀 BGR 시작
    const unsigned char* p00 = row0 + c0;
    const unsigned char* p01 = row0 + c1;
    const unsigned char* p10 = row1 + c0;
    const unsigned char* p11 = row1 + c1;

    // BGR 순서로 보간 → R/G/B 평면에 분배(B↔R 스왑) + 정규화.
    float b = bilinear(p00[0], p01[0], p10[0], p11[0], wx, wy);
    float g = bilinear(p00[1], p01[1], p10[1], p11[1], wx, wy);
    float r = bilinear(p00[2], p01[2], p10[2], p11[2], wx, wy);

    int di = (oy + padY) * kSize + (padX + ox);
    dst[di]                     = r * kInv255; // 채널0 = R
    dst[kChannelStride + di]    = g * kInv255; // 채널1 = G
    dst[2 * kChannelStride + di] = b * kInv255; // 채널2 = B
}

} // namespace

// BGR 호스트 버퍼 → [1,3,640,640] float 텐서(호스트).
//   bgr       : width*height*3 byte (B-G-R, row-major)
//   outTensor : 3*640*640 float. outLen 은 원소 수(=1228800) 검증용.
//   newW/newH/padX/padY/scale : C# 가 계산해 넘긴 letterbox 파라미터.
// 반환: 0=성공, 음수=실패(단계별 코드).
VCUDA_API int vcuda_preprocess(
    const unsigned char* bgr, int width, int height,
    int newW, int newH, int padX, int padY, float scale,
    float* outTensor, int outLen)
{
    if (bgr == nullptr || outTensor == nullptr)
    {
        return -1;
    }
    if (width <= 0 || height <= 0 || newW <= 0 || newH <= 0)
    {
        return -2;
    }
    if (outLen != kTensorLen)
    {
        return -3; // 호출자 버퍼 길이가 [1,3,640,640] 와 불일치
    }

    const size_t srcBytes = (size_t)width * height * 3;
    const size_t dstBytes = (size_t)kTensorLen * sizeof(float);

    std::lock_guard<std::mutex> lock(g_mtx);

    // 상주 출력 버퍼(고정 크기): 최초 1회만 할당.
    if (g_dDst == nullptr)
    {
        if (cudaMalloc(&g_dDst, dstBytes) != cudaSuccess) { return -4; }
    }
    // 상주 입력 버퍼(가변 크기): 현재 용량이 모자랄 때만 재할당.
    if (g_dSrcCap < srcBytes)
    {
        if (g_dSrc != nullptr) { cudaFree(g_dSrc); g_dSrc = nullptr; }
        if (cudaMalloc(&g_dSrc, srcBytes) != cudaSuccess) { g_dSrcCap = 0; return -4; }
        g_dSrcCap = srcBytes;
    }

    if (cudaMemcpy(g_dSrc, bgr, srcBytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        return -5;
    }

    // (1) 전체를 letterbox 패딩값으로 채움
    {
        int threads = 256;
        int blocks  = (kTensorLen + threads - 1) / threads;
        fillKernel<<<blocks, threads>>>(g_dDst, kTensorLen, kPadValue);
    }

    // (2) 이미지 영역만 리사이즈+정규화+CHW 로 덮어씀
    {
        dim3 block(16, 16);
        dim3 grid((newW + block.x - 1) / block.x, (newH + block.y - 1) / block.y);
        resizeKernel<<<grid, block>>>(g_dSrc, width, height, g_dDst, newW, newH, padX, padY, scale);
    }

    if (cudaGetLastError() != cudaSuccess)      { return -6; }
    if (cudaDeviceSynchronize() != cudaSuccess) { return -6; }

    if (cudaMemcpy(outTensor, g_dDst, dstBytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return -7;
    }

    return 0;
}
