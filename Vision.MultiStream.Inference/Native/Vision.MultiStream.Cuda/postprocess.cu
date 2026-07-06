// postprocess.cu
//
// Phase 4 Step 14: YOLOv8 후처리 앞단(ParseOutput)을 CUDA 커널로 이전.
// C# YoloPostprocessor.ParseOutput 이 CPU 에서 하던 "앵커 필터 + bbox 디코드 + letterbox 역변환"을
// GPU 로 옮긴다. 이 부분이 numAnchors(8400) × numClasses(80) ≈ 67만 회 접근으로 후처리의 큰 덩어리다.
//
// 경계(이번 단계):
//   - 입력 : 추론 raw 출력 [1, numChannels, numAnchors] flat(channel-major). C# _outputBuffer(호스트).
//   - 출력 : 신뢰도 임계값을 넘은 "후보" 배열(박스 xywh + 클래스 + 점수)과 그 개수.
//   - NMS(겹치는 박스 억제)는 후보 수가 적어(보통 수십~수백) C# ApplyNms 를 그대로 재사용한다.
//     → GPU 는 대량 병렬 필터/디코드만 담당, 소수 후보 정렬/억제는 CPU. (실용적 분업)
//
// 전처리(preprocess.cu)와 동일하게 device 버퍼를 상주시켜 프레임당 cudaMalloc 을 없앤다.

#include <cuda_runtime.h>
#include <mutex>

#define VCUDA_API extern "C" __declspec(dllexport)

namespace {

// ── 상주 device 버퍼 ──
std::mutex g_mtx;
float* g_dOutput   = nullptr;  // 추론 raw 출력(numChannels*numAnchors). 크기 부족 시만 재할당.
size_t g_dOutputCap = 0;
float* g_dBoxes    = nullptr;  // 후보 박스 xywh (maxOut*4)
int*   g_dClasses  = nullptr;  // 후보 클래스 id (maxOut)
float* g_dScores   = nullptr;  // 후보 점수 (maxOut)
int*   g_dCount    = nullptr;  // atomic 후보 카운터 (1)
int    g_maxOut    = 0;        // 현재 할당된 후보 버퍼 용량

// 앵커 1개 = 스레드 1개. C# ParseOutput 로직을 그대로 복제:
//   80개 클래스 중 max score 가 임계값 초과면 후보로 채택 → cxcywh 디코드 → letterbox 역변환 → 클램프.
//   살아남으면 atomicAdd 로 출력 슬롯을 하나 확보해 기록(순서는 비결정적이지만 이후 NMS 가 점수정렬함).
__global__ void parseKernel(
    const float* __restrict__ output, int numChannels, int numAnchors,
    float scale, int padX, int padY, int origW, int origH, float confThreshold,
    float* outBoxes, int* outClasses, float* outScores, int maxOut, int* outCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAnchors)
    {
        return;
    }

    int numClasses = numChannels - 4;

    // 80개 클래스 중 최고 점수 클래스 선택(임계값 미만이면 bestClass = -1).
    int   bestClass = -1;
    float bestScore = confThreshold;
    for (int c = 0; c < numClasses; c++)
    {
        float score = output[(4 + c) * numAnchors + i];
        if (score > bestScore)
        {
            bestScore = score;
            bestClass = c;
        }
    }
    if (bestClass < 0)
    {
        return;
    }

    // 640 letterbox 기준 (cx, cy, w, h) → 원본 이미지 좌표로 역변환.
    float cx = output[i];
    float cy = output[numAnchors + i];
    float w  = output[2 * numAnchors + i];
    float h  = output[3 * numAnchors + i];

    float x  = (cx - w * 0.5f - padX) / scale;
    float y  = (cy - h * 0.5f - padY) / scale;
    float bw = w / scale;
    float bh = h / scale;

    // 이미지 경계로 클램프(C# Math.Clamp / Math.Min 과 동일).
    x = fminf(fmaxf(x, 0.0f), (float)origW);
    y = fminf(fmaxf(y, 0.0f), (float)origH);
    bw = fminf(bw, (float)origW - x);
    bh = fminf(bh, (float)origH - y);
    if (bw <= 0.0f || bh <= 0.0f)
    {
        return;
    }

    // 출력 슬롯 확보. maxOut 초과분은 버린다(카운터는 계속 증가시켜 총량 파악 가능).
    int slot = atomicAdd(outCount, 1);
    if (slot < maxOut)
    {
        outBoxes[slot * 4 + 0] = x;
        outBoxes[slot * 4 + 1] = y;
        outBoxes[slot * 4 + 2] = bw;
        outBoxes[slot * 4 + 3] = bh;
        outClasses[slot] = bestClass;
        outScores[slot]  = bestScore;
    }
}

} // namespace

// 추론 raw 출력 → 신뢰도 필터/디코드/역변환한 후보 배열(호스트).
//   output    : numChannels*numAnchors float (channel-major, [1,C,A] flat)
//   scale/padX/padY/origW/origH : letterbox 역변환 파라미터(전처리 때와 동일값)
//   outBoxes  : maxOut*4 float (x,y,w,h), outClasses/outScores : maxOut
//   outCount  : [out] 임계값을 넘은 총 후보 수(maxOut 초과 가능 → 호출자가 min 처리)
// 반환: 0=성공, 음수=실패.
VCUDA_API int vcuda_postprocess(
    const float* output, int numChannels, int numAnchors,
    float scale, int padX, int padY, int origW, int origH, float confThreshold,
    float* outBoxes, int* outClasses, float* outScores, int maxOut, int* outCount)
{
    if (output == nullptr || outBoxes == nullptr || outClasses == nullptr ||
        outScores == nullptr || outCount == nullptr)
    {
        return -1;
    }
    if (numChannels <= 4 || numAnchors <= 0 || maxOut <= 0)
    {
        return -2;
    }

    const size_t outElems  = (size_t)numChannels * numAnchors;
    const size_t outBytes  = outElems * sizeof(float);

    std::lock_guard<std::mutex> lock(g_mtx);

    // 상주 입력 버퍼: 용량 부족할 때만 재할당.
    if (g_dOutputCap < outBytes)
    {
        if (g_dOutput != nullptr) { cudaFree(g_dOutput); g_dOutput = nullptr; }
        if (cudaMalloc(&g_dOutput, outBytes) != cudaSuccess) { g_dOutputCap = 0; return -3; }
        g_dOutputCap = outBytes;
    }
    // 상주 후보 버퍼: maxOut 이 커지면 재할당.
    if (g_maxOut < maxOut)
    {
        if (g_dBoxes   != nullptr) { cudaFree(g_dBoxes); }
        if (g_dClasses != nullptr) { cudaFree(g_dClasses); }
        if (g_dScores  != nullptr) { cudaFree(g_dScores); }
        g_dBoxes = nullptr; g_dClasses = nullptr; g_dScores = nullptr;
        bool ok = cudaMalloc(&g_dBoxes,   (size_t)maxOut * 4 * sizeof(float)) == cudaSuccess
               && cudaMalloc(&g_dClasses, (size_t)maxOut * sizeof(int))       == cudaSuccess
               && cudaMalloc(&g_dScores,  (size_t)maxOut * sizeof(float))     == cudaSuccess;
        if (!ok) { g_maxOut = 0; return -3; }
        g_maxOut = maxOut;
    }
    if (g_dCount == nullptr)
    {
        if (cudaMalloc(&g_dCount, sizeof(int)) != cudaSuccess) { return -3; }
    }

    if (cudaMemcpy(g_dOutput, output, outBytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        return -4;
    }
    if (cudaMemset(g_dCount, 0, sizeof(int)) != cudaSuccess)
    {
        return -4;
    }

    {
        int threads = 256;
        int blocks  = (numAnchors + threads - 1) / threads;
        parseKernel<<<blocks, threads>>>(
            g_dOutput, numChannels, numAnchors,
            scale, padX, padY, origW, origH, confThreshold,
            g_dBoxes, g_dClasses, g_dScores, maxOut, g_dCount);
    }

    if (cudaGetLastError() != cudaSuccess)      { return -5; }
    if (cudaDeviceSynchronize() != cudaSuccess) { return -5; }

    // 후보 개수 먼저 받고, 실제 기록된 수(min(count,maxOut))만큼만 D2H.
    int count = 0;
    if (cudaMemcpy(&count, g_dCount, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return -6;
    }
    int copyN = count < maxOut ? count : maxOut;
    if (copyN > 0)
    {
        if (cudaMemcpy(outBoxes,   g_dBoxes,   (size_t)copyN * 4 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) { return -6; }
        if (cudaMemcpy(outClasses, g_dClasses, (size_t)copyN * sizeof(int),       cudaMemcpyDeviceToHost) != cudaSuccess) { return -6; }
        if (cudaMemcpy(outScores,  g_dScores,  (size_t)copyN * sizeof(float),     cudaMemcpyDeviceToHost) != cudaSuccess) { return -6; }
    }

    *outCount = count;
    return 0;
}
