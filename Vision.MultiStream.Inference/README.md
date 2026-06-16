# Vision.MultiStream.Inference

WPF (.NET 10) + ONNX Runtime 기반 **다중 RTSP 스트림 실시간 객체 검출** 프로젝트.
카메라를 여러 개 등록해 동시에 스트리밍하고, YOLOv8로 객체를 검출해 화면에 박스를 그린다.
**오디오 재생**(카메라 음성 듣기/끄기)과, 사람이 검출되면 로컬 **VLM(Ollama)으로 장면을 한 문장 묘사**하는 기능도 함께 지원한다.
표시는 스트림별 개별 렌더링 또는 **GPU 컴포지터(D3D11 → 단일 D3DImage)** 경로를 선택할 수 있고, 하드웨어 디코딩(D3D11VA)도 지원한다.

---

## 0. 성능 최적화 노트 (공부용)

> 이 프로젝트를 하면서 "왜 이렇게 했는가 / 핵심 코드 한두 줄은 무엇인가"를 정리한 메모.
> 다중 스트림 실시간 추론은 **프레임당 비용 × N스트림 × FPS** 라 작은 할당·복사 하나도 누적되면 GC 스파이크나 GPU 경합으로 영상이 끊긴다. 그래서 "프레임당 새 할당 0, 복사 0, CPU↔GPU 왕복 0" 을 목표로 잡았다.
>
> 노트는 두 묶음으로 나눈다. **A. 일반**(스트리밍·디코딩·표시·메모리·interop — 추론이 없어도 의미 있는 것)과 **B. 추론**(YOLO/VLM 모델 실행에 특화된 것). 둘은 서로 다른 관심사다.

### A. 일반 최적화 (스트리밍 · 디코딩 · 표시 · 메모리 · interop)

#### 0-1. 버퍼 풀로 GC 스파이크 제거 (`ArrayPool`)

- **문제**: 전처리 텐서(~4.9MB)·BGR 프레임(~5.9MB) 같은 큰 버퍼를 프레임당 `new` 하면 LOH(Large Object Heap)에 쌓여 Gen2 GC가 자주 돌고, 그때마다 전 스트림이 멈춘다.
- **해결**: 큰 버퍼는 `ArrayPool<T>.Shared`에서 빌려 쓰고 사용이 끝나면 반납. 프레임당 관리 힙 할당이 0이 된다.
- **코드**:
  - BGR 프레임 풀: `Services/Rtsp/RtspFrame.cs:84` — `ArrayPool<byte>.Shared.Return(...)` (`clearArray=false` — 다음 사용자가 `sws_scale` 결과로 전체를 덮어쓰므로 0으로 지우는 비용도 생략)
  - 전처리 텐서 풀(추론 경로)도 같은 패턴: 대여 `Services/Yolo/YoloPreprocessor.cs:57`, 반납 `Services/Yolo/LetterboxResult.cs:33` (→ B 그룹 0-8·0-10에서 다시 다룸)

#### 0-2. GPU VRAM 안에서 끝내기 (CPU 다운로드 회피)

- **문제**: HW 디코딩(D3D11VA) 결과를 화면에 그리려고 매번 CPU로 내려받아 색변환하면 PCIe 왕복 + CPU 부하가 크다.
- **해결**: HW 디코드 NV12 텍스처를 CPU로 내리지 않고 **GPU 안에서** 컴포지터 슬롯 텍스처로 복사해 셰이더로 바로 샘플링. D3D11 합성 결과는 같은 VRAM을 가리키는 D3D9 공유 텍스처로 열어 WPF D3DImage에 present. (전체 흐름은 §4 참고)
- **코드**:
  - HW 프레임을 CPU 거치지 않고 컴포지터로 핸드오프: `Services/Rtsp/Pipeline/VideoRenderer.cs:113`
  - GPU→GPU 복사(VRAM 내부, 다운로드 0): `Services/Direct3D/StreamCompositor.cs:423` — `_context.CopySubresourceRegion(slot.Nv12Tex!, ..., srcTex, frame.ArrayIndex)`
  - D3D11↔D3D9 공유(같은 VRAM): `Services/Direct3D/StreamCompositor.cs:670` 부근
  - 컴포지터는 프레임당 D3DImage present를 **1회**만(UI 스레드 부담↓), 풀스크린 삼각형은 `SV_VertexID`로 생성해 정점 버퍼도 안 쓴다: `Services/Direct3D/StreamCompositor.cs:44-52`

#### 0-3. 멀티스테이지 파이프라인 — 큐로 스레드를 쪼개 단일 스레드 한계 돌파

- **문제**: 수신(demux) → 디코딩 → 색변환/페이싱(render) → (오디오) 를 **한 스레드에서 순차로** 돌리면, 한 단계가 느려지면 전체가 막히고 멀티코어를 못 쓴다. 스트림 N개면 한계가 더 빨리 온다.
- **해결**: 역할별 단계를 **각각 별도 스레드**로 띄우고, 단계 사이는 **바운디드 큐**로만 주고받는 생산자-소비자 파이프라인으로 분리. 단계들이 서로 다른 코어에서 동시에 돌아 처리량이 올라가고, 한 단계의 지연이 큐가 흡수한다.
- **구조**: `[RtspDemuxer] → 패킷큐 → [VideoDecoder] → 프레임큐 → [VideoRenderer]` (+오디오 디코더/렌더러). 단계 간 핸드오프는 **데이터 복사가 아니라 `av_frame_clone`(refcount만 ↑)** 인 얕은 복사다.
- **코드**:
  - 파사드/단계 조립: `Services/Rtsp/RtspFrameSource.cs:13-21`(파이프라인 주석), 단계 기동 순서 `:234-239`
  - 단계 클래스: `Services/Rtsp/Pipeline/`(`RtspDemuxer` / `VideoDecoder` / `VideoRenderer` / `AudioDecoder` / `AudioRenderer`)
  - 얕은 핸드오프: `Services/Rtsp/Pipeline/VideoDecoder.cs:204` — `av_frame_clone`
  - **끈 스트림은 비용 0**: 오디오/비디오가 꺼져 있으면 해당 단계 스레드·큐를 아예 안 만든다 — `RtspFrameSource.cs:184-209` (큐 정책·페이싱은 0-4 참고)

#### 0-4. 최신 프레임 우선 + 백프레셔 (지연 최소화)

- **문제**: 소비자(추론·표시)가 느리면 프레임이 밀린다. 무작정 쌓으면 지연이 커지고, `DropOldest`로 버리면 그 프레임의 풀 버퍼가 반납되지 않아 샌다.
- **해결**: 전달 채널 용량을 **1**로 두고(최신 우선), 밀어내는 프레임은 직접 꺼내 `Dispose`(풀 반납)한 뒤 새 프레임을 넣는다. 디코드 단계 큐(0-3)는 bounded `BlockingCollection`이라 가득 차면 생산자가 대기(backpressure)하고, 페이싱은 `MediaClock`이 맡는다.
- **코드**:
  - 용량 1 + `Wait` 모드: `Services/Rtsp/RtspFrameSource.cs:60-65`
  - 수동 evict + 반납: `Services/Rtsp/RtspFrameSource.cs:319-336` (`PublishInferenceFrame`)
  - bounded 큐 backpressure 주석: `Services/Rtsp/RtspFrameSource.cs:158-166`

#### 0-5. UI 스레드 디스패치 합치기 (coalescing)

- **문제**: 디코드 스레드가 초당 수십 프레임을 그릴 때마다 `Dispatcher.BeginInvoke`를 던지면 UI 메시지 큐가 밀려, 영상은 끊기고 버튼 클릭도 늦어진다(스트림 N개면 폭증).
- **해결**: 최신 프레임은 용량 1 채널에 넣되, **이미 예약된 UI 펌프가 없을 때만** 하나를 예약(`Interlocked.CompareExchange`)한다. UI 콜백은 큐에 쌓인 걸 **한 번에 비운다**(drain). 프레임이 아무리 자주 와도 UI 스레드에는 "처리 중이면 1건"만 걸린다.
- **코드**: `ViewModels/StreamItemViewModel.cs`
  - 예약 게이트: `:1103-1105` — `if (Interlocked.CompareExchange(ref _displayPumpScheduled, 1, 0) == 0) BeginInvoke(DrainDisplayFrames, DispatcherPriority.Render)`
  - 드레인 + 재무장: `:1113-1138` (`DrainDisplayFrames`)
  - YUV 표시 경로도 동일 패턴: `:1040-1044`, `:1050-1084`

#### 0-6. P/Invoke (네이티브 경계)

- FFmpeg(`FFmpeg.AutoGen`)와 자작 C++ 추론 DLL을 P/Invoke로 호출한다. 관리/네이티브 경계에서 **복사 없이 포인터만** 넘기는 게 핵심(B 그룹 0-10 참고).
- **코드**:
  - FFmpeg 네이티브 DLL 검색 경로 등록: `Services/Rtsp/FFmpeg/FFmpegLibraryLoader.cs`
  - 디코드 결과 복사가 불가피한 지점(`sws_scale` 후)만 `Marshal.Copy`로 최소 복사: `Services/Rtsp/Pipeline/VideoRenderer.cs:266`
  - C++ 추론 DLL 선언(추론 경로): `Services/Yolo/NativeYoloEngine.cs:23-33` — `[DllImport("vision_infer")] static extern unsafe int vinfer_infer(IntPtr handle, float* input, float* output, int outputLen);`

#### 0-7. (심화) 왜 `fixed`가 필요한가 — 스택 / 힙 / GC

`unsafe`/P/Invoke에서 관리 배열을 네이티브로 넘길 때(0-6, 그리고 B 그룹 0-8·0-10) `fixed`를 쓴다. "내가 변수를 다시 대입하지도 않는데 왜 고정이 필요하지?" 가 헷갈리기 쉬운데, 핵심은 **.NET GC가 힙 데이터를 물리적으로 옮긴다**는 점이다.

**스택 vs 힙**

- `int n = 10;` 같은 지역 값 타입 → **스택**에 값이 직접 들어가고 GC 대상이 아니다.
- `float[] buf = ...;` 같은 배열 → **둘로 쪼개진다**:
  - 참조(배열을 가리키는 주소 하나) → 스택(지역) 또는 객체 내부(필드)
  - 배열 실데이터(헤더 + 원소들) → **GC 힙** (← 옮겨질 수 있는 대상)

**GC가 옮길 때 무슨 일이 일어나나 (compaction)**

.NET은 compacting GC라, 살아있는 객체를 한쪽으로 몰아 정리하면서 배열의 *물리 주소*가 바뀔 수 있다. 이때:

| 무엇을 들고 있나 | GC가 이동 시 갱신? | `fixed` 필요? |
|---|---|---|
| 관리 참조 (`float[]`, `Span<float>`, `arr[i]`) | ✅ 자동 갱신 (그래서 관리 코드는 눈치 못 챔) | ❌ |
| raw 포인터 (`float* p`) | ❌ GC가 존재를 모름 → 옛 주소 그대로 = 댕글링 | ✅ |

```
GC compaction 전:   힙 ...[배열@1000]...        GC compaction 후:   힙 ...[배열@600]...
  buf (관리 참조) → 1000                          buf (관리 참조) → 600   ← GC 자동 갱신 ✅
  float* p        → 1000                          float* p        → 1000  ← 아무도 안 고침 ❌
```

→ raw 포인터를 쓰는 동안 GC가 그 배열을 옮기면 포인터가 옛 주소를 가리켜 메모리가 깨진다. `fixed`는 GC에게 **"이 블록 동안 이 배열 옮기지 마(pin)"** 라고 못 박아 그 이동을 막는다.

```csharp
fixed (float* pIn = input)            // ← 이 { } 안에서만 input 배열을 pin
fixed (float* pOut = _outputBuffer)
{
    vinfer_infer(_handle, pIn, pOut, _outputLen);  // 네이티브가 이 주소를 읽고/씀
}                                     // ← 블록을 벗어나면 pin 해제
```

**정리**

- 그래서 C# 코드 대부분은 `fixed`를 평생 안 써도 된다(관리 참조는 GC가 알아서 따라가므로).
- `fixed` / `stackalloc` / `Marshal` / `unsafe`가 등장하는 순간 = "GC 자동 추적 바깥으로 포인터를 꺼냈다"는 신호 → 핀이나 수동 메모리 관리가 필요한 지점이다.
- 이 프로젝트에서 `fixed`가 나오는 곳(`YoloPreprocessor.cs:66-67`, `NativeYoloEngine.cs:85-89`)은 전부 **관리 배열을 네이티브/`unsafe` 경계로 넘기는** 자리다. (반대로 `Marshal.AllocHGlobal` 같은 **비관리** 메모리는 애초에 GC 대상이 아니라 `fixed`가 필요 없다.)

### B. 추론 최적화 (모델 실행 — YOLO · VLM)

> 한 프레임의 추론 경로: **전처리(0-8) → 추론 실행(0-9 출력 재사용 · 0-10 입력 제로카피) → 디바이스별 직렬화(0-11)**. HW 표시 경로에서 추론 입력을 만들 땐 0-12(지연 readback)이 끼어든다. 엔진별 가속은 0-13, VLM은 0-14.

#### 0-8. 전처리: `Span<T>` / `unsafe` 단일 패스로 연산 절약

- **문제**: 전처리에서 ImageSharp 객체 할당 + B↔R 스왑 풀복사 + Resize 재할당이 겹치고, 4차원 인덱서 루프는 느리다.
- **해결**: BGR byte[] → CHW float 텐서를 **한 번의 `unsafe` 패스**로 (리사이즈+스왑+정규화+HWC→CHW 동시) 처리. 원본 전체가 아니라 출력 영역(≈640×newH)만 순회.
- **코드**: `Services/Yolo/YoloPreprocessor.cs:42-123`
  - `fixed (byte* srcBase = bgrPixels) fixed (float* dstBase = buffer)` (`:66-67`) — 핀 후 포인터 산술로 직접 기록
  - letterbox 빈 영역 초기화는 `buffer.AsSpan(...).Fill(PadValueNormalized)` (`:61`) — `Span.Fill`은 SIMD라 인덱서 루프보다 빠름
  - 후처리도 출력 버퍼를 복사 없이 슬라이스해 파싱: `YoloInferenceEngine.cs:184` — `_outputBuffer.AsSpan(0, _outputLen)`
- 텐서 백킹은 `ArrayPool`에서 빌린다(0-1) → 대여 `:57`, 반납 `LetterboxResult.cs:33`.

#### 0-9. 추론 출력 버퍼 재사용 (ORT `IoBinding`)

- **문제**: `session.Run(inputs)` 레거시 API는 매 추론마다 출력 텐서([1,84,8400] ≈ 2.7MB)를 새로 할당 → 또 LOH/Gen2.
- **해결**: 출력 `OrtValue`를 고정 버퍼 위에 **한 번만** 바인드(`IoBinding`)해 재사용. 매 프레임 출력 할당이 사라진다.
- **코드**: `Services/Yolo/YoloInferenceEngine.cs:149-153`
  - `_outputBuffer = new float[_outputLen];` (1회 할당)
  - `_outputOrt = OrtValue.CreateTensorValueFromMemory(_outputBuffer, _outputShape);`
  - `_binding.BindOutput(_outputName, _outputOrt);` → 이후 `_session.RunWithBinding(...)` (`:176`)이 같은 버퍼에 in-place 기록

#### 0-10. 입력도 복사 없이 포인터/메모리만 넘기기 (제로카피)

- **문제**: "전처리 텐서 → ORT가 읽을 텐서" 로 넘길 때 또 복사하면 헛수고.
- **해결**: 입력도 **풀 버퍼의 메모리를 그대로 가리키는** `OrtValue`로 감싼다(복사 없음).
- **코드**:
  - 입력 제로카피: `Services/Yolo/YoloInferenceEngine.cs:171-172` — `OrtValue.CreateTensorValueFromMemory(..., lb.Tensor.Buffer, _inputShape)`
  - 네이티브(C++) 엔진은 입출력 버퍼를 `fixed`로 핀해 포인터만 전달: `Services/Yolo/NativeYoloEngine.cs:85-89` — `fixed (float* pIn = input) fixed (float* pOut = _outputBuffer) { vinfer_infer(_handle, pIn, pOut, _outputLen); }` (왜 `fixed`인지는 0-7)

#### 0-11. 단일 스레드 병목 제거 — "전역 직렬화"가 아니라 "디바이스별 직렬화"

- **문제**: 같은 ONNX 세션을 여러 스트림이 동시에 호출하면 깨진다. 그렇다고 추론 전체를 하나의 락으로 묶으면 CPU·GPU가 서로를 기다리는 단일 스레드 병목이 된다.
- **해결**: 락의 단위를 **디바이스(엔진)별**로 쪼갠다. 같은 디바이스끼리만 직렬화하고, CPU↔CUDA 처럼 다른 디바이스는 병렬로 돈다.
- **코드**:
  - 직렬화 데코레이터: `Services/Rtsp/SerializedFrameDetector.cs:15,25` — `SemaphoreSlim(1,1)` + `WaitAsync`
  - 디바이스마다 **별도** 데코레이터 인스턴스: `MainWindow.xaml.cs:94-99` (CPU/DML/CUDA/C++/TRT 각각 1개)

#### 0-12. 지연 readback — HW 프레임은 추론이 필요할 때만 CPU로 내린다

- **문제**: 표시는 GPU에서 끝나지만(0-2), YOLO 추론은 CPU 텐서가 필요하다. 매 프레임 GPU→CPU 다운로드 + 색변환하면 GPU 경합·헛수고가 크다.
- **해결**: **추론 ON 이고 직전 추론 프레임이 이미 소비된(채널이 빈)** 스트림만 GPU→CPU 다운로드(`av_hwframe_transfer_data`) + BGR 변환(`sws_scale`)을 수행.
- **코드**:
  - 게이트: `Services/Rtsp/Pipeline/VideoRenderer.cs:181`, 람다 `RtspFrameSource.cs:179` (`() => _channel.Reader.Count == 0`)
  - 다운로드/변환: `VideoRenderer.cs:211`(`av_hwframe_transfer_data`), `:260`(`sws_scale`)

#### 0-13. 엔진별 가속 — TensorRT 캐시 / FP16 모델

- **TensorRT 엔진 디스크 캐시**: 첫 실행의 엔진 빌드(수십 초)를 두 번째 실행부터 건너뛴다 — `Services/Yolo/YoloInferenceEngine.cs:113-116` (`trt_engine_cache_enable`).
- **FP16 모델(DirectML 전용)**: GPU FP16 가속 + VRAM 절반. 입출력은 FP32로 유지해 전/후처리 코드를 공유 — `MainWindow.xaml.cs:60-61` (§8 참고).

#### 0-14. VLM 호출이 파이프라인을 막지 않게

- **문제**: VLM(Ollama)은 한 번에 수백 ms~수 초. 추론 루프에서 동기로 부르면 영상·검출이 같이 멈춘다.
- **해결**: 사람 검출 + 쿨다운(10초) 게이트를 통과한 프레임만 **논블로킹**으로 용량 1 큐에 던지고(처리 중이면 drop), 백그라운드 워커가 JPEG 인코딩(긴 변 640으로 다운스케일 → 비전 토큰·VRAM↓) 후 호출.
- **코드**:
  - 논블로킹 트리거 + drop: `Services/Vlm/VlmDescriptionService.cs:44-49`(`DropWrite`), `:57-82`(`TryTrigger`)
  - 입력 다운스케일: `Services/Vlm/VlmDescriptionService.cs:20`(`MaxEdge=640`), `:121-138`(`EncodeJpeg`)
  - CPU 모드일 때 Ollama 프로세스 우선순위를 BelowNormal로 낮춰 영상에 CPU 양보: `Services/Vlm/OllamaVlmClient.cs:47`
  - `HttpClient`는 정적 1개 공유(소켓 고갈 방지): `Services/Vlm/OllamaVlmClient.cs:22`
  - **워밍업**: 앱 시작 시 빈 프롬프트로 모델을 미리 적재해 첫 호출 콜드 로딩을 당김 — `StreamItemViewModel.WarmUpVlmAsync()`.

---

## 1. 개발 환경

| 항목 | 버전 / 비고 |
|---|---|
| OS | Windows 11 |
| IDE | Visual Studio 2022 (C++ 워크로드 포함 — 네이티브 추론 DLL 빌드용) |
| .NET SDK | .NET 10 (`net10.0-windows`) |
| 언어 | C# (Nullable enable, ImplicitUsings enable, AllowUnsafeBlocks) + 네이티브 C++ |
| 플랫폼 | x64 강제 (FFmpeg 네이티브 DLL이 x64) |
| UI | WPF (MVVM) |

### NuGet 패키지

| 패키지 | 용도 |
|---|---|
| `Microsoft.ML.OnnxRuntime.Gpu` / `Microsoft.ML.OnnxRuntime.DirectML` (1.20.1) | ONNX 추론. `UseDirectML` 토글로 둘 중 하나 선택 (§10) |
| `SixLabors.ImageSharp` | 이미지 전처리 (리사이즈, 정규화, HWC→CHW) |
| `OpenCvSharp4` / `OpenCvSharp4.runtime.win` | 보조 이미지 처리 (VLM 입력 JPEG 인코딩 등) |
| `FFmpeg.AutoGen` (8.1.0) | RTSP 수신 + H.264/AAC 디코딩 P/Invoke 바인딩 |
| `Vortice.Direct3D11` / `Direct3D9` / `D3DCompiler` (3.8.3) | GPU 컴포지터 + D3DImage 표시 + YUV→RGB 셰이더 |
| `NAudio` (2.2.1) | 디코딩된 PCM을 스피커로 출력 (WaveOut/BufferedWaveProvider) |

> ORT 패키지 기본값은 `UseDirectML=false` → **`Microsoft.ML.OnnxRuntime.Gpu`(CUDA/TensorRT)** 이다. DirectML로 쓰려면 `UseDirectML=true` 로 전환한다 (§10).

### FFmpeg 네이티브 DLL

`Native/win-x64/` 아래 FFmpeg 8.x DLL들(`avformat-*.dll`, `avcodec-*.dll`, `avutil-*.dll`, `swscale-*.dll`, `swresample-*.dll`)을 두면 빌드 시 출력 폴더로 자동 복사된다. 앱 시작 시 `FFmpegLibraryLoader.EnsureRegistered()`가 `ffmpeg.RootPath`에 출력 폴더를 박아 P/Invoke 검색 경로를 잡는다.

---

## 2. 폴더 구조

```
Vision.MultiStream.Inference/                <-- 저장소 루트
├── README.md
├── Tester/                                  <-- 가상 CCTV 환경
│   └── cameraTest/
│       ├── Video1.mp4 / Video2.mp4          샘플 CCTV 영상
│       ├── run_cameras_tcp.bat              TCP 모드 다중 카메라 실행
│       └── run_cameras_udp.bat              UDP 모드 다중 카메라 실행
└── Vision.MultiStream.Inference/            <-- 솔루션 (.slnx: C# + C++ 2개 프로젝트)
    ├── Native/
    │   └── Vision.MultiStream.Native/       <-- C++ 네이티브 추론 DLL (vision_infer.dll)
    │       └── Vision.MultiStream.Native.vcxproj   ORT+DirectML 세션 (Phase 3 "GPU(C++)")
    └── Vision.MultiStream.Inference/        <-- C# 메인 프로젝트
        ├── Assets/Models/yolov8n.onnx       FP32 원본 (CPU / CUDA / TensorRT / C++)
        ├── Assets/Models/yolov8n_fp16.onnx  FP16 변환본 (DirectML 전용)
        ├── Assets/TestImages/               스냅샷 탭 테스트용 정적 이미지
        ├── Native/win-x64/                  FFmpeg 네이티브 DLL (빌드 시 출력 폴더로 복사)
        ├── Common/
        │   ├── BaseViewModel.cs
        │   ├── RelayCommand.cs / AsyncRelayCommand.cs
        │   ├── InverseBoolConverter.cs
        │   └── PerfProbe.cs                 CPU/메모리/GC 샘플링 (성능 상태바)
        ├── Models/
        │   ├── Detection.cs                 검출 결과 1건 (박스 좌표, 클래스, 신뢰도)
        │   └── InferenceTimings.cs          추론 단계별 소요 시간
        ├── Services/
        │   ├── CocoLabels.cs
        │   ├── Audio/
        │   │   ├── AudioFrame.cs            PCM 1프레임 (byte[] + sampleRate + channels)
        │   │   ├── IAudioOutput.cs          오디오 sink 인터페이스
        │   │   └── WasapiAudioOutput.cs     NAudio WaveOutEvent 기반 출력 구현
        │   ├── Direct3D/
        │   │   ├── StreamCompositor.cs      여러 스트림을 단일 D3D11 surface 에 합성 → 1개 D3DImage
        │   │   └── D3DImageYuvPresenter.cs  per-stream(개별) YUV→D3DImage 표시 경로
        │   ├── Rtsp/
        │   │   ├── RtspFrameSource.cs       RTSP 파이프라인 파사드 (단계 조립·구동)
        │   │   ├── RtspFrame.cs             추론용 BGR 프레임 (immutable)
        │   │   ├── RtspYuvFrame.cs          표시용 YUV 평면 프레임 (SW 디코드)
        │   │   ├── RtspD3D11Frame.cs        표시용 D3D11 텍스처 프레임 (HW 디코드)
        │   │   ├── MediaClock.cs            오디오 마스터 / wall-clock 동기화 기준
        │   │   ├── RtspFrameDetector.cs     RTSP 도메인 추론 어댑터
        │   │   ├── SerializedFrameDetector.cs  디바이스별 추론 직렬화 데코레이터
        │   │   ├── IRtspFrameDetector.cs
        │   │   ├── FFmpeg/FFmpegLibraryLoader.cs  네이티브 DLL 경로 등록 + 로그 레벨
        │   │   └── Pipeline/                <-- 단계별 분해 (각 단계 = 별도 스레드)
        │   │       ├── RtspDemuxer.cs       av_read_frame → 비디오/오디오 패킷 큐로 분기
        │   │       ├── VideoDecoder.cs      패킷 → 프레임 (SW YUV / HW D3D11VA)
        │   │       ├── VideoRenderer.cs     MediaClock 게이팅 + 표시/추론 프레임 발행
        │   │       ├── AudioDecoder.cs      패킷 → PCM 프레임 (swr_convert)
        │   │       ├── AudioRenderer.cs     IAudioOutput 으로 페이싱 재생
        │   │       ├── HwDeviceContext.cs   D3D11VA 디바이스/컨텍스트 (디코더·컴포지터 공유)
        │   │       └── FfmpegNative.cs      P/Invoke 헬퍼 + 큐 잔량 free
        │   ├── Snapshot/
        │   │   ├── SnapshotDetector.cs      정적 이미지 추론 어댑터
        │   │   └── ISnapshotDetector.cs
        │   ├── Vlm/
        │   │   ├── IVlmClient.cs
        │   │   ├── OllamaVlmClient.cs       로컬 Ollama HTTP 호출 (CPU/GPU 토글, 워밍업)
        │   │   └── VlmDescriptionService.cs 사람 검출 프레임만 쿨다운 게이트 → VLM 묘사
        │   └── Yolo/
        │       ├── IYoloEngine.cs           추론 엔진 추상화 (관리/네이티브 공통)
        │       ├── YoloInferenceEngine.cs   ONNX 세션 실행 (CPU/DML/CUDA/TensorRT)
        │       ├── NativeYoloEngine.cs      vision_infer.dll 호출 ("GPU(C++)")
        │       ├── YoloPreprocessor.cs      letterbox + 정규화 + CHW 텐서 변환
        │       ├── YoloPostprocessor.cs     [1,84,8400] → NMS → Detection 리스트
        │       └── LetterboxResult.cs       텐서 + 좌표 역변환 메타정보
        ├── ViewModels/
        │   ├── ShellViewModel.cs            최상위 VM (탭 묶음)
        │   ├── MultiStreamViewModel.cs      다중 스트림 컬렉션 + 전체 제어 + 컴포지터 연결
        │   ├── StreamItemViewModel.cs       스트림 1개 단위 VM (캡처/추론/렌더링/오디오/VLM)
        │   ├── SnapshotViewModel.cs         스냅샷 탭 VM (정적 이미지 검출)
        │   └── PerformanceViewModel.cs      성능 상태바 VM (CPU/RAM/GC)
        ├── Views/
        │   └── BulkAddStreamsWindow.xaml    URL 일괄 추가 모달
        └── MainWindow.xaml(.cs)
```

---

## 3. RTSP 수신 — 파이프라인 모델

`RtspFrameSource`는 파이프라인 파사드다. 역할별 단계 클래스를 조립·구동하고, 각 단계는 별도 스레드에서 돌며 단계 사이는 바운디드 큐로만 주고받는다.

```
[RtspDemuxer] → VideoPacketQueue → [VideoDecoder] → VideoFrameQueue → [VideoRenderer]
              → AudioPacketQueue → [AudioDecoder] → AudioFrameQueue → [AudioRenderer]
```

- **RtspDemuxer**: `av_read_frame()` 로 RTSP 패킷 수신 → `stream_index` 로 비디오/오디오 큐 분기(`av_packet_clone`). 오디오 미사용이면 오디오 패킷은 즉시 drop.
- **VideoDecoder**: `avcodec_send_packet` → `avcodec_receive_frame`. SW 경로는 YUV420P, HW 경로는 D3D11VA(NV12 텍스처)로 디코딩.
- **VideoRenderer**: `MediaClock` 게이팅(늦으면 drop, 이르면 sleep)으로 페이싱하며, 표시 프레임(YUV/D3D11)과 추론 프레임(BGR)을 발행. 렌더러 페이싱이 디코더→디먹서로 backpressure 된다.
- **AudioDecoder / AudioRenderer**: PCM(S16) 변환 후 `IAudioOutput`(WASAPI/NAudio)으로 사운드카드 페이싱에 맞춰 재생.

큐는 `BlockingCollection`(bounded): 가득 차면 생산자가 대기(backpressure)하므로 멀쩡한 프레임을 임의로 버리지 않는다. 지연 제어는 `VideoRenderer`의 `MediaClock` 게이팅이 담당한다.

### 비디오/오디오 활성화 정책

`Start(audioOutput, useHardwareDecoding)`에서 `audioOutput != null` 이고 스트림에 오디오 트랙이 있을 때만 오디오 단계(디코더/렌더러/큐)를 만든다. 오디오 OFF인 쪽은 스레드도 큐도 만들지 않아 비용 0. 오디오 코덱 열기에 실패해도 영상은 그대로 진행한다.

### 동기화 (MediaClock)

- 오디오가 있으면 **audio-master**: 사운드카드가 실제로 재생 중인 위치를 기준으로 영상 PTS를 맞춘다.
- 오디오가 없으면 **wall-clock**: 시작 시각 기준으로 페이싱.

### 메모리/소유권 규약

- 디먹서가 `av_packet_clone`으로 만든 사본의 소유권은 디코더 스레드로 이전 → 디코더가 free 책임.
- `Stop()`은 생산자→소비자 순으로 큐를 닫으며 연쇄 종료: 디먹서 Join → 패킷 큐 `CompleteAdding` → 디코더 Join → 프레임 큐 `CompleteAdding` → 렌더러 Join → 큐 잔량(unmanaged 포인터) free.

---

## 4. 표시 경로 (개별 / 컴포지터 / 하드웨어 디코딩)

스트림 추가 시 **표시 모드**를 3가지 중 고른다.

| 표시 모드 | 디코딩 | 표시 |
|---|---|---|
| **CPU+개별** | SW (YUV420P) | 스트림마다 `D3DImageYuvPresenter` 1개 → 셀별 D3DImage |
| **CPU+컴포지터** | SW (YUV420P) | 단일 `StreamCompositor` 가 전 스트림을 한 surface 에 합성 → D3DImage 1개 |
| **GPU+컴포지터** | HW (D3D11VA, NV12) | HW 디코드 텍스처를 CPU 다운로드 없이 GPU 안에서 슬롯 텍스처로 복사 → 합성 |

- **컴포지터**(`StreamCompositor`)는 D3D11 디바이스 하나로 여러 스트림을 타일 위치에 YUV→RGB 셰이더로 드로우해 공유 RT에 합성하고, UI는 프레임당 1회만 D3DImage present 한다. HW 디코더와 같은 D3D11 디바이스(`HwDeviceContext`)를 공유해 GPU 안에서 zero-copy로 처리한다.
- 컴포지터 초기화(D3D9/D3D11)에 실패하면(GPU/드라이버 부재 등) 자동으로 per-stream `D3DImageYuvPresenter` 경로로 폴백하고, 그것도 실패하면 WriteableBitmap CPU 경로로 떨어진다.

---

## 5. 추론 파이프라인 (스트림 단위)

```
[StreamItemViewModel N개]  각각 독립 실행
  │
  ├─ RtspFrameSource (파이프라인, 위 §3)
  │    ├─ 표시 프레임 이벤트 → Dispatcher → 컴포지터/Presenter 갱신 (DisplayFps 측정)
  │    └─ PublishInferenceFrame → 추론 Channel(capacity=1)
  │
  └─ 추론 Task (ThreadPool, InferenceLoopAsync)  — 스트림별 "추론 사용" ON 일 때만
       └─ Channel.ReadAllAsync → 최신 BGR 프레임 1장씩
            ├─ SerializedFrameDetector.DetectAsync (같은 디바이스끼리 직렬화)
            │    ├─ YoloPreprocessor:  BGR byte[] → [1,3,640,640] 텐서
            │    ├─ IYoloEngine.Detect: ONNX Run → [1,84,8400]
            │    └─ YoloPostprocessor: NMS → Detection 리스트
            ├─ Dispatcher.BeginInvoke → Detections 갱신 (박스, InferenceFps)
            └─ 사람(class 0) 검출 시 → VlmDescriptionService.TryTrigger (§6)
```

**설계 포인트**
- `RtspFrame`은 immutable → 표시(이벤트)와 추론(Channel)에 동시 송출.
- 추론 Channel은 capacity=1 + `Wait` 모드. `PublishInferenceFrame`이 직접 이전 프레임을 evict + `Dispose`(ArrayPool 버퍼 반납)한 뒤 새 프레임을 넣는다. "최신 우선" 의미는 유지하되, DropOldest가 풀 버퍼를 반납 없이 버려 LOH/Gen2로 새던 문제를 막는다.
- `SerializedFrameDetector`: 같은 ONNX 세션의 동시 호출을 막는 데코레이터. 디바이스별로 1개씩 공유 → 같은 디바이스끼리는 직렬화, 다른 디바이스끼리는 병렬.
- 추론은 스트림별로 ON/OFF (🧠 토글). OFF면 추론 Task 자체를 띄우지 않는다.

---

## 6. VLM 장면 묘사 (Ollama)

YOLO가 **사람(COCO class 0)** 을 검출한 프레임만 로컬 VLM에 넘겨 "사람들이 무엇을 하는지"를 한 문장(한국어)으로 묘사하고, 타일 하단에 자막으로 표시한다. 스트림별 **'VLM 사용'** 체크박스로 켠다.

```
추론 루프 (사람 검출)
  └─ VlmDescriptionService.TryTrigger(bgr, w, h, personCount)   ← 논블로킹
       ├─ 쿨다운 게이트 (10초) + 사람 수 > 0 일 때만 통과
       ├─ 용량 1 큐(DropWrite) — 처리 중이면 새 요청은 조용히 drop
       └─ 백그라운드 워커: BGR → JPEG(긴 변 640 다운스케일) → IVlmClient.DescribeAsync
            └─ DescriptionReady 이벤트 → UI 자막 갱신
```

- VLM이 느려도 추론/표시는 막히지 않는다(fire-and-forget + drop).
- **모델**: `qwen2.5vl:3b` (Ollama). 사전 준비: `ollama pull qwen2.5vl:3b` 후 Ollama 데몬 기동(기본 `localhost:11434`).
- **디바이스 토글(전역)**: VLM CPU / GPU 선택. Ollama는 모델을 한 번만 로드하므로 디바이스는 앱 전역 공통이다.
  - CPU: `num_gpu=0`, `num_thread=4` + Ollama 프로세스 우선순위 BelowNormal → GPU/CPU를 영상·YOLO에 양보.
  - GPU: Ollama가 VRAM에 맞게 자동 오프로드(빠르지만 영상과 GPU 경합).
- 앱 시작 시 `WarmUpVlmAsync()`가 빈 프롬프트로 모델을 미리 적재해 첫 묘사의 콜드 로딩 지연을 줄인다(Ollama 미기동 시 조용히 무시).

---

## 7. UI 구조

상단에 **성능 상태바**(CPU% / RAM MB / GC Gen0~2 / GC Pause)가 항상 떠 있고, 아래에 **2개 탭**이 있다.

### RTSP (다중 스트림) 탭

```
┌──────────────────────────┬──────────────────────────────────────┐
│  스트림 추가 폼          │  레이아웃: [Auto▼]                    │
│   이름 / URL             │ ┌──────┬──────┬──────┐              │
│   표시: 개별/컴포지터/GPU│ │ tile │ tile │ tile │              │
│   추론: CPU/DML/CUDA/    │ ├──────┼──────┼──────┤              │
│         TRT/C++  ☑추론   │ │ tile │ tile │ tile │              │
│   VLM: CPU/GPU  ☑VLM     │ └──────┴──────┴──────┘              │
│   [+ 추가] [📋 일괄]     │   UniformGrid 자동 N×N (배경=컴포지터)│
│──────────────────────────│                                      │
│  스트림 목록             │  각 타일(오버레이):                   │
│   ● cam1  추론:CPU 표시:개별│   상단바: 이름/추론/디코더/상태       │
│     🎥 🔊 🧠 ▶ ⏹ ✕      │          + 🎥🔊🧠 토글 + ▶⏹✕         │
│──────────────────────────│   검출 박스 (Yellow)                  │
│  전체 영상 🎥 ON/OFF     │   하단: 💬 VLM 자막                   │
│  전체 소리 🔊 ON/OFF     │        + 화면/추론 FPS                │
│  전체 추론 🧠 ON/OFF     │        + 전/추론/후처리 ms            │
│  ▶전체 ⏹전체 ✕전체삭제   │        + 오디오 fill/drop             │
│  등록: N개               │                                      │
└──────────────────────────┴──────────────────────────────────────┘
```

- **레이아웃**: Auto(자동 N×N) / 고정 그리드 선택.
- **표시 모드**: 개별 / CPU+컴포지터 / GPU+컴포지터 (컴포지터 사용 가능 시 활성).
- **추론 디바이스**: 현재 빌드에서 사용 가능한 옵션만 활성화(§8). '추론 사용' 체크로 추론 루프 ON/OFF.
- **VLM**: 'VLM 사용' 체크 + CPU/GPU 디바이스(전역).
- **토글**: 스트림별 🎥(영상) / 🔊(소리) / 🧠(추론) 개별 ON/OFF, 전체 버튼도 별도 제공.

### 스냅샷 (정적 이미지) 탭

이미지를 열어 CPU 엔진으로 1회 검출하고 박스 + 검출 결과 리스트(클래스/신뢰도/좌표)를 표시한다.

---

## 8. 추론 디바이스 선택

스트림 추가 시 개별로 선택한다. **현재 빌드의 ORT 패키지(`UseDirectML` 토글, §10)에 따라 사용 가능한 디바이스가 달라진다.**

| 옵션 | 설명 | 사용 가능 빌드 |
|---|---|---|
| **CPU** | 항상 사용 가능 | 모든 빌드 |
| **DirectML** | Windows 내장 DirectX 12 ML (추가 설치 없음) | `UseDirectML=true` |
| **GPU(C++)** | 네이티브 `vision_infer.dll`(ORT+DirectML) (§9) | `UseDirectML=true` |
| **CUDA** | NVIDIA CUDA 가속 (CUDA Toolkit + cuDNN 필요) | `UseDirectML=false` |
| **TensorRT** | NVIDIA TensorRT EP (FP16 + 엔진 캐시). 첫 실행 시 엔진 빌드로 수십 초 | `UseDirectML=false` |

> 현재 빌드에서 비활성인 디바이스를 선택하면(엔진 `null`) 자동으로 **CPU로 폴백**된다. 초기화에 실패한 GPU 옵션도 경고 팝업 후 CPU로 폴백.

### 모델 정밀도 (FP32 / FP16)

디바이스별로 다른 모델 파일을 로드한다 (`MainWindow.xaml.cs`).

| 엔진 | 모델 파일 | 정밀도 | 이유 |
|---|---|---|---|
| **DirectML** | `yolov8n_fp16.onnx` | FP16 | GPU에서 FP16 가속 + VRAM 절반 |
| **CPU / CUDA / TensorRT / C++** | `yolov8n.onnx` | FP32 | CPU는 FP16 네이티브 연산이 없어 오히려 느림. TensorRT는 FP32 입력 + 자체 `trt_fp16_enable`이 정석. (C++ 엔진도 FP32 입력) |

- 두 모델 모두 **입출력은 FP32 + 형상 동일**(`[1,3,640,640]` / `[1,84,8400]`)이라 전처리(`YoloPreprocessor`)·후처리(`YoloPostprocessor`)·`Detect()` 코드는 **그대로** 공유한다.
- FP16 변환은 `onnxconverter-common`의 `float16.convert_float_to_float16(..., keep_io_types=True)`로 1회 생성한다(입출력 FP32 유지가 핵심).

  ```python
  import onnx
  from onnxconverter_common import float16
  m = onnx.load("yolov8n.onnx")
  m16 = float16.convert_float_to_float16(m, keep_io_types=True)  # I/O 는 FP32 유지
  onnx.save(m16, "yolov8n_fp16.onnx")
  ```

---

## 9. Native C++ 추론 엔진 ("GPU(C++)")

`Native/Vision.MultiStream.Native/vision_infer.dll` 은 C++로 작성한 ORT+DirectML 세션이다. 전처리/후처리는 C#과 동일하게 재사용하고 **가운데 ORT 실행만 네이티브로** 넘긴다(`NativeYoloEngine` → P/Invoke).

- 경계: C#이 만든 입력 텐서(`[1,3,640,640]`)와 재사용 출력 버퍼를 `unsafe` 포인터로 핀해 네이티브로 그대로 전달(프레임당 신규 할당 0).
- 빌드: C++ 프로젝트는 `dotnet`으로 빌드할 수 없으므로, csproj의 `BuildNativeInferDll` 타깃이 `vswhere`로 VS 정식 `MSBuild.exe`를 찾아 `.vcxproj`를 x64로 빌드한 뒤, 산출 DLL을 C# 출력 폴더(`onnxruntime.dll`과 같은 폴더)로 복사한다.
- 네이티브 DLL은 ORT+DirectML 전제이므로 **`UseDirectML=true` 빌드에서만** 자동 빌드/적재된다(`EnableNativeInfer`가 함께 켜짐). CUDA 빌드에서는 타깃이 스킵된다.
- Visual Studio C++ 워크로드(vswhere/MSBuild)가 없으면 DirectML 빌드 시 빌드 오류가 난다.

---

## 10. 추론 패키지 전환 (UseDirectML 토글)

ORT의 DirectML 패키지와 Gpu(CUDA/TensorRT) 패키지는 같은 `onnxruntime.dll`을 서로 다른 빌드로 들고 와 한 빌드에 공존할 수 없다. 그래서 csproj의 `UseDirectML` 값 하나로 **패키지 + 컴파일 심볼 + 네이티브 DLL 빌드**를 함께 전환한다.

```xml
<!-- Vision.MultiStream.Inference.csproj -->
<PropertyGroup>
  <UseDirectML>false</UseDirectML>   <!-- 기본값 -->
</PropertyGroup>
```

| `UseDirectML` | ORT 패키지 | 컴파일 심볼 | 사용 가능 디바이스 |
|---|---|---|---|
| `true` | `Microsoft.ML.OnnxRuntime.DirectML` | `USE_DIRECTML` + `EnableNativeInfer` | CPU, DirectML, GPU(C++) |
| `false` (기본) | `Microsoft.ML.OnnxRuntime.Gpu` | — | CPU, CUDA, TensorRT |

> 값을 바꾼 뒤 리빌드하면 NuGet이 DLL을 자동으로 교체한다. `UseDirectML=true`는 추가로 네이티브 `vision_infer.dll`을 빌드한다(VS C++ 워크로드 필요, §9).

---

## 11. CUDA 환경 구축 (UseDirectML=false 빌드)

### Step 1. CUDA Toolkit 설치

```
winget install Nvidia.CUDA
```

또는 NVIDIA 공식 사이트에서 직접 다운로드 후 설치 (~3GB).

### Step 2. cuDNN 설치

- Windows / x86_64 / Tarball 다운로드
- 압축 해제 후 `bin/` DLL들을 CUDA 설치 경로에 복사

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin\
```

### Step 3. csproj 토글 확인

`UseDirectML=false`(기본값)인지 확인 (§10).

### 설치 확인

```powershell
nvcc --version
nvidia-smi
```

---

## 12. FFmpeg 네이티브 DLL 배치

빌드 전 `Vision.MultiStream.Inference/Native/win-x64/` 아래 FFmpeg 8.x shared DLL을 배치한다.

필요한 DLL:
- `avformat-*.dll`, `avcodec-*.dll`, `avutil-*.dll`
- `swscale-*.dll` (비디오 색공간 변환)
- `swresample-*.dll` (오디오 샘플 변환)

`csproj`에 다음 규칙이 있어 빌드 시 출력 폴더(`bin\$(Configuration)\net10.0-windows\`)로 자동 복사된다:

```xml
<None Include="Native\win-x64\*.dll">
  <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  <Link>%(Filename)%(Extension)</Link>
</None>
```

> 다운로드 추천: https://www.gyan.dev/ffmpeg/builds/ 의 "shared" 빌드.

---

## 13. 가상 CCTV 환경 구축

### Step 1. MediaMTX (RTSP 서버)

- 다운로드: https://github.com/bluenviron/mediamtx/releases
- `mediamtx_vX.X.X_windows_amd64.zip` 압축 해제 후 `mediamtx.exe` 실행
- 성공 로그: `[RTSP] listener opened on :8554`

### Step 2. FFmpeg 다운로드

- `bin/ffmpeg.exe` 를 `Tester/cameraTest/ffmpeg.exe` 위치에 배치

### Step 3. 단일 카메라 송출

```cmd
cd Tester/cameraTest
ffmpeg -re -stream_loop -1 -i Video1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
```

| 옵션 | 의미 |
|---|---|
| `-re` | 원본 프레임레이트로 재생 |
| `-stream_loop -1` | 무한 반복 |
| `-c copy` | 재인코딩 없이 패킷만 복사 → CPU 거의 안 씀 |

> 음원이 포함된 mp4를 그대로 송출하면 RTSP에 비디오+오디오 트랙이 함께 실린다.

### Step 4. 다중 카메라 일괄 실행

`Tester/cameraTest/` 폴더의 배치 파일 사용:

- `run_cameras_tcp.bat` — TCP 트랜스포트 (방화벽/NAT 환경)
- `run_cameras_udp.bat` — 기본 UDP 트랜스포트

송출 주소: `rtsp://localhost:8554/cam1`, `/cam2`, `/cam3` ...

---

## 14. 앱 실행

1. `Assets/Models/yolov8n.onnx`(FP32) 및 `yolov8n_fp16.onnx`(DirectML용) 파일 확인
2. `Native/win-x64/` 에 FFmpeg DLL 배치 (§12)
3. (선택) VLM을 쓰려면 Ollama 기동 + `ollama pull qwen2.5vl:3b`
4. csproj `UseDirectML` 토글 확인 (§10) 후 Visual Studio에서 빌드·실행
5. **RTSP 탭**: 이름·URL·표시모드·추론디바이스 입력 → `+ 추가` → `▶ 시작`
   - 여러 개 한 번에: `📋 일괄 추가` → URL 줄당 하나씩 붙여넣기
   - 영상/소리/추론 토글: 타일별 또는 전체 버튼 (🎥/🔊/🧠)
   - 전체 시작/중지: 하단 `▶ 전체 시작` / `⏹ 전체 정지`

---

## 디코딩 흐름 정리

- 디먹싱 스레드(RtspDemuxer)
  -> RTSP 연결(ffmpeg.avformat_open_input)
  -> 스트림 정보 파악(ffmpeg.avformat_find_stream_info)
  --> 스트림 인덱스 / 비디오·오디오 유무 / 디코딩 가능 코덱 확인
  --> 비디오·오디오 디코더/렌더러 스레드 시작
  --> rtsp 패킷을 받아 stream_index 보고 비디오/오디오 큐로 분기

    - 비디오 디코더(VideoDecoder)
      -> 큐에서 패킷을 꺼내 디코더에 넣음(avcodec_send_packet, 압축 포맷)
      -> 디코더가 만든 frame을 꺼냄(avcodec_receive_frame; SW=YUV420P / HW=D3D11VA NV12)
      -> 프레임 큐로 전달
    - 비디오 렌더러(VideoRenderer)
      -> MediaClock 으로 페이싱(오디오 있으면 오디오 위치, 없으면 wall-clock)
      -> 표시 프레임(YUV/D3D11) 발행 → 화면(컴포지터/Presenter)
      --> 추론 프레임(BGR) 발행 → 추론 Channel

    - 오디오 디코더(AudioDecoder)
      -> 큐에서 패킷을 꺼내 디코더에 넣음(avcodec_send_packet)
      -> 디코더가 만든 frame을 꺼냄(avcodec_receive_frame)
      -> 출력 장치용 포맷(S16)으로 변환(swr_convert) → 프레임 큐
    - 오디오 렌더러(AudioRenderer)
      -> IAudioOutput 으로 사운드카드가 페이싱하는 위치에 맞춰 재생
      -> 이 재생 위치가 MediaClock 의 audio-master 기준이 된다
