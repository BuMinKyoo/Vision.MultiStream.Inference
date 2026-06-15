# Vision.MultiStream.Inference

WPF (.NET 10) + ONNX Runtime 기반 **다중 RTSP 스트림 실시간 객체 검출** 프로젝트.
카메라를 여러 개 등록해 동시에 스트리밍하고, YOLOv8로 객체를 검출해 화면에 박스를 그린다.
**오디오 재생도 함께 지원**해 카메라 음성을 듣거나 끌 수 있다.

---

## 1. 개발 환경

| 항목 | 버전 / 비고 |
|---|---|
| OS | Windows 11 |
| IDE | Visual Studio 2022 (또는 Rider) |
| .NET SDK | .NET 10 (`net10.0-windows`) |
| 언어 | C# (Nullable enable, ImplicitUsings enable, AllowUnsafeBlocks) |
| 플랫폼 | x64 강제 (FFmpeg 네이티브 DLL이 x64) |
| UI | WPF (MVVM) |

### NuGet 패키지

| 패키지 | 용도 |
|---|---|
| `Microsoft.ML.OnnxRuntime.DirectML` | ONNX 추론 — DirectML(GPU) 기본값 |
| `SixLabors.ImageSharp` | 이미지 전처리 (리사이즈, 정규화, HWC→CHW) |
| `OpenCvSharp4` / `OpenCvSharp4.runtime.win` | 보조 이미지 처리 |
| `FFmpeg.AutoGen` (8.1.0) | RTSP 수신 + H.264/AAC 디코딩 P/Invoke 바인딩 |
| `NAudio` (2.2.1) | 디코딩된 PCM을 스피커로 출력 (WaveOut/BufferedWaveProvider) |

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
└── Vision.MultiStream.Inference/            <-- 솔루션
    └── Vision.MultiStream.Inference/
        ├── Assets/Models/yolov8n.onnx       FP32 원본 (CPU / CUDA / TensorRT 폴백)
        ├── Assets/Models/yolov8n_fp16.onnx  FP16 변환본 (DirectML 전용)
        ├── Native/win-x64/                  FFmpeg 네이티브 DLL (빌드 시 출력 폴더로 복사)
        ├── Common/
        │   ├── BaseViewModel.cs
        │   ├── RelayCommand.cs
        │   ├── AsyncRelayCommand.cs
        │   └── InverseBoolConverter.cs
        ├── Models/
        │   ├── Detection.cs                 검출 결과 1건 (박스 좌표, 클래스, 신뢰도)
        │   └── InferenceTimings.cs          추론 단계별 소요 시간
        ├── Services/
        │   ├── CocoLabels.cs
        │   ├── Audio/
        │   │   ├── AudioFrame.cs            PCM 1프레임 (byte[] + sampleRate + channels)
        │   │   ├── IAudioOutput.cs          오디오 sink 인터페이스
        │   │   └── WasapiAudioOutput.cs     NAudio WaveOutEvent 기반 출력 구현
        │   ├── Rtsp/
        │   │   ├── RtspFrame.cs             디코딩된 프레임 1개 (BGR byte[])
        │   │   ├── RtspFrameSource.cs       RTSP 수신 + 영상/오디오 디코딩 (3-스레드 모델)
        │   │   ├── RtspFrameDetector.cs     RTSP 도메인 추론 어댑터
        │   │   ├── SerializedFrameDetector.cs  디바이스별 추론 직렬화 데코레이터
        │   │   ├── IRtspFrameDetector.cs
        │   │   └── FFmpeg/
        │   │       └── FFmpegLibraryLoader.cs  네이티브 DLL 경로 등록 + 로그 레벨
        │   ├── Snapshot/
        │   │   ├── SnapshotDetector.cs      정적 이미지 추론 어댑터
        │   │   └── ISnapshotDetector.cs
        │   └── Yolo/
        │       ├── YoloPreprocessor.cs      letterbox + 정규화 + CHW 텐서 변환
        │       ├── YoloInferenceEngine.cs   ONNX 세션 실행 + NMS 후처리
        │       └── LetterboxResult.cs       텐서 + 좌표 역변환 메타정보
        ├── ViewModels/
        │   ├── ShellViewModel.cs            최상위 VM
        │   ├── MultiStreamViewModel.cs      다중 스트림 컬렉션 + 전체 제어 VM
        │   └── StreamItemViewModel.cs       스트림 1개 단위 VM (캡처/추론/렌더링/오디오)
        ├── Views/
        │   └── BulkAddStreamsWindow.xaml    URL 일괄 추가 모달
        └── MainWindow.xaml(.cs)
```

---

## 3. RTSP 수신 — FFmpeg 3-스레드 모델

`RtspFrameSource`는 스트림 1개당 최대 3개의 백그라운드 스레드를 띄운다. 비디오/오디오를 ON한 상태에 따라 필요한 스레드만 생성된다 ("끈 스트림은 비용 0" 정책).

```
RtspFrameSource.Start(videoEnabled, audioOutput)
  │
  ├─ Thread 1: Demuxer (RtspDemux[url])
  │    av_read_frame() 으로 RTSP 패킷 수신
  │    └─ stream_index 보고 비디오/오디오 큐로 분기 (av_packet_clone)
  │
  ├─ Thread 2: Video Decoder (RtspVideo[url]) — videoEnabled=true 일 때만
  │    BlockingCollection<AVPacket*> → avcodec_send_packet
  │    → avcodec_receive_frame (YUV)
  │    → sws_scale (BGR24)
  │    → Marshal.Copy → managed byte[]
  │    └─ FrameCaptured 이벤트 + Channel.TryWrite (1슬롯 DropOldest)
  │
  └─ Thread 3: Audio Decoder (RtspAudio[url]) — audioOutput!=null 일 때만
       BlockingCollection<AVPacket*> → avcodec_send_packet
       → avcodec_receive_frame (FLTP 등)
       → swr_convert (S16 stereo interleaved)
       └─ IAudioOutput.Push (WasapiAudioOutput)
```

### 비디오/오디오 활성화 정책

| `videoEnabled` | `audioOutput` | 동작 |
|---|---|---|
| `true` | `null` | 비디오만 디코딩, 오디오 패킷은 디먹서가 즉시 drop |
| `false` | non-null | 오디오만 디코딩, 비디오 패킷은 디먹서가 즉시 drop |
| `true` | non-null | 둘 다 디코딩 (3-스레드) |
| `false` | `null` | RTSP 자체를 안 연다 |

OFF인 쪽은 큐도 디코더 스레드도 만들지 않아서 CPU/메모리 비용 0.

### 메모리/소유권 규약

- 디먹서가 `av_packet_clone`으로 만든 사본의 소유권은 디코더 스레드로 이전 → 디코더가 `av_packet_free` 책임.
- `codecCtx`는 DemuxLoop의 finally가 해제. 디코더 스레드는 손대지 않음.
- `Stop()`은 Cancel → `CompleteAdding` → 스레드 Join → 큐 잔량 free 순서로 정리.

---

## 4. 추론 파이프라인 (스트림 단위)

```
[StreamItemViewModel N개]
  각각 독립 실행
  │
  ├─ RtspFrameSource (Thread 1~3, 위 §3)
  │    ├─ FrameCaptured 이벤트 → Dispatcher.BeginInvoke(Render)
  │    │    └─ WriteableBitmap 갱신 (영상 표시, DisplayFps 측정)
  │    └─ Channel.TryWrite → 1슬롯 DropOldest 큐
  │
  └─ 추론 Task (ThreadPool, InferenceLoopAsync)
       └─ Channel.ReadAllAsync → 최신 프레임 1장씩
            └─ SerializedFrameDetector.DetectAsync
                 ├─ SemaphoreSlim(1,1) 대기 (같은 디바이스끼리 직렬화)
                 ├─ YoloPreprocessor: BGR byte[] → [1,3,640,640] 텐서
                 ├─ YoloInferenceEngine: ONNX Run → [1,84,8400] → NMS
                 └─ Dispatcher.BeginInvoke(Background) → Detections 갱신 (박스, InferenceFps)
```

**설계 포인트**
- `RtspFrame`은 immutable record → 같은 인스턴스를 UI 디스플레이(이벤트)와 추론(Channel)에 동시 송출 (zero-copy 분기).
- 디스플레이는 이벤트 경로라 모든 프레임을 받음 (영상 끊김 최소화).
- 추론은 Channel(capacity=1, DropOldest) → 추론이 느려도 항상 최신 프레임만 처리, 결과와 영상 시간차 최소화.
- `SerializedFrameDetector`: 같은 ONNX 세션의 동시 호출을 막는 데코레이터. 디바이스별로 1개씩 공유 → CPU끼리, CUDA끼리는 직렬화, CPU↔CUDA는 병렬.

---

## 5. UI 구조 (RTSP 다중 스트림 탭)

```
┌──────────────────────┬──────────────────────────────────────┐
│  스트림 추가 폼       │  레이아웃: [Auto▼]                    │
│  이름 / URL / 디바이스│ ┌──────┬──────┬──────┐              │
│  [+ 추가] [📋 일괄]   │ │ tile │ tile │ tile │              │
│──────────────────────│ ├──────┼──────┼──────┤              │
│  스트림 목록          │ │ tile │ tile │ tile │              │
│  ● cam1  CPU         │ └──────┴──────┴──────┘              │
│    🎥 ON  🔊 ON      │   UniformGrid 자동 N×N               │
│    ▶/⏸ ✕            │                                      │
│──────────────────────│  각 타일: 영상 + 검출박스             │
│  [🎥 ON] [🎥 OFF]    │          + 이름/디바이스/FPS 오버레이  │
│  [🔊 ON] [🔊 OFF]    │          + 비디오/오디오 토글         │
│  [▶전체] [⏸전체]     │          + ✕ 즉시 제거 버튼           │
│  [✕전체삭제]         │                                      │
│  등록: N개            │                                      │
└──────────────────────┴──────────────────────────────────────┘
```

- **레이아웃**: Auto(자동 N×N) / 2×2 / 3×3 / 4×4 선택 가능
- **일괄 추가**: URL을 줄당 하나씩 붙여넣기 → 디바이스 선택 → 한 번에 등록
- **🎥 / 🔊 토글**: 스트림별로 비디오/오디오 개별 ON/OFF. 전체 ON/OFF 버튼도 별도 제공.
- **타일별 표시**: DisplayFps, InferenceFps, Preprocess/Inference/Postprocess ms.

---

## 6. 추론 디바이스 선택 (CPU / DirectML / CUDA)

스트림 추가 시 개별로 선택합니다.

| 옵션 | 설명 | 추가 설치 |
|---|---|---|
| **CPU** | 항상 사용 가능 | 없음 |
| **DirectML** | Windows 내장 DirectX 12 ML | 없음 |
| **CUDA** | NVIDIA CUDA 가속. 가장 빠름 | CUDA Toolkit + cuDNN 필요 |

> 초기화 실패한 옵션은 경고 팝업 후 자동으로 CPU로 폴백됩니다.

### 모델 정밀도 (FP32 / FP16)

디바이스별로 다른 모델 파일을 로드한다 (`MainWindow.xaml.cs`).

| 엔진 | 모델 파일 | 정밀도 | 이유 |
|---|---|---|---|
| **DirectML** | `yolov8n_fp16.onnx` | FP16 | GPU에서 FP16 가속 + VRAM 절반 |
| **CPU / CUDA / TensorRT 폴백** | `yolov8n.onnx` | FP32 | CPU는 FP16 네이티브 연산이 없어 오히려 느림. TensorRT는 FP32 입력 + 자체 `trt_fp16_enable`이 정석 |

- 두 모델 모두 **입출력은 FP32 + 형상 동일**(`[1,3,640,640]` / `[1,84,8400]`)이라 전처리(`YoloPreprocessor`)·후처리·`Detect()` 코드는 **그대로** 공유한다.
- FP16 변환은 `onnxconverter-common`의 `float16.convert_float_to_float16(..., keep_io_types=True)`로 1회 생성한다(입출력 FP32 유지가 핵심).

  ```python
  import onnx
  from onnxconverter_common import float16
  m = onnx.load("yolov8n.onnx")
  m16 = float16.convert_float_to_float16(m, keep_io_types=True)  # I/O 는 FP32 유지
  onnx.save(m16, "yolov8n_fp16.onnx")
  ```

- **A/B 속도 비교**: `MainWindow.xaml.cs`의 `modelPathDml` 파일명을 `yolov8n_fp16.onnx` ↔ `yolov8n.onnx`로 바꿔 리빌드. (DirectML 첫 추론은 셰이더 컴파일로 느리니 워밍업 후 정상상태 `InferenceMs`를 비교)

---

## 7. 추론 패키지 전환 방법

### DirectML (기본값)

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.20.1" />
```

### CUDA

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.25.1" />
```

> DirectML ↔ CUDA 패키지는 동시에 설치 불가합니다 (같은 `onnxruntime.dll` 이름 충돌).
> csproj에서 한 줄씩 교체 후 빌드하면 NuGet이 자동으로 DLL을 교체합니다.

---

## 8. CUDA 환경 구축 (선택)

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

### Step 3. csproj 패키지 교체

위 §7 참고.

### 설치 확인

```powershell
nvcc --version
nvidia-smi
```

---

## 9. FFmpeg 네이티브 DLL 배치

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

## 10. 가상 CCTV 환경 구축

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

## 11. 앱 실행

1. `Assets/Models/yolov8n.onnx`(FP32) 및 `yolov8n_fp16.onnx`(DirectML용) 파일 확인
2. `Native/win-x64/` 에 FFmpeg DLL 배치 (위 §9)
3. Visual Studio에서 빌드 후 실행
4. **RTSP 탭**: 이름·URL·디바이스 입력 → `+ 추가` → `▶ 시작`
   - 여러 개 한 번에: `📋 일괄 추가` → URL 줄당 하나씩 붙여넣기
   - 비디오/오디오 토글: 타일별 또는 전체 버튼 (🎥/🔊)
   - 전체 시작/중지: 하단 `▶ 전체 시작` / `⏸ 전체 중지`

---

## 디코딩 흐름 정리

- 디먹싱쓰레드(DemuxLoop)
  -> RTSP연결(ffmpeg.avformat_open_input)
  -> 스트림 몇개 읽어 데이터 파악(ffmpeg.avformat_find_stream_info)
  --> 스트림 인덱스 확인
  --> 비디오 있는지, 오디오 있는지 확인
  --> 디코딩 할 수 있는 코덱인지 확인
  --> 비디오 쓰레드 시작(VideoDecodeLoop)
  --> 오디오 쓰레드 시작(AudioDecodeLoop)
  --> rtsp에서 받은 패킷을 받아서 각쓰레드에 전달

    - 비디오쓰레드
      -> 큐에 있는 패킷을꺼내 디코더에 넣음(ffmpeg.avcodec_send_packet, 현재 압축된 포맷)
      -> 디코더가 만들어준 frame을 꺼내씀(ffmpeg.avcodec_receive_frame, 압축 풀린 포맷 ,EX. YUV)
      -> 오디오가 없으면 wallStartTicks 사용 / 오디오가 있으면 오디오 Ticks사용
      -> 디코딩된 frame을 BGR24로 변환
        --> 화면에 뿌리기
      --> 추론에 넘기기

    - 오디오쓰레드
      -> 큐에 있는 패킷을꺼내 디코더에 넣음(ffmpeg.avcodec_send_packet, 현재 압축된 포맷)
      -> 디코더가 만들어준 frame을 꺼내씀(ffmpeg.avcodec_receive_frame, 압축 풀린 포맷)
      -> 디코딩된 오디오를 출력 장치에 넣기 좋은 포맷으로 변경(EX. S16)
      -> 사운드카드가 페이싱하는 실제 오디오 재생 위치를 ticks로 삼는다
