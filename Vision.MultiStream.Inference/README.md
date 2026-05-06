# Vision.MultiStream.Inference

WPF (.NET 10) + ONNX Runtime 기반 **다중 RTSP 스트림 실시간 객체 검출** 프로젝트.
카메라를 여러 개 등록해 동시에 스트리밍하고, YOLOv8로 객체를 검출해 화면에 박스를 그린다.

---

## 1. 개발 환경

| 항목 | 버전 / 비고 |
|---|---|
| OS | Windows 11 |
| IDE | Visual Studio 2022 (또는 Rider) |
| .NET SDK | .NET 10 (`net10.0-windows`) |
| 언어 | C# (Nullable enable, ImplicitUsings enable) |
| UI | WPF (MVVM) |

### NuGet 패키지

| 패키지 | 용도 |
|---|---|
| `Microsoft.ML.OnnxRuntime.DirectML` | ONNX 추론 — DirectML(GPU) 기본값 |
| `SixLabors.ImageSharp` | 이미지 전처리 (리사이즈, 정규화, HWC→CHW) |
| `OpenCvSharp4` | RTSP 수신 + H.264 디코딩 |
| `OpenCvSharp4.runtime.win` | OpenCV 네이티브 바이너리 |

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
        ├── Assets/Models/yolov8n.onnx
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
        │   ├── Rtsp/
        │   │   ├── RtspFrame.cs             디코딩된 프레임 1개 (BGR byte[])
        │   │   ├── RtspFrameSource.cs       RTSP 수신 + 1슬롯 최신 프레임 채널
        │   │   ├── RtspFrameDetector.cs     RTSP 도메인 추론 어댑터
        │   │   ├── SerializedFrameDetector.cs  디바이스별 추론 직렬화 데코레이터
        │   │   └── IRtspFrameDetector.cs
        │   ├── Snapshot/
        │   │   ├── SnapshotDetector.cs      정적 이미지 추론 어댑터
        │   │   └── ISnapshotDetector.cs
        │   └── Yolo/
        │       ├── YoloPreprocessor.cs      letterbox + 정규화 + CHW 텐서 변환
        │       ├── YoloInferenceEngine.cs   ONNX 세션 실행 + NMS 후처리
        │       └── LetterboxResult.cs       텐서 + 좌표 역변환 메타정보
        ├── ViewModels/
        │   ├── ShellViewModel.cs            탭 두 개를 담는 최상위 VM
        │   ├── SnapshotViewModel.cs         정적 이미지 검출 VM
        │   ├── MultiStreamViewModel.cs      다중 스트림 컬렉션 + 전체 제어 VM
        │   └── StreamItemViewModel.cs       스트림 1개 단위 VM (캡처/추론/렌더링)
        ├── Views/
        │   └── BulkAddStreamsWindow.xaml    URL 일괄 추가 모달
        └── MainWindow.xaml(.cs)
```

---

## 3. 스레드 구조

스트림 1개당 아래 구조가 독립적으로 실행됩니다.

```
[StreamItemViewModel N개]
  각각 독립 실행
  │
  ├─ 캡처 스레드 (전용 Thread, RtspFrameSource)
  │    └─ VideoCapture.Read() 블로킹 루프
  │         ├─ FrameCaptured 이벤트 → Dispatcher.BeginInvoke(Render)
  │         │    └─ WriteableBitmap 갱신 (영상 표시)
  │         └─ Channel.TryWrite → 1슬롯 DropOldest 큐
  │
  └─ 추론 Task (ThreadPool, InferenceLoopAsync)
       └─ Channel.ReadAllAsync → 최신 프레임 1장씩
            └─ SerializedFrameDetector.DetectAsync
                 ├─ SemaphoreSlim(1,1) 대기 (같은 디바이스끼리 직렬화)
                 ├─ YoloPreprocessor: BGR byte[] → [1,3,640,640] 텐서
                 ├─ YoloInferenceEngine: ONNX Run → [1,84,8400] → NMS
                 └─ Dispatcher.BeginInvoke(Background) → Detections 갱신 (박스)
```

**설계 포인트**
- 캡처 스레드가 전용 Thread인 이유: `VideoCapture.Read()`가 블로킹 호출이라 ThreadPool을 점유하면 다른 Task가 굶음
- 1슬롯 `DropOldest` 채널: 추론이 느려도 항상 최신 프레임만 처리 → 영상/박스 시간차 최소화
- `SerializedFrameDetector`: 같은 ONNX 세션의 동시 호출을 막는 데코레이터. 디바이스별로 1개씩 공유하므로 CPU 스트림끼리, CUDA 스트림끼리는 직렬화되고, CPU ↔ CUDA는 병렬 동작

---

## 4. UI 구조 (RTSP 다중 스트림 탭)

```
┌─────────────────────┬──────────────────────────────────────┐
│  스트림 추가 폼      │  레이아웃: [Auto▼]                    │
│  이름 / URL / 디바이스│ ┌──────┬──────┬──────┐              │
│  [+ 추가] [📋 일괄]  │ │ tile │ tile │ tile │              │
│─────────────────────│ ├──────┼──────┼──────┤              │
│  스트림 목록         │ │ tile │ tile │ tile │              │
│  ● cam1  CPU  ▶/⏸ ✕│ └──────┴──────┴──────┘              │
│  ○ cam2  CUDA ▶/⏸ ✕│   UniformGrid 자동 N×N               │
│─────────────────────│                                      │
│  [▶전체] [⏸전체]    │  각 타일: 영상 + 검출박스             │
│  [✕전체삭제]        │          + 이름/디바이스/FPS 오버레이  │
│  등록: N개           │          + ✕ 즉시 제거 버튼           │
└─────────────────────┴──────────────────────────────────────┘
```

- **레이아웃**: Auto(자동 N×N) / 2×2 / 3×3 / 4×4 선택 가능
- **일괄 추가**: URL을 줄당 하나씩 붙여넣기 → 디바이스 선택 → 한 번에 등록

---

## 5. 추론 디바이스 선택 (CPU / DirectML / CUDA)

스트림 추가 시 개별로 선택합니다.

| 옵션 | 설명 | 추가 설치 |
|---|---|---|
| **CPU** | 항상 사용 가능 | 없음 |
| **DirectML** | Windows 내장 DirectX 12 ML | 없음 |
| **CUDA** | NVIDIA CUDA 가속. 가장 빠름 | CUDA Toolkit + cuDNN 필요 |

> 초기화 실패한 옵션은 경고 팝업 후 자동으로 CPU로 폴백됩니다.

---

## 6. 추론 패키지 전환 방법

### DirectML (기본값)

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.25.1" />
```

### CUDA

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.25.1" />
```

> DirectML ↔ CUDA 패키지는 동시에 설치 불가합니다 (같은 `onnxruntime.dll` 이름 충돌).
> csproj에서 한 줄씩 교체 후 빌드하면 NuGet이 자동으로 DLL을 교체합니다.

---

## 7. CUDA 환경 구축 (선택)

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

위 §6 참고.

### 설치 확인

```powershell
nvcc --version
nvidia-smi
```

---

## 8. 가상 CCTV 환경 구축

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

### Step 4. 다중 카메라 일괄 실행

`Tester/cameraTest/` 폴더의 배치 파일 사용:

- `run_cameras_tcp.bat` — TCP 트랜스포트 (방화벽/NAT 환경)
- `run_cameras_udp.bat` — 기본 UDP 트랜스포트

송출 주소: `rtsp://localhost:8554/cam1`, `/cam2`, `/cam3` ...

---

## 9. 앱 실행

1. `Assets/Models/yolov8n.onnx` 파일 확인
2. Visual Studio에서 빌드 후 실행
3. **Snapshot 탭**: 이미지 파일 열기 → 검출 버튼
4. **RTSP 탭**: 이름·URL·디바이스 입력 → `+ 추가` → `▶ 시작`
   - 여러 개 한 번에: `📋 일괄 추가` → URL 줄당 하나씩 붙여넣기
   - 전체 시작/중지: 하단 `▶ 전체 시작` / `⏸ 전체 중지`
