# Vision.LiveStream.Inference

WPF (.NET 10) + ONNX Runtime 기반 **실시간 RTSP 영상 객체 검출** 학습 프로젝트.
선행 프로젝트 `Vision.OnnxTester` (정적 이미지 + YOLOv8) 의 검출 엔진을 베이스로,
RTSP 스트림을 받아 실시간으로 추론하고 화면에 박스를 그리는 것이 목표.

> **학습 목표 (Step 4):** 영상 수신 스레드 / AI 추론 스레드 / UI 렌더링(Dispatcher) 스레드를
> 완벽히 분리해서, 어느 한쪽이 막혀도 화면이 끊기지 않게 만든다.

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
Vision.LiveStream.Inference/                 <-- 저장소 루트
├── README.md
├── Tester/                                  <-- 가상 CCTV 환경
│   └── cameraTest/
│       ├── Video1.mp4 / Video2.mp4         샘플 CCTV 영상
│       ├── run_cameras_tcp.bat             TCP 모드 다중 카메라 실행
│       └── run_cameras_udp.bat             UDP 모드 다중 카메라 실행
└── Vision.LiveStream.Inference/             <-- 솔루션
    └── Vision.LiveStream.Inference/
        ├── Assets/Models/yolov8n.onnx
        ├── Common/
        │   ├── BaseViewModel.cs
        │   ├── RelayCommand.cs
        │   ├── AsyncRelayCommand.cs
        │   └── InverseBoolConverter.cs
        ├── Models/
        │   └── Detection.cs               검출 결과 1건 (박스 좌표, 클래스, 신뢰도)
        ├── Services/
        │   ├── CocoLabels.cs
        │   ├── Rtsp/
        │   │   ├── RtspFrame.cs           디코딩된 프레임 1개 (BGR byte[])
        │   │   ├── RtspFrameSource.cs     RTSP 수신 + 1슬롯 최신 프레임 채널
        │   │   ├── RtspFrameDetector.cs   RTSP 도메인 추론 어댑터
        │   │   └── IRtspFrameDetector.cs
        │   ├── Snapshot/
        │   │   ├── SnapshotDetector.cs    정적 이미지 추론 어댑터
        │   │   └── ISnapshotDetector.cs
        │   └── Yolo/
        │       ├── YoloPreprocessor.cs    letterbox + 정규화 + CHW 텐서 변환
        │       ├── YoloInferenceEngine.cs ONNX 세션 실행 + NMS 후처리
        │       └── LetterboxResult.cs     텐서 + 좌표 역변환 메타정보
        ├── ViewModels/
        │   ├── ShellViewModel.cs
        │   ├── SnapshotViewModel.cs
        │   └── RtspViewModel.cs
        └── MainWindow.xaml(.cs)
```

> 직접 받아야 하는 것: `mediamtx.exe`, `ffmpeg.exe`

---

## 3. 스레드 구조 (Step 4 핵심)

```
캡처 스레드 (전용 Thread)
  └─ VideoCapture.Read() 블로킹 루프
       ├─ FrameCaptured 이벤트 → Dispatcher.BeginInvoke(Render) → WriteableBitmap 갱신 (영상)
       └─ Channel.TryWrite → 1슬롯 DropOldest 큐 (추론용)

추론 Task (ThreadPool)
  └─ Channel.ReadAllAsync → 최신 프레임 1장씩
       └─ DetectAsync → Task.Run
            ├─ YoloPreprocessor: BGR byte[] → [1,3,640,640] 텐서
            ├─ YoloInferenceEngine: ONNX Run → [1,84,8400] 출력 → NMS
            └─ Dispatcher.BeginInvoke(Background) → Detections 컬렉션 갱신 (박스)

UI 스레드 (WPF)
  ├─ WriteableBitmap.WritePixels → 영상 렌더링
  └─ ItemsControl + DataTemplate → Canvas 위 Rectangle 박스 자동 생성
```

**설계 포인트**
- 캡처 스레드가 ThreadPool이 아닌 전용 Thread인 이유: `VideoCapture.Read()`가 블로킹 호출이라 ThreadPool 스레드를 영구 점유하면 다른 Task들이 굶음
- 1슬롯 `DropOldest` 채널: 추론이 느려도 항상 최신 프레임만 처리 → 영상과 박스 간 시간차 최소화
- `BeginInvoke` (비동기): 캡처/추론 스레드가 UI를 기다리지 않아 영상 끊김 없음

---

## 4. 추론 디바이스 선택 (CPU / DirectML / CUDA)

앱 실행 시 RTSP 탭에서 연결 전에 선택합니다.

| 옵션 | 설명 | 추가 설치 |
|---|---|---|
| **CPU** | 항상 사용 가능. 200~500ms/frame | 없음 |
| **DirectML** | Windows 내장 DirectX 12 ML. 기본값 | 없음 |
| **CUDA** | NVIDIA CUDA 가속. 가장 빠름 | CUDA Toolkit + cuDNN 필요 |

> 초기화 실패한 옵션은 경고 팝업 후 자동으로 CPU로 폴백됩니다.

---

## 5. 추론 패키지 전환 방법

### DirectML (기본값)

```xml
<!-- Vision.LiveStream.Inference.csproj -->
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.25.1" />
```

### CUDA

```xml
<!-- Vision.LiveStream.Inference.csproj -->
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.25.1" />
```

> DirectML ↔ CUDA 패키지는 동시에 설치 불가합니다 (같은 `onnxruntime.dll` 이름 충돌).
> csproj에서 한 줄씩 교체 후 빌드하면 NuGet이 자동으로 DLL을 교체합니다.

---

## 6. CUDA 환경 구축 (선택)

CUDA 옵션을 사용하려면 아래 두 가지를 설치해야 합니다.

### Step 1. CUDA Toolkit 설치

```
winget install Nvidia.CUDA
```

또는 NVIDIA 공식 사이트에서 직접 다운로드:
- https://developer.nvidia.com/cuda-downloads
- Windows / x86_64 선택 후 설치 (~3GB)

### Step 2. cuDNN 설치

- https://developer.nvidia.com/cudnn-downloads
- Windows / x86_64 / Tarball 다운로드
- 압축 해제 후 `bin/` 폴더의 DLL들을 CUDA 설치 경로에 복사

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin\
```

### Step 3. csproj 패키지 교체

위 §5 참고.

### 설치 확인

```powershell
nvcc --version   # CUDA 버전 확인
nvidia-smi       # GPU 상태 확인
```

---

## 7. 가상 CCTV 환경 구축

### Step 1. MediaMTX (RTSP 서버) 다운로드 + 실행

- 다운로드: https://github.com/bluenviron/mediamtx/releases
- `mediamtx_vX.X.X_windows_amd64.zip` 압축 해제 후 `mediamtx.exe` 실행
- 성공 로그: `[RTSP] listener opened on :8554`
- 이 콘솔 창은 켜둘 것 (닫으면 서버 종료)

### Step 2. FFmpeg (송출기) 다운로드

- https://www.gyan.dev/ffmpeg/builds/ → `ffmpeg-master-latest-win64-gpl.zip`
- `bin/ffmpeg.exe` 를 `Tester/cameraTest/ffmpeg.exe` 위치에 배치

### Step 3. 단일 카메라 송출

```cmd
cd Tester/cameraTest
ffmpeg -re -stream_loop -1 -i Video1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
```

| 옵션 | 의미 |
|---|---|
| `-re` | 원본 프레임레이트로 재생 (실시간 흉내) |
| `-stream_loop -1` | 무한 반복 |
| `-c copy` | 재인코딩 없이 패킷만 복사 → CPU 거의 안 씀 |

### Step 4. 다중 카메라 일괄 실행

`Tester/cameraTest/` 폴더의 배치 파일 사용:

- `run_cameras_tcp.bat` — RTSP 트랜스포트 TCP 강제 (방화벽/NAT 환경)
- `run_cameras_udp.bat` — 기본 UDP 트랜스포트

송출 주소: `rtsp://localhost:8554/cam1`, `/cam2`, `/cam3` ...

---

## 8. 앱 실행

1. `Assets/Models/yolov8n.onnx` 파일이 있는지 확인
2. Visual Studio에서 빌드 후 실행
3. **Snapshot 탭**: 이미지 파일 열기 → 검출 버튼
4. **RTSP 탭**: URL 입력 → CPU / DirectML / CUDA 선택 → 연결
