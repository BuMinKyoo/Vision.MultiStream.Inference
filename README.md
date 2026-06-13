# Vision.LiveStream.Inference

<br/>

<img width="1918" height="1031" alt="image" src="https://github.com/user-attachments/assets/f6024d39-4179-4e19-8c54-374b610f3464" />

<br/>

WPF (.NET 10) + ONNX Runtime 기반 **실시간 RTSP 영상 객체 검출** 학습 프로젝트.
선행 프로젝트 [`Vision.OnnxTester`](https://github.com/) (정적 이미지 + YOLOv8) 의 검출 엔진을 베이스로,
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
| UI | WPF |

### NuGet 패키지 (이미 csproj 에 포함됨)

| 패키지 | 용도 |
|---|---|
| `Microsoft.ML.OnnxRuntime` | ONNX 모델 추론 (CPU). GPU 쓰려면 `.Gpu` 로 교체 |
| `SixLabors.ImageSharp` | 이미지 픽셀 조작 (리사이즈, 정규화, HWC→CHW) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` | RTSP `VideoCapture` 로 프레임 수신 |

---

## 2. 폴더 구조

```
Vision.LiveStream.Inference/                 <-- 저장소 루트
├── README.md
├── Tester/                                  <-- 가상 CCTV 환경
│   └── cameraTest/
│       ├── Video1.mp4 / Video2.mp4         샘플 CCTV 영상 (저장소 포함)
│       ├── (Video3.mp4)                    100MB 초과로 직접 추가 — gitignore
│       ├── (ffmpeg.exe)                    별도 다운로드 필요 — gitignore
│       ├── run_cameras_tcp.bat             TCP 모드 다중 카메라 실행
│       └── run_cameras_udp.bat             UDP 모드 다중 카메라 실행
└── Vision.LiveStream.Inference/             <-- 솔루션 + WPF 프로젝트
    └── Vision.LiveStream.Inference/
        ├── Assets/
        │   ├── Models/yolov8n.onnx, yolov8n.pt
        │   └── TestImages/{bus,zidane}.jpg            스냅샷 동작 확인용
        ├── Common/   RelayCommand, AsyncRelayCommand, BaseViewModel
        ├── Models/   Detection
        ├── Services/
        │   ├── CocoLabels.cs                          80개 라벨 (도메인 무관 공용)
        │   ├── Yolo/                                  ★ 공통 추론 엔진
        │   │   ├── LetterboxResult.cs                   전처리 결과 DTO
        │   │   ├── YoloPreprocessor.cs                  string / byte[] 두 입력 → tensor
        │   │   └── YoloInferenceEngine.cs               tensor → Detections (Run + NMS)
        │   ├── Snapshot/                              ★ 정적 이미지 도메인
        │   │   ├── ISnapshotDetector.cs
        │   │   └── SnapshotDetector.cs                  파일 경로 → 결과
        │   └── Rtsp/                                  ★ RTSP 도메인
        │       ├── RtspFrame.cs                         BGR 프레임 DTO
        │       ├── RtspFrameSource.cs                   VideoCapture + 1슬롯 latest-only
        │       ├── IRtspFrameDetector.cs
        │       └── RtspFrameDetector.cs                 byte[] BGR → 결과
        ├── ViewModels/
        │   ├── ShellViewModel.cs                      탭 두 개 컨테이너
        │   ├── SnapshotViewModel.cs                   스냅샷 탭 상태
        │   └── RtspViewModel.cs                       RTSP 탭 상태 + 스레드 분리 로직
        ├── MainWindow.xaml(.cs)                       TabControl: 스냅샷 / RTSP
        └── App.xaml(.cs)
```

> 저장소에 포함된 것: 샘플 영상 2개(`Video1.mp4`, `Video2.mp4`), 배치 파일, ONNX 모델.
> 직접 받아야 하는 것: `mediamtx.exe`, `ffmpeg.exe`, (선택) `Video3.mp4`.
> 모두 아래 §3 절차대로 받아서 `Tester/` 하위에 배치하면 됨.

---

## 3. 가상 CCTV 환경 구축 (간추림)

> 아래 절차는 PC 한 대 안에서 **RTSP 서버 → 송출기 → 클라이언트** 구조를 그대로 재현하는 흐름.

### Step 1. MediaMTX (RTSP 서버) 다운로드 + 실행

PC 를 RTSP 분배기로 만들어 주는 서버.

- 다운로드: <https://github.com/bluenviron/mediamtx/releases> 에서 `mediamtx_vX.X.X_windows_amd64.zip`
- 압축을 풀어 `Tester/mediamtx_v.../` 형태로 저장소 안에 두면 편함 (gitignore 처리됨)
- 실행: `mediamtx.exe` 더블클릭
- 성공 로그: `[RTSP] listener opened on :8554`
- ⚠ 이 콘솔 창은 **끄지 말고 켜둘 것** (서버가 죽음)

### Step 2. FFmpeg (송출기) 다운로드

mp4 파일을 디먹싱 → RTSP 패킷으로 다시 먹싱해서 서버로 쏴주는 도구.

- 다운로드: <https://www.gyan.dev/ffmpeg/builds/> → `ffmpeg-master-latest-win64-gpl.zip`
- 압축의 `bin/ffmpeg.exe` 만 꺼내서 `Tester/cameraTest/ffmpeg.exe` 위치에 둘 것
  (배치 파일들이 같은 폴더에 있는 `ffmpeg` 를 호출함)

### Step 3. 단일 카메라 송출 (수동 명령어)

`Tester/cameraTest/` 안에서 cmd 또는 PowerShell 열고:

```cmd
ffmpeg -re -stream_loop -1 -i Video1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
```

옵션 의미

- `-re` : 원본 프레임레이트로 재생 (실시간 흉내)
- `-stream_loop -1` : 무한 반복
- `-c copy` : 재인코딩 없이 패킷만 복사 → CPU 거의 안 씀 (스트림 카피의 위력)
- `-f rtsp rtsp://localhost:8554/cam1` : RTSP 로 서버에 송출

`frame=... fps=...` 가 쭉 올라오면 정상.

### Step 4. VLC 로 수신 검증

C# 앱을 짜기 전에 상용 플레이어로 먼저 받아보자.

- VLC: <https://www.videolan.org/>
- VLC 실행 → **미디어 → 네트워크 스트림 열기** → `rtsp://localhost:8554/cam1` 입력 → 재생
- 영상이 뜨면 환경 구축 OK

---

## 4. 다중 카메라 일괄 실행 — `run_cameras_tcp.bat` / `run_cameras_udp.bat`

여러 채널(cam1, cam2, cam3 …) 을 한 번에 띄우려고 만든 배치 파일.
`Tester/cameraTest/` 폴더 안에 있는 `Video*.mp4` 파일을 자동으로 카운트해서
같은 개수만큼 `cam1`, `cam2`, `cam3` … 로 송출한다.

### 사용 방법

1. `mediamtx.exe` 가 떠 있는 상태인지 먼저 확인 (Step 1 참고)
2. `Tester/cameraTest/` 폴더로 이동
3. 둘 중 하나 더블클릭
   - `run_cameras_tcp.bat` — RTSP 트랜스포트를 **TCP** 로 강제
   - `run_cameras_udp.bat` — 기본(UDP) 트랜스포트 사용
4. 콘솔에 `Found N video files. Starting N cameras in background...` 출력
5. **창은 끄지 말 것** — 닫으면 송출도 끊김
6. 종료할 때는 콘솔에서 **아무 키나 누르면** `taskkill /F /IM ffmpeg.exe` 로 일괄 종료

### 송출되는 RTSP 주소

`Video1.mp4` → `rtsp://localhost:8554/cam1`
`Video2.mp4` → `rtsp://localhost:8554/cam2`
`Video3.mp4` → `rtsp://localhost:8554/cam3`
… (파일 개수만큼)

### TCP 와 UDP 의 차이 (간단히)

| 모드 | 특징 | 언제 쓰나 |
|---|---|---|
| **UDP** (`run_cameras_udp.bat`) | 기본값. 빠르지만 패킷 유실 가능 | LAN 내부 / 지연이 우선일 때 |
| **TCP** (`run_cameras_tcp.bat`) | `-rtsp_transport tcp` 강제. 손실에 강하고 NAT/방화벽에 잘 통과 | 네트워크가 불안정 / 영상 깨짐 발생 시 |

> 둘 다 동작이 안 되면 보통 **MediaMTX 가 안 떠있거나** Windows 방화벽이 8554 를 막은 경우.

### 동작 원리 (배치 파일 내부)

- `Video*.mp4` 패턴으로 파일 카운트
- `FOR /L` 루프로 `start /B` 해서 ffmpeg 를 백그라운드 실행 (창 안 뜸)
- 로그는 `> NUL 2>&1` 로 휴지통행 (콘솔 깨끗)
- 종료 키 입력 시 `taskkill /F /IM ffmpeg.exe` 로 모든 ffmpeg 프로세스 정리

---

## 5. 앱 실행 — 스냅샷 / RTSP 두 가지 모드

WPF 앱은 상단 탭으로 두 시나리오를 분리.

### 스냅샷 탭 (Step 3 영역)

`Assets/TestImages/bus.jpg` 같은 정적 이미지 한 장에서 객체 검출.

1. [이미지 열기...] → 파일 선택
2. [객체 검출] → 박스 + 라벨 + 신뢰도 표시

> 추론 엔진 자체 검증용. 모델 파일 / Onnx Runtime 환경이 정상 동작하는지 가장 빠르게 확인하는 경로.

### RTSP 탭 (Step 4 영역)

§3·§4 절차로 띄운 RTSP 스트림(`rtsp://localhost:8554/cam1` 등)을 받아 실시간 추론.

1. URL 입력 (기본값 `rtsp://localhost:8554/cam1`)
2. [연결] → 영상 표시 시작 + 박스 오버레이 (노란색)
3. 상단에 **화면 FPS** / **추론 FPS** 두 값이 따로 표시됨
4. [중지] 로 끊기

#### 스레드 분리 구조 (Step 4 핵심)

```
[VideoCapture Thread]                            [Inference Task]               [UI Dispatcher]
RtspFrameSource.CaptureLoop                                                      WriteableBitmap.WritePixels
  ↓ (모든 프레임)                                                                 Detections.Add(...)
  ├── FrameCaptured 이벤트 ────────────────────────────────────────────────────► 영상 표시 (Render 우선)
  │
  └── Channel(DropOldest, 1) ───────► Reader.ReadAllAsync ─► Detect ─────────► 박스 갱신 (Background 우선)
```

- 영상 표시는 추론을 기다리지 않음 → "**화면 FPS ≈ 카메라 FPS**" 유지
- 추론은 1슬롯 latest-only 큐로 받아 자기 페이스대로 처리 → "**추론 FPS** 는 모델 속도에 따름"
- 두 FPS 값이 다르면 분리가 잘 동작 중 (CPU YOLOv8n 기준 화면 30 / 추론 5~15 정도)

### 동작 검증 시나리오

1. `mediamtx.exe` 실행 → `[RTSP] listener opened on :8554`
2. `Tester/cameraTest/run_cameras_udp.bat` 실행 → `Found N video files...`
3. VLC 로 먼저 수신 확인 (선택)
4. WPF 앱 실행 → RTSP 탭 → URL 입력 → [연결]
5. 차/사람 등이 나오는 영상이면 박스가 따라오는지 확인

---

## 6. 다음 작업 (Step 5 — 다중 채널)

- [ ] 단일 채널 검증 후, `RtspViewModel` 을 N개 인스턴스화하는 그리드 UI (4ch/8ch)
- [ ] `YoloInferenceEngine` 1개를 N개 채널이 공유 (직렬 추론 큐 검토)
- [ ] dotMemory / dotTrace 로 GC 스파이크 / 메모리 폭발 지점 측정
- [ ] 매 프레임 `byte[]` 새 할당 → 풀링 / 재사용으로 LOH 부담 줄이기
- [ ] 끊김 자동 재연결 정책

---
