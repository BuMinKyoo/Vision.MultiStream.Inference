using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using SharpGen.Runtime;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vortice.D3DCompiler;
using Vortice.Direct3D9;

namespace Vision.MultiStream.Inference.Services.Direct3D
{
    /// <summary>UI 가 컴포지터에게 알려주는 슬롯별 타일 위치(컴포지터 surface 픽셀 좌표).</summary>
    public readonly record struct TileRect(int SlotId, int X, int Y, int Width, int Height);

    /// <summary>
    /// [Step 3] 여러 스트림의 YUV 프레임을 한 D3D9Ex 디바이스로 합성해 단일 D3DImage 로 표시.
    ///
    ///   - 스트림마다 RegisterStream → slotId. VideoRenderer 가 SubmitFrame(slotId, yuvFrame) 로 최신 프레임 제출.
    ///   - 전용 렌더 스레드가: 슬롯별 YUV 텍스처 업로드 → 타일 위치에 YUV→RGB 셰이더로 드로우 →
    ///     더블버퍼 RT 교체 → UI 에서 프레임당 1회만 D3DImage present.
    ///   - 모든 D3D 리소스(텍스처/서피스) 생성·업로드·드로우는 렌더 스레드 단독 → 디바이스 멀티스레드 경쟁 없음.
    ///
    /// 현재(Step 3): 타일 배치는 임시 auto-grid(정사각 분할). 실제 레이아웃 동기화(SetLayout)는 이후 단계.
    /// </summary>
    public sealed class StreamCompositor : IDisposable
    {
        private const string PixelShaderSource = @"
sampler ySampler : register(s0);
sampler uSampler : register(s1);
sampler vSampler : register(s2);

struct PS_INPUT
{
    float2 yuv0 : TEXCOORD0;
    float2 yuv1 : TEXCOORD1;
    float2 yuv2 : TEXCOORD2;
};

float4 main(PS_INPUT input) : COLOR0
{
    float y = tex2D(ySampler, input.yuv0).r;
    float u = tex2D(uSampler, input.yuv1).r - 0.5;
    float v = tex2D(vSampler, input.yuv2).r - 0.5;
    float3 rgb;
    rgb.r = y + 1.402 * v;
    rgb.g = y - 0.344136 * u - 0.714136 * v;
    rgb.b = y + 1.772 * u;
    return float4(saturate(rgb), 1.0);
}";

        // 한 스트림의 표시 상태. 텍스처 생성/업로드/해제는 모두 렌더 스레드에서만 한다.
        private sealed class Slot
        {
            public RtspYuvFrame? Latest;        // 생산자(SubmitFrame)와 소비자(렌더 스레드)가 atomic 교체
            public volatile bool RemoveRequested;

            public IDirect3DTexture9? YTex;
            public IDirect3DTexture9? UTex;
            public IDirect3DTexture9? VTex;
            public int TexWidth;
            public int TexHeight;

            public bool HasTextures => YTex != null;
        }

        private readonly Dispatcher _dispatcher;
        private readonly D3DImage _image;
        private readonly int _width;
        private readonly int _height;
        private readonly IntPtr _windowHandle;

        private readonly ConcurrentDictionary<int, Slot> _slots = new();
        private int _nextSlotId;

        private IDirect3D9Ex? _d3d;
        private IDirect3DDevice9Ex? _device;
        private IDirect3DPixelShader9? _pixelShader;
        // 렌더 스레드 전용 오프스크린 RT (그리는 대상).
        private IDirect3DSurface9? _offscreenRT;
        // D3DImage 가 읽는 backbuffer. 첫 present 시 SetBackBuffer 1회, 이후 안 바꿈.
        // 매 프레임 UI Lock 안에서 StretchRect 로 오프스크린의 내용을 복사한다.
        private IDirect3DSurface9? _backbufferSurface;
        private bool _backbufferAttached;
        // UI 가 SetLayout 으로 알려주는 슬롯별 타일 rect. 렌더 스레드는 매 프레임 이 참조를 읽어 사용.
        // 참조 자체는 atomic 으로 교체된다(읽는 쪽은 그대로 들고 한 프레임 처리).
        private volatile TileRect[]? _layout;

        private Thread? _renderThread;
        private volatile bool _running;

        public StreamCompositor(int width, int height)
        {
            _dispatcher = Application.Current.Dispatcher;
            _image = new D3DImage();
            _width = width;
            _height = height;
            _windowHandle = GetMainWindowHandle();
        }

        public ImageSource Image => _image;
        public int Width => _width;
        public int Height => _height;

        // ===== 스트림 등록/제출 API (임의 스레드에서 호출 가능) =====

        public int RegisterStream()
        {
            int id = Interlocked.Increment(ref _nextSlotId);
            _slots[id] = new Slot();
            return id;
        }

        public void UnregisterStream(int slotId)
        {
            if (_slots.TryGetValue(slotId, out Slot? slot))
            {
                slot.RemoveRequested = true; // 실제 텍스처 해제는 렌더 스레드가 처리
            }
        }

        /// <summary>UI 가 호출. 슬롯별 타일 위치를 알려준다(컴포지터 surface 좌표). 가벼운 atomic 참조 교체.</summary>
        public void SetLayout(IReadOnlyList<TileRect> layout)
        {
            var arr = new TileRect[layout.Count];
            for (int i = 0; i < layout.Count; i++)
            {
                arr[i] = layout[i];
            }
            _layout = arr;
        }

        /// <summary>VideoRenderer 가 호출. 최신 프레임으로 교체하고 직전 프레임은 풀에 반납.</summary>
        public void SubmitFrame(int slotId, RtspYuvFrame frame)
        {
            if (!_running || !_slots.TryGetValue(slotId, out Slot? slot) || slot.RemoveRequested)
            {
                frame.Dispose();
                return;
            }
            RtspYuvFrame? dropped = Interlocked.Exchange(ref slot.Latest, frame);
            dropped?.Dispose();
        }

        // ===== 수명 =====

        /// <summary>
        /// 디바이스를 동기로 초기화한 뒤 렌더 스레드 시작. 디바이스 생성 실패(예: GPU/D3D9 없음)면 throw 한다.
        /// 호출자(MainWindow)가 catch 하여 컴포지터 없이 per-stream 폴백 경로로 가야 한다.
        /// </summary>
        public void Start()
        {
            if (_running)
            {
                return;
            }

            try
            {
                InitializeDevice();
            }
            catch
            {
                ReleaseAll();
                throw;
            }

            _running = true;
            _renderThread = new Thread(RenderLoop)
            {
                IsBackground = true,
                Name = "StreamCompositor"
            };
            _renderThread.Start();
        }

        public void Stop()
        {
            if (!_running)
            {
                return;
            }
            _running = false;
            _renderThread?.Join(TimeSpan.FromSeconds(2));
            _renderThread = null;
        }

        private void RenderLoop()
        {
            try
            {
                // 디바이스/리소스는 Start() 에서 동기 초기화됨. 여기선 루프만.
                while (_running)
                {
                    long start = System.Diagnostics.Stopwatch.GetTimestamp();

                    // 오프스크린 RT 에 전 슬롯을 그림.
                    DrawAllSlots();

                    // UI 스레드: Lock 안에서 오프스크린 → backbuffer 로 StretchRect 복사 후 AddDirtyRect.
                    // backbuffer 는 한 번 SetBackBuffer 하고 이후 안 바꿈 → WPF 가 항상 같은 서피스를 봄.
                    _dispatcher.Invoke(PresentToBackbuffer);

                    double elapsedMs = (System.Diagnostics.Stopwatch.GetTimestamp() - start)
                        * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                    int sleep = 16 - (int)elapsedMs; // 1000ms / 60fps ≈ 16.67ms. 한 프레임에 허용된 시간 예산이 ~16ms
                    if (sleep > 0)
                    {
                        Thread.Sleep(sleep);
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[StreamCompositor] render loop error: {ex}");
            }
            finally
            {
                ReleaseAll();
            }
        }

        // 렌더 스레드: 슬롯 정리 → 클리어 → 슬롯별 업로드+타일 드로우.
        private void DrawAllSlots()
        {
            // 제거 요청된 슬롯 정리(텍스처/잔여 프레임 해제는 여기 렌더 스레드에서).
            foreach (KeyValuePair<int, Slot> kv in _slots)
            {
                if (kv.Value.RemoveRequested && _slots.TryRemove(kv.Key, out Slot? removed))
                {
                    ReleaseSlot(removed);
                }
            }

            _device!.SetRenderTarget(0, _offscreenRT!);
            _device.Clear(ClearFlags.Target, new Vortice.Mathematics.Color((byte)0, (byte)0, (byte)0, (byte)255), 1f, 0);

            // UI 가 알려준 타일 위치. 없으면 그릴 게 없음 → 검정 배경만.
            TileRect[]? layout = _layout;
            if (layout == null || layout.Length == 0)
            {
                return;
            }

            _device.BeginScene();
            try
            {
                foreach (TileRect tile in layout)
                {
                    if (!_slots.TryGetValue(tile.SlotId, out Slot? slot))
                    {
                        continue;
                    }

                    // 최신 프레임이 있으면 업로드 후 반납. 없으면 직전 텍스처 그대로.
                    RtspYuvFrame? frame = Interlocked.Exchange(ref slot.Latest, null);
                    if (frame != null)
                    {
                        try
                        {
                            // 이 스트림의 원본 픽셀을 담을 GPU 텍스처 만들기
                            EnsureSlotTextures(slot, frame.Width, frame.Height);

                            // 원본 YUV 데이터를 텍스처에 memcpy
                            UploadFrame(slot, frame);
                        }
                        finally
                        {
                            frame.Dispose();
                        }
                    }

                    if (!slot.HasTextures)
                    {
                        continue;
                    }

                    // 1920×1080짜리 텍스처를 426×240 셀에 quad로 그려라 (GPU가 다운스케일)
                    DrawTile(slot, tile.X, tile.Y, tile.Width, tile.Height);
                }
            }
            finally
            {
                _device.EndScene();
            }
        }

        private void DrawTile(Slot slot, int x, int y, int w, int h)
        {
            for (int s = 0; s < 3; s++)
            {
                _device!.SetSamplerState(s, SamplerState.MinFilter, (int)TextureFilter.Linear);
                _device.SetSamplerState(s, SamplerState.MagFilter, (int)TextureFilter.Linear);
                _device.SetSamplerState(s, SamplerState.AddressU, (int)TextureAddress.Clamp);
                _device.SetSamplerState(s, SamplerState.AddressV, (int)TextureAddress.Clamp);
            }

            _device!.SetTexture(0, slot.YTex);
            _device.SetTexture(1, slot.UTex);
            _device.SetTexture(2, slot.VTex);
            _device.PixelShader = _pixelShader;
            _device.VertexFormat = VertexFormat.PositionRhw | VertexFormat.Texture3;

            float left = x - 0.5f;
            float top = y - 0.5f;
            float right = x + w - 0.5f;
            float bottom = y + h - 0.5f;

            Span<Vertex> vertices =
            [
                new Vertex(left, top, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f),
                new Vertex(right, top, 0f, 1f, 1f, 0f, 1f, 0f, 1f, 0f),
                new Vertex(left, bottom, 0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f),
                new Vertex(right, bottom, 0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f)
            ];

            unsafe
            {
                fixed (Vertex* p = vertices)
                {
                    _device.DrawPrimitiveUP(PrimitiveType.TriangleStrip, 2, (IntPtr)p, (uint)Marshal.SizeOf<Vertex>());
                }
            }
        }

        private void EnsureSlotTextures(Slot slot, int width, int height)
        {
            if (slot.HasTextures && slot.TexWidth == width && slot.TexHeight == height)
            {
                return;
            }

            ReleaseSlotTextures(slot);
            slot.YTex = _device!.CreateTexture((uint)width, (uint)height, 1, Usage.Dynamic, Format.L8, Pool.Default);
            slot.UTex = _device.CreateTexture((uint)(width / 2), (uint)(height / 2), 1, Usage.Dynamic, Format.L8, Pool.Default);
            slot.VTex = _device.CreateTexture((uint)(width / 2), (uint)(height / 2), 1, Usage.Dynamic, Format.L8, Pool.Default);
            slot.TexWidth = width;
            slot.TexHeight = height;
        }

        private static void UploadFrame(Slot slot, RtspYuvFrame frame)
        {
            int chromaHeight = (frame.Height + 1) / 2;
            UploadPlane(slot.YTex!, frame.YPlane, frame.YStride, frame.Height);
            UploadPlane(slot.UTex!, frame.UPlane, frame.UStride, chromaHeight);
            UploadPlane(slot.VTex!, frame.VPlane, frame.VStride, chromaHeight);
        }

        private static unsafe void UploadPlane(IDirect3DTexture9 texture, byte[] source, int sourceStride, int rows)
        {
            LockedRectangle rect = texture.LockRect(0, LockFlags.Discard);
            try
            {
                int copyStride = Math.Min(sourceStride, rect.Pitch);
                fixed (byte* srcBase = source)
                {
                    byte* src = srcBase;
                    byte* dst = (byte*)rect.DataPointer;
                    for (int y = 0; y < rows; y++)
                    {
                        Buffer.MemoryCopy(src, dst, rect.Pitch, copyStride);
                        src += sourceStride;
                        dst += rect.Pitch;
                    }
                }
            }
            finally
            {
                texture.UnlockRect(0);
            }
        }

        // UI 스레드(Invoke 로 호출): 오프스크린의 그림을 backbuffer 로 복사 후 dirty 표시(프레임당 1회).
        //   - SetBackBuffer 는 첫 호출에만 (이후 같은 서피스를 계속 씀 → 깜박임 없음)
        //   - StretchRect 가 Lock 안에서 일어나므로 WPF 의 compose 와 안 부딪힘
        private void PresentToBackbuffer()
        {
            if (_device == null || _offscreenRT == null || _backbufferSurface == null)
            {
                return;
            }
            _image.Lock();
            try
            {
                if (!_backbufferAttached)
                {
                    // 표시 원천을 이 서피스로 등록
                    _image.SetBackBuffer(D3DResourceType.IDirect3DSurface9, _backbufferSurface.NativePointer);
                    _backbufferAttached = true;
                }
                var full = new Vortice.Direct3D9.Rect(0, 0, _width, _height);

                // 그 서피스의 픽셀 내용을 새로 채움
                _device.StretchRect(_offscreenRT, full, _backbufferSurface, full, TextureFilter.None);

                // 그 서피스의 이 영역이 바뀌었으니 다시 읽어 그려
                _image.AddDirtyRect(new Int32Rect(0, 0, _width, _height));
            }
            finally
            {
                _image.Unlock();
            }
        }

        private void InitializeDevice()
        {
            _d3d = D3D9.Direct3DCreate9Ex();
            _device = _d3d.CreateDeviceEx(
                0,
                DeviceType.Hardware,
                _windowHandle,
                CreateFlags.HardwareVertexProcessing | CreateFlags.Multithreaded | CreateFlags.FpuPreserve,
                new PresentParameters
                {
                    BackBufferWidth = 1,
                    BackBufferHeight = 1,
                    BackBufferFormat = Format.A8R8G8B8,
                    BackBufferCount = 1,
                    SwapEffect = SwapEffect.Discard,
                    Windowed = true,
                    PresentationInterval = PresentInterval.Immediate
                });

            ReadOnlyMemory<byte> shaderBytes = Compiler.Compile(
                PixelShaderSource, "main", "Yuv420ToRgb", "ps_2_0",
                ShaderFlags.OptimizationLevel3, EffectFlags.None);
            _pixelShader = _device.CreatePixelShader(shaderBytes.Span);

            _offscreenRT = CreateRenderTarget();
            _backbufferSurface = CreateRenderTarget();
        }

        private IDirect3DSurface9 CreateRenderTarget()
        {
            return _device!.CreateRenderTargetEx(
                (uint)_width, (uint)_height, Format.A8R8G8B8,
                MultisampleType.None, 0, false, Usage.None);
        }

        private void ReleaseSlotTextures(Slot slot)
        {
            slot.VTex?.Dispose();
            slot.UTex?.Dispose();
            slot.YTex?.Dispose();
            slot.VTex = null;
            slot.UTex = null;
            slot.YTex = null;
        }

        private void ReleaseSlot(Slot slot)
        {
            ReleaseSlotTextures(slot);
            Interlocked.Exchange(ref slot.Latest, null)?.Dispose();
        }

        private void ReleaseAll()
        {
            foreach (KeyValuePair<int, Slot> kv in _slots)
            {
                ReleaseSlot(kv.Value);
            }
            _slots.Clear();

            _pixelShader?.Dispose();
            _backbufferSurface?.Dispose();
            _offscreenRT?.Dispose();
            _device?.Dispose();
            _d3d?.Dispose();
            _pixelShader = null;
            _backbufferSurface = null;
            _offscreenRT = null;
            _device = null;
            _d3d = null;
        }

        private static IntPtr GetMainWindowHandle()
        {
            Window? window = Application.Current?.MainWindow;
            return window == null ? IntPtr.Zero : new WindowInteropHelper(window).Handle;
        }

        public void Dispose()
        {
            Stop();
            _dispatcher.Invoke(() =>
            {
                _image.Lock();
                try
                {
                    _image.SetBackBuffer(D3DResourceType.IDirect3DSurface9, IntPtr.Zero);
                }
                finally
                {
                    _image.Unlock();
                }
            });
        }

        [StructLayout(LayoutKind.Sequential)]
        private readonly struct Vertex
        {
            public Vertex(float x, float y, float z, float rhw,
                float tu0, float tv0, float tu1, float tv1, float tu2, float tv2)
            {
                X = x; Y = y; Z = z; Rhw = rhw;
                Tu0 = tu0; Tv0 = tv0; Tu1 = tu1; Tv1 = tv1; Tu2 = tu2; Tv2 = tv2;
            }

            private readonly float X;
            private readonly float Y;
            private readonly float Z;
            private readonly float Rhw;
            private readonly float Tu0;
            private readonly float Tv0;
            private readonly float Tu1;
            private readonly float Tv1;
            private readonly float Tu2;
            private readonly float Tv2;
        }
    }
}
