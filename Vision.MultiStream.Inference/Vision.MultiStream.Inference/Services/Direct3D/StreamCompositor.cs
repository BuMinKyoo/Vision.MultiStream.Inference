using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.Direct3D9;
using Vortice.DXGI;
using Vortice.Mathematics;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vision.MultiStream.Inference.Services.Rtsp.Pipeline;
using D3D11Device = Vortice.Direct3D11.ID3D11Device;
using D3D11Context = Vortice.Direct3D11.ID3D11DeviceContext;
using D3D11Texture = Vortice.Direct3D11.ID3D11Texture2D;
using D9Format = Vortice.Direct3D9.Format;
using DxgiFormat = Vortice.DXGI.Format;
using Viewport = Vortice.Mathematics.Viewport;
using D9PresentParameters = Vortice.Direct3D9.PresentParameters;
using D9SwapEffect = Vortice.Direct3D9.SwapEffect;

namespace Vision.MultiStream.Inference.Services.Direct3D
{
    /// <summary>UI 가 컴포지터에게 알려주는 슬롯별 타일 위치(컴포지터 surface 픽셀 좌표).</summary>
    public readonly record struct TileRect(int SlotId, int X, int Y, int Width, int Height);

    /// <summary>
    /// [Step 3 / GPU 직결] 여러 스트림의 프레임을 하나의 D3D11 디바이스로 합성해 단일 D3DImage 로 표시.
    ///
    ///   - 디바이스는 <see cref="HwDeviceContext"/> 가 소유(디코더와 동일) → HW 디코드 NV12 텍스처를
    ///     CPU 다운로드 없이 GPU 안에서 슬롯 텍스처로 복사해 바로 샘플링한다.
    ///   - SW 스트림은 기존처럼 CPU YUV420P 평면을 슬롯 텍스처로 업로드한다.
    ///   - 렌더 스레드가: 슬롯별 텍스처 준비 → 타일 위치에 YUV→RGB 셰이더로 드로우 →
    ///     공유 RT 에 합성 → UI 에서 프레임당 1회 D3DImage present.
    ///   - immediate context 는 디코드 스레드와 공유하므로 사용 전후로 HwDeviceContext.Lock/Unlock.
    /// </summary>
    public sealed unsafe class StreamCompositor : IDisposable
    {
        // 공유 정점 셰이더: SV_VertexID 로 풀스크린 삼각형 생성(정점 버퍼 불필요). 뷰포트가 타일을 결정.
        private const string VertexShaderSource = @"
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VSOut main(uint id : SV_VertexID)
{
    VSOut o;
    o.uv  = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}";

        // uvScale: 디코드 텍스처의 coded 크기 중 실제 표시 영역 비율(padding 행 제외). SW 는 (1,1).
        private const string PixelShaderHeader = @"
cbuffer Params : register(b0) { float2 uvScale; float2 pad; };
SamplerState samp : register(s0);
float3 YuvToRgb(float y, float u, float v)
{
    u -= 0.5; v -= 0.5;
    float3 rgb;
    rgb.r = y + 1.402 * v;
    rgb.g = y - 0.344136 * u - 0.714136 * v;
    rgb.b = y + 1.772 * u;
    return saturate(rgb);
}";

        // HW: NV12 = Y(R8) + 인터리브 UV(R8G8)
        private const string PixelShaderNv12 = PixelShaderHeader + @"
Texture2D<float>  texY  : register(t0);
Texture2D<float2> texUV : register(t1);
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 c = uv * uvScale;
    float  y  = texY.Sample(samp, c);
    float2 uvv = texUV.Sample(samp, c);
    return float4(YuvToRgb(y, uvv.x, uvv.y), 1.0);
}";

        // SW: YUV420P = Y,U,V 각각 R8
        private const string PixelShaderYuv420 = PixelShaderHeader + @"
Texture2D<float> texY : register(t0);
Texture2D<float> texU : register(t1);
Texture2D<float> texV : register(t2);
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float2 c = uv * uvScale;
    float y = texY.Sample(samp, c);
    float u = texU.Sample(samp, c);
    float v = texV.Sample(samp, c);
    return float4(YuvToRgb(y, u, v), 1.0);
}";

        // 한 스트림의 표시 상태. 모든 텍스처 생성/복사/업로드/해제는 렌더 스레드에서만.
        private sealed class Slot
        {
            // 생산자(SubmitFrame/SubmitD3D11Frame)와 소비자(렌더 스레드)가 atomic 교체.
            public RtspYuvFrame? LatestYuv;
            public RtspD3D11Frame? LatestD3D11;
            public volatile bool RemoveRequested;

            // 슬롯 소유 텍스처 + SRV.
            // HW: NV12 단일 텍스처(코드 크기). SW: Y/U/V 3개.
            public D3D11Texture? Nv12Tex;
            public D3D11Texture? YTex;
            public D3D11Texture? UTex;
            public D3D11Texture? VTex;
            public ID3D11ShaderResourceView? Srv0; // HW:Y  SW:Y
            public ID3D11ShaderResourceView? Srv1; // HW:UV SW:U
            public ID3D11ShaderResourceView? Srv2; // SW:V
            public bool IsHardware;
            public int TexW;
            public int TexH;
            public float UScale = 1f;
            public float VScale = 1f;

            public bool HasContent => Srv0 != null;
        }

        private readonly Dispatcher _dispatcher;
        private readonly D3DImage _image;
        private readonly int _width;
        private readonly int _height;

        private readonly ConcurrentDictionary<int, Slot> _slots = new();
        private int _nextSlotId;

        // 공유 D3D11 디바이스(HwDeviceContext 소유) — 빌려 쓴다.
        private D3D11Device _device = null!;
        private D3D11Context _context = null!;

        private ID3D11VertexShader? _vs;
        private ID3D11PixelShader? _psNv12;
        private ID3D11PixelShader? _psYuv420;
        private ID3D11SamplerState? _sampler;
        private ID3D11Buffer? _paramsCb;

        // 더블버퍼: 렌더 스레드는 _offscreenTex(WPF 가 보지 않음) 에 그리고,
        // UI 스레드가 _image.Lock 안에서만 _sharedTex(= D3D9 백버퍼) 로 복사한다.
        // → 백버퍼는 Lock 안에서만 바뀌므로 WPF 가 그리다 만 프레임을 읽어 깜박이지 않는다.
        private D3D11Texture? _offscreenTex;
        private ID3D11RenderTargetView? _offscreenRtv;
        private D3D11Texture? _sharedTex;
        private IDirect3D9Ex? _d3d9;
        private IDirect3DDevice9Ex? _d3d9Device;
        private IDirect3DTexture9? _d3d9Tex;
        private IDirect3DSurface9? _d3d9Surface;
        private bool _backbufferAttached;

        private volatile TileRect[]? _layout;
        private Thread? _renderThread;
        private volatile bool _running;

        public StreamCompositor(int width, int height)
        {
            _dispatcher = Application.Current.Dispatcher;
            _image = new D3DImage();
            _width = width;
            _height = height;
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
                slot.RemoveRequested = true; // 실제 해제는 렌더 스레드가 처리
            }
        }

        public void SetLayout(IReadOnlyList<TileRect> layout)
        {
            var arr = new TileRect[layout.Count];
            for (int i = 0; i < layout.Count; i++)
            {
                arr[i] = layout[i];
            }
            _layout = arr;
        }

        /// <summary>SW 스트림: 최신 YUV420P 프레임으로 교체하고 직전 프레임은 풀에 반납.</summary>
        public void SubmitFrame(int slotId, RtspYuvFrame frame)
        {
            if (!_running || !_slots.TryGetValue(slotId, out Slot? slot) || slot.RemoveRequested)
            {
                frame.Dispose();
                return;
            }
            RtspYuvFrame? dropped = Interlocked.Exchange(ref slot.LatestYuv, frame);
            dropped?.Dispose();
        }

        /// <summary>HW 스트림: 최신 D3D11 GPU 프레임으로 교체하고 직전 프레임은 ref 반납(풀로).</summary>
        public void SubmitD3D11Frame(int slotId, RtspD3D11Frame frame)
        {
            if (!_running || !_slots.TryGetValue(slotId, out Slot? slot) || slot.RemoveRequested)
            {
                frame.Dispose();
                return;
            }
            RtspD3D11Frame? dropped = Interlocked.Exchange(ref slot.LatestD3D11, frame);
            dropped?.Dispose();
        }

        // ===== 수명 =====

        /// <summary>
        /// 디바이스/리소스를 동기 초기화한 뒤 렌더 스레드 시작. D3D11 디바이스 생성 실패 등으로
        /// 초기화 못 하면 throw → 호출자(MainWindow)가 컴포지터 없이 폴백 경로로 가야 한다.
        /// </summary>
        public void Start()
        {
            if (_running)
            {
                return;
            }

            try
            {
                InitializeResources();
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
                while (_running)
                {
                    long start = System.Diagnostics.Stopwatch.GetTimestamp();

                    DrawAllSlots();
                    _dispatcher.Invoke(PresentToImage);

                    double elapsedMs = (System.Diagnostics.Stopwatch.GetTimestamp() - start)
                        * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                    int sleep = 16 - (int)elapsedMs; // ~60fps 예산
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

        // 렌더 스레드: 슬롯 정리 → 클리어 → 슬롯별 준비 + 타일 드로우. immediate context 는 락으로 보호.
        private void DrawAllSlots()
        {
            foreach (KeyValuePair<int, Slot> kv in _slots)
            {
                if (kv.Value.RemoveRequested && _slots.TryRemove(kv.Key, out Slot? removed))
                {
                    ReleaseSlot(removed);
                }
            }

            HwDeviceContext.Lock();
            try
            {
                _context.OMSetRenderTargets(_offscreenRtv!);
                _context.ClearRenderTargetView(_offscreenRtv!, new Color4(0f, 0f, 0f, 1f));

                TileRect[]? layout = _layout;
                if (layout == null || layout.Length == 0)
                {
                    return;
                }

                _context.VSSetShader(_vs);
                _context.PSSetSampler(0, _sampler);
                _context.PSSetConstantBuffer(0, _paramsCb);
                _context.IASetInputLayout(null);
                _context.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);

                foreach (TileRect tile in layout)
                {
                    if (!_slots.TryGetValue(tile.SlotId, out Slot? slot))
                    {
                        continue;
                    }

                    UpdateSlot(slot);
                    if (!slot.HasContent)
                    {
                        continue;
                    }

                    DrawTile(slot, tile);
                }
                // 실제 GPU 제출(Flush)은 PresentToImage 의 복사 직후 1회만 한다.
            }
            finally
            {
                HwDeviceContext.Unlock();
            }
        }

        // 슬롯의 최신 프레임을 받아 슬롯 소유 텍스처를 갱신한다(없으면 직전 텍스처 유지 → 재드로우).
        private void UpdateSlot(Slot slot)
        {
            RtspD3D11Frame? hw = Interlocked.Exchange(ref slot.LatestD3D11, null);
            if (hw != null)
            {
                try
                {
                    UpdateHardwareSlot(slot, hw);
                }
                finally
                {
                    hw.Dispose(); // 복사 끝났으니 디코드 surface 즉시 반납.
                }
                return;
            }

            RtspYuvFrame? yuv = Interlocked.Exchange(ref slot.LatestYuv, null);
            if (yuv != null)
            {
                try
                {
                    UpdateSoftwareSlot(slot, yuv);
                }
                finally
                {
                    yuv.Dispose();
                }
            }
        }

        // HW: 디코드 NV12 텍스처 배열의 해당 슬라이스를 슬롯 소유 NV12 텍스처로 GPU→GPU 복사.
        private void UpdateHardwareSlot(Slot slot, RtspD3D11Frame frame)
        {
            var srcTex = new D3D11Texture(frame.Texture);
            srcTex.AddRef();
            try
            {
                Texture2DDescription srcDesc = srcTex.Description;
                int codedW = (int)srcDesc.Width;
                int codedH = (int)srcDesc.Height;

                if (!slot.IsHardware || slot.Nv12Tex == null || slot.TexW != codedW || slot.TexH != codedH)
                {
                    ReleaseSlotTextures(slot);
                    slot.IsHardware = true;
                    slot.TexW = codedW;
                    slot.TexH = codedH;

                    slot.Nv12Tex = _device.CreateTexture2D(new Texture2DDescription
                    {
                        Width = (uint)codedW,
                        Height = (uint)codedH,
                        MipLevels = 1,
                        ArraySize = 1,
                        Format = DxgiFormat.NV12,
                        SampleDescription = new SampleDescription(1, 0),
                        Usage = ResourceUsage.Default,
                        BindFlags = BindFlags.ShaderResource,
                        CPUAccessFlags = CpuAccessFlags.None,
                        MiscFlags = ResourceOptionFlags.None
                    });

                    slot.Srv0 = _device.CreateShaderResourceView(slot.Nv12Tex, new ShaderResourceViewDescription
                    {
                        Format = DxgiFormat.R8_UNorm,
                        ViewDimension = ShaderResourceViewDimension.Texture2D,
                        Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                    });
                    slot.Srv1 = _device.CreateShaderResourceView(slot.Nv12Tex, new ShaderResourceViewDescription
                    {
                        Format = DxgiFormat.R8G8_UNorm,
                        ViewDimension = ShaderResourceViewDimension.Texture2D,
                        Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
                    });
                }

                slot.UScale = codedW > 0 ? (float)frame.Width / codedW : 1f;
                slot.VScale = codedH > 0 ? (float)frame.Height / codedH : 1f;

                // 디코드 텍스처 배열의 arrayIndex 슬라이스(subresource = arrayIndex, mip 1개) → 슬롯 텍스처.
                _context.CopySubresourceRegion(slot.Nv12Tex!, 0, 0, 0, 0, srcTex, (uint)frame.ArrayIndex);
            }
            finally
            {
                srcTex.Dispose();
            }
        }

        // SW: YUV420P 평면을 슬롯 소유 Y/U/V(R8) 텍스처로 업로드.
        private void UpdateSoftwareSlot(Slot slot, RtspYuvFrame frame)
        {
            int w = frame.Width;
            int h = frame.Height;
            int cw = (w + 1) / 2;
            int ch = (h + 1) / 2;

            if (slot.IsHardware || slot.YTex == null || slot.TexW != w || slot.TexH != h)
            {
                ReleaseSlotTextures(slot);
                slot.IsHardware = false;
                slot.TexW = w;
                slot.TexH = h;
                slot.UScale = 1f;
                slot.VScale = 1f;

                slot.YTex = CreateDynamicR8(w, h);
                slot.UTex = CreateDynamicR8(cw, ch);
                slot.VTex = CreateDynamicR8(cw, ch);
                slot.Srv0 = _device.CreateShaderResourceView(slot.YTex, R8Srv());
                slot.Srv1 = _device.CreateShaderResourceView(slot.UTex, R8Srv());
                slot.Srv2 = _device.CreateShaderResourceView(slot.VTex, R8Srv());
            }

            UploadPlane(slot.YTex!, frame.YPlane, frame.YStride, w, h);
            UploadPlane(slot.UTex!, frame.UPlane, frame.UStride, cw, ch);
            UploadPlane(slot.VTex!, frame.VPlane, frame.VStride, cw, ch);
        }

        private D3D11Texture CreateDynamicR8(int w, int h)
        {
            return _device.CreateTexture2D(new Texture2DDescription
            {
                Width = (uint)w,
                Height = (uint)h,
                MipLevels = 1,
                ArraySize = 1,
                Format = DxgiFormat.R8_UNorm,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Dynamic,
                BindFlags = BindFlags.ShaderResource,
                CPUAccessFlags = CpuAccessFlags.Write,
                MiscFlags = ResourceOptionFlags.None
            });
        }

        private static ShaderResourceViewDescription R8Srv()
        {
            return new ShaderResourceViewDescription
            {
                Format = DxgiFormat.R8_UNorm,
                ViewDimension = ShaderResourceViewDimension.Texture2D,
                Texture2D = new Texture2DShaderResourceView { MostDetailedMip = 0, MipLevels = 1 }
            };
        }

        private void UploadPlane(D3D11Texture texture, byte[] source, int sourceStride, int width, int rows)
        {
            MappedSubresource map = _context.Map(texture, 0, MapMode.WriteDiscard);
            try
            {
                int copy = Math.Min(sourceStride, (int)map.RowPitch);
                fixed (byte* srcBase = source)
                {
                    byte* src = srcBase;
                    byte* dst = (byte*)map.DataPointer;
                    for (int y = 0; y < rows; y++)
                    {
                        Buffer.MemoryCopy(src, dst, map.RowPitch, copy);
                        src += sourceStride;
                        dst += map.RowPitch;
                    }
                }
            }
            finally
            {
                _context.Unmap(texture, 0);
            }
        }

        private void DrawTile(Slot slot, TileRect tile)
        {
            // uvScale 상수 갱신.
            MappedSubresource cb = _context.Map(_paramsCb!, 0, MapMode.WriteDiscard);
            float* p = (float*)cb.DataPointer;
            p[0] = slot.UScale;
            p[1] = slot.VScale;
            p[2] = 0f;
            p[3] = 0f;
            _context.Unmap(_paramsCb!, 0);

            _context.RSSetViewport(new Viewport(tile.X, tile.Y, tile.Width, tile.Height, 0f, 1f));

            if (slot.IsHardware)
            {
                _context.PSSetShader(_psNv12);
                _context.PSSetShaderResource(0, slot.Srv0!);
                _context.PSSetShaderResource(1, slot.Srv1!);
            }
            else
            {
                _context.PSSetShader(_psYuv420);
                _context.PSSetShaderResource(0, slot.Srv0!);
                _context.PSSetShaderResource(1, slot.Srv1!);
                _context.PSSetShaderResource(2, slot.Srv2!);
            }

            _context.Draw(3, 0);
        }

        // UI 스레드: _image.Lock 안에서만 오프스크린 → 공유(백버퍼) 로 복사하고 dirty 표시.
        // 백버퍼가 Lock 동안에만 갱신되므로 WPF 가 그리다 만 프레임을 읽지 않는다(깜박임 방지).
        private void PresentToImage()
        {
            if (_d3d9Surface == null || _offscreenTex == null || _sharedTex == null)
            {
                return;
            }
            _image.Lock();
            try
            {
                if (!_backbufferAttached)
                {
                    _image.SetBackBuffer(D3DResourceType.IDirect3DSurface9, _d3d9Surface.NativePointer);
                    _backbufferAttached = true;
                }

                HwDeviceContext.Lock();
                try
                {
                    _context.CopyResource(_sharedTex, _offscreenTex);
                    _context.Flush();
                }
                finally
                {
                    HwDeviceContext.Unlock();
                }

                _image.AddDirtyRect(new Int32Rect(0, 0, _width, _height));
            }
            finally
            {
                _image.Unlock();
            }
        }

        private void InitializeResources()
        {
            _device = HwDeviceContext.Device
                ?? throw new InvalidOperationException("D3D11 디바이스 생성 불가");
            _context = HwDeviceContext.Context
                ?? throw new InvalidOperationException("D3D11 컨텍스트 없음");

            ReadOnlyMemory<byte> vsBytes = Compiler.Compile(
                VertexShaderSource, "main", "CompositorVS", "vs_4_0");
            _vs = _device.CreateVertexShader(vsBytes.Span);

            ReadOnlyMemory<byte> nv12Bytes = Compiler.Compile(
                PixelShaderNv12, "main", "CompositorNV12", "ps_4_0");
            _psNv12 = _device.CreatePixelShader(nv12Bytes.Span);

            ReadOnlyMemory<byte> yuvBytes = Compiler.Compile(
                PixelShaderYuv420, "main", "CompositorYUV420", "ps_4_0");
            _psYuv420 = _device.CreatePixelShader(yuvBytes.Span);

            _sampler = _device.CreateSamplerState(new SamplerDescription
            {
                Filter = Filter.MinMagMipLinear,
                AddressU = TextureAddressMode.Clamp,
                AddressV = TextureAddressMode.Clamp,
                AddressW = TextureAddressMode.Clamp,
                ComparisonFunc = ComparisonFunction.Never,
                MinLOD = 0,
                MaxLOD = float.MaxValue
            });

            _paramsCb = _device.CreateBuffer(new BufferDescription
            {
                ByteWidth = 16,
                Usage = ResourceUsage.Dynamic,
                BindFlags = BindFlags.ConstantBuffer,
                CPUAccessFlags = CpuAccessFlags.Write
            });

            // 오프스크린 RT: 렌더 스레드가 그리는 대상(WPF 가 직접 보지 않음).
            _offscreenTex = _device.CreateTexture2D(new Texture2DDescription
            {
                Width = (uint)_width,
                Height = (uint)_height,
                MipLevels = 1,
                ArraySize = 1,
                Format = DxgiFormat.B8G8R8A8_UNorm,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.RenderTarget,
                CPUAccessFlags = CpuAccessFlags.None,
                MiscFlags = ResourceOptionFlags.None
            });
            _offscreenRtv = _device.CreateRenderTargetView(_offscreenTex);

            // 공유 텍스처: 오프스크린 복사 대상이자 D3D9 백버퍼. BGRA + Shared(레거시 핸들).
            _sharedTex = _device.CreateTexture2D(new Texture2DDescription
            {
                Width = (uint)_width,
                Height = (uint)_height,
                MipLevels = 1,
                ArraySize = 1,
                Format = DxgiFormat.B8G8R8A8_UNorm,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.RenderTarget | BindFlags.ShaderResource,
                CPUAccessFlags = CpuAccessFlags.None,
                MiscFlags = ResourceOptionFlags.Shared
            });

            IntPtr sharedHandle;
            using (IDXGIResource dxgiRes = _sharedTex.QueryInterface<IDXGIResource>())
            {
                sharedHandle = dxgiRes.SharedHandle;
            }

            _d3d9 = D3D9.Direct3DCreate9Ex();
            _d3d9Device = _d3d9.CreateDeviceEx(
                0,
                DeviceType.Hardware,
                GetMainWindowHandle(),
                CreateFlags.HardwareVertexProcessing | CreateFlags.Multithreaded | CreateFlags.FpuPreserve,
                new D9PresentParameters
                {
                    BackBufferWidth = 1,
                    BackBufferHeight = 1,
                    BackBufferFormat = D9Format.A8R8G8B8,
                    BackBufferCount = 1,
                    SwapEffect = D9SwapEffect.Discard,
                    Windowed = true,
                    PresentationInterval = PresentInterval.Immediate
                });

            // D3D11 공유 텍스처를 D3D9 텍스처로 연다(같은 VRAM). pSharedHandle 에 핸들을 넘기면
            // 새 surface 를 만드는 대신 기존 공유 리소스를 연다.
            IntPtr openHandle = sharedHandle;
            _d3d9Tex = _d3d9Device.CreateTexture(
                (uint)_width, (uint)_height, 1,
                Vortice.Direct3D9.Usage.RenderTarget,
                D9Format.A8R8G8B8,
                Pool.Default,
                ref openHandle);
            _d3d9Surface = _d3d9Tex.GetSurfaceLevel(0);
        }

        private void ReleaseSlotTextures(Slot slot)
        {
            slot.Srv0?.Dispose();
            slot.Srv1?.Dispose();
            slot.Srv2?.Dispose();
            slot.Nv12Tex?.Dispose();
            slot.YTex?.Dispose();
            slot.UTex?.Dispose();
            slot.VTex?.Dispose();
            slot.Srv0 = null;
            slot.Srv1 = null;
            slot.Srv2 = null;
            slot.Nv12Tex = null;
            slot.YTex = null;
            slot.UTex = null;
            slot.VTex = null;
        }

        private void ReleaseSlot(Slot slot)
        {
            ReleaseSlotTextures(slot);
            Interlocked.Exchange(ref slot.LatestYuv, null)?.Dispose();
            Interlocked.Exchange(ref slot.LatestD3D11, null)?.Dispose();
        }

        private void ReleaseAll()
        {
            foreach (KeyValuePair<int, Slot> kv in _slots)
            {
                ReleaseSlot(kv.Value);
            }
            _slots.Clear();

            _d3d9Surface?.Dispose();
            _d3d9Tex?.Dispose();
            _d3d9Device?.Dispose();
            _d3d9?.Dispose();
            _offscreenRtv?.Dispose();
            _offscreenTex?.Dispose();
            _sharedTex?.Dispose();
            _paramsCb?.Dispose();
            _sampler?.Dispose();
            _psNv12?.Dispose();
            _psYuv420?.Dispose();
            _vs?.Dispose();

            _d3d9Surface = null;
            _d3d9Tex = null;
            _d3d9Device = null;
            _d3d9 = null;
            _offscreenRtv = null;
            _offscreenTex = null;
            _sharedTex = null;
            _paramsCb = null;
            _sampler = null;
            _psNv12 = null;
            _psYuv420 = null;
            _vs = null;
            // _device/_context 는 HwDeviceContext 소유라 여기서 해제하지 않는다.
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
    }
}
