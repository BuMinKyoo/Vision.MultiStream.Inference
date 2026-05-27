using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using SharpGen.Runtime;
using Vision.MultiStream.Inference.Common;
using Vision.MultiStream.Inference.Services.Rtsp;
using Vortice.D3DCompiler;
using Vortice.Direct3D9;

namespace Vision.MultiStream.Inference.Services.Direct3D
{
    public sealed class D3DImageYuvPresenter : IDisposable
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

        private readonly D3DImage _image = new();
        private readonly IDirect3D9Ex _d3d;
        private readonly IDirect3DDevice9Ex _device;
        private readonly IDirect3DPixelShader9 _pixelShader;

        private IDirect3DSurface9? _renderTarget;
        private IDirect3DTexture9? _yTexture;
        private IDirect3DTexture9? _uTexture;
        private IDirect3DTexture9? _vTexture;
        private int _width;
        private int _height;
        private bool _disposed;

        public D3DImageYuvPresenter()
        {
            _d3d = D3D9.Direct3DCreate9Ex();
            _device = _d3d.CreateDeviceEx(
                0,
                DeviceType.Hardware,
                GetMainWindowHandle(),
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
                PixelShaderSource,
                "main",
                "Yuv420ToRgb",
                "ps_2_0",
                ShaderFlags.OptimizationLevel3,
                EffectFlags.None);
            _pixelShader = _device.CreatePixelShader(shaderBytes.Span);
        }

        public ImageSource Image => _image;

        public void Render(RtspYuvFrame frame)
        {
            ThrowIfDisposed();
            if (frame.Width <= 0 || frame.Height <= 0)
            {
                return;
            }

            using (PerfProbe.Measure("display.d3d.render_yuv"))
            {
                EnsureResources(frame.Width, frame.Height);
                int chromaHeight = (frame.Height + 1) / 2;
                UploadPlane(_yTexture!, frame.YPlane, frame.YStride, frame.Height);
                UploadPlane(_uTexture!, frame.UPlane, frame.UStride, chromaHeight);
                UploadPlane(_vTexture!, frame.VPlane, frame.VStride, chromaHeight);
                DrawFrame(frame.Width, frame.Height);

                _image.Lock();
                try
                {
                    _image.AddDirtyRect(new Int32Rect(0, 0, frame.Width, frame.Height));
                }
                finally
                {
                    _image.Unlock();
                }
            }
        }

        private static IntPtr GetMainWindowHandle()
        {
            Window? window = Application.Current?.MainWindow;
            return window == null ? IntPtr.Zero : new WindowInteropHelper(window).Handle;
        }

        private void EnsureResources(int width, int height)
        {
            if (_renderTarget != null && width == _width && height == _height)
            {
                return;
            }

            ReleaseFrameResources();

            _width = width;
            _height = height;
            _renderTarget = _device.CreateRenderTargetEx(
                (uint)width,
                (uint)height,
                Format.A8R8G8B8,
                MultisampleType.None,
                0,
                false,
                Usage.None);
            _yTexture = _device.CreateTexture((uint)width, (uint)height, 1, Usage.Dynamic, Format.L8, Pool.Default);
            _uTexture = _device.CreateTexture((uint)(width / 2), (uint)(height / 2), 1, Usage.Dynamic, Format.L8, Pool.Default);
            _vTexture = _device.CreateTexture((uint)(width / 2), (uint)(height / 2), 1, Usage.Dynamic, Format.L8, Pool.Default);

            _image.Lock();
            try
            {
                _image.SetBackBuffer(D3DResourceType.IDirect3DSurface9, _renderTarget.NativePointer);
            }
            finally
            {
                _image.Unlock();
            }
        }

        // source 는 풀에서 빌린 버퍼라 Length 가 실제 평면보다 클 수 있으므로 rows 를 명시로 받는다.
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

        private unsafe void DrawFrame(int width, int height)
        {
            _device.SetRenderTarget(0, _renderTarget!);
            _device.Viewport = new Viewport
            {
                X = 0,
                Y = 0,
                Width = width,
                Height = height,
                MinZ = 0f,
                MaxZ = 1f
            };
            _device.SetRenderState(RenderState.ZEnable, false);
            _device.SetRenderState(RenderState.Lighting, false);
            _device.SetSamplerState(0, SamplerState.MinFilter, (int)TextureFilter.Linear);
            _device.SetSamplerState(0, SamplerState.MagFilter, (int)TextureFilter.Linear);
            _device.SetSamplerState(0, SamplerState.AddressU, (int)TextureAddress.Clamp);
            _device.SetSamplerState(0, SamplerState.AddressV, (int)TextureAddress.Clamp);
            _device.SetSamplerState(1, SamplerState.MinFilter, (int)TextureFilter.Linear);
            _device.SetSamplerState(1, SamplerState.MagFilter, (int)TextureFilter.Linear);
            _device.SetSamplerState(1, SamplerState.AddressU, (int)TextureAddress.Clamp);
            _device.SetSamplerState(1, SamplerState.AddressV, (int)TextureAddress.Clamp);
            _device.SetSamplerState(2, SamplerState.MinFilter, (int)TextureFilter.Linear);
            _device.SetSamplerState(2, SamplerState.MagFilter, (int)TextureFilter.Linear);
            _device.SetSamplerState(2, SamplerState.AddressU, (int)TextureAddress.Clamp);
            _device.SetSamplerState(2, SamplerState.AddressV, (int)TextureAddress.Clamp);
            _device.SetTexture(0, _yTexture);
            _device.SetTexture(1, _uTexture);
            _device.SetTexture(2, _vTexture);
            _device.PixelShader = _pixelShader;
            _device.VertexFormat = VertexFormat.PositionRhw | VertexFormat.Texture3;

            Span<Vertex> vertices =
            [
                new Vertex(-0.5f, -0.5f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f),
                new Vertex(width - 0.5f, -0.5f, 0f, 1f, 1f, 0f, 1f, 0f, 1f, 0f),
                new Vertex(-0.5f, height - 0.5f, 0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f),
                new Vertex(width - 0.5f, height - 0.5f, 0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f)
            ];

            fixed (Vertex* vertexPtr = vertices)
            {
                _device.BeginScene();
                try
                {
                    _device.DrawPrimitiveUP(
                        PrimitiveType.TriangleStrip,
                        2,
                        (IntPtr)vertexPtr,
                        (uint)Marshal.SizeOf<Vertex>());
                }
                finally
                {
                    _device.EndScene();
                }
            }
        }

        private void ReleaseFrameResources()
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

            _vTexture?.Dispose();
            _uTexture?.Dispose();
            _yTexture?.Dispose();
            _renderTarget?.Dispose();
            _vTexture = null;
            _uTexture = null;
            _yTexture = null;
            _renderTarget = null;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(D3DImageYuvPresenter));
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            ReleaseFrameResources();
            _pixelShader.Dispose();
            _device.Dispose();
            _d3d.Dispose();
        }

        [StructLayout(LayoutKind.Sequential)]
        private readonly struct Vertex
        {
            public Vertex(
                float x,
                float y,
                float z,
                float rhw,
                float tu0,
                float tv0,
                float tu1,
                float tv1,
                float tu2,
                float tv2)
            {
                X = x;
                Y = y;
                Z = z;
                Rhw = rhw;
                Tu0 = tu0;
                Tv0 = tv0;
                Tu1 = tu1;
                Tv1 = tv1;
                Tu2 = tu2;
                Tv2 = tv2;
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
