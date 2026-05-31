using System;
using System.Runtime.InteropServices;
using FFmpeg.AutoGen;
using Vision.MultiStream.Inference.Services.Rtsp.FFmpeg;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using D3D11Device = Vortice.Direct3D11.ID3D11Device;
using D3D11Context = Vortice.Direct3D11.ID3D11DeviceContext;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 프로세스 1개로 공유하는 D3D11 디바이스의 소유자.
    ///
    /// 핵심: 디바이스를 우리가 직접 만들고(<see cref="Device"/>/<see cref="Context"/>),
    /// 그 디바이스를 FFmpeg D3D11VA 컨텍스트로 감싼다(<see cref="AcquireRef"/>).
    /// → 디코더(FFmpeg)와 컴포지터(StreamCompositor)가 같은 ID3D11Device 를 쓰므로
    ///   디코더가 만든 NV12 텍스처를 컴포지터가 복사 0으로 직접 샘플링할 수 있다.
    ///
    /// 디바이스 immediate context 는 디코드 스레드(FFmpeg 내부)와 컴포지터 렌더 스레드가
    /// 공유한다. ID3D11DeviceContext 는 스레드 안전이 아니므로 양쪽 모두
    /// <see cref="Lock"/>/<see cref="Unlock"/>(= FFmpeg 기본 락) 으로 직렬화해야 한다.
    ///
    /// 한 번 초기화에 실패하면(드라이버 미지원 등) 같은 프로세스에서 재시도하지 않는다.
    /// </summary>
    internal static unsafe class HwDeviceContext
    {
        // FFmpeg 의 lock/unlock 콜백 시그니처: void f(void* lock_ctx)
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void LockDelegate(void* lockCtx);

        private const uint D3D11_BIND_SHADER_RESOURCE = 0x8;

        private static readonly object _gate = new();
        private static bool _initFailed;

        private static D3D11Device? _device;
        private static D3D11Context? _context;
        private static AVBufferRef* _shared;     // 우리 디바이스를 감싼 FFmpeg D3D11VA 디바이스 ctx
        private static LockDelegate? _ffLock;
        private static LockDelegate? _ffUnlock;
        private static void* _lockCtx;

        /// <summary>디코드 텍스처에 SRV 를 만들 수 있도록 frames pool 에 추가로 줄 bind flag.</summary>
        public static uint FramesShaderResourceBindFlag => D3D11_BIND_SHADER_RESOURCE;

        /// <summary>공유 D3D11 디바이스. 초기화 실패 시 null.</summary>
        public static D3D11Device? Device
        {
            get
            {
                lock (_gate)
                {
                    return EnsureCore() ? _device : null;
                }
            }
        }

        /// <summary>공유 immediate context. 사용 전후로 반드시 Lock/Unlock.</summary>
        public static D3D11Context? Context
        {
            get
            {
                lock (_gate)
                {
                    return EnsureCore() ? _context : null;
                }
            }
        }

        /// <summary>디바이스 초기화 가능 여부(컴포지터가 D3D11 모드로 갈지 판단).</summary>
        public static bool IsAvailable
        {
            get
            {
                lock (_gate)
                {
                    return EnsureCore();
                }
            }
        }

        /// <summary>immediate context 직렬화용(= FFmpeg 기본 락). 디코드 스레드와 공유.</summary>
        public static void Lock()
        {
            _ffLock?.Invoke(_lockCtx);
        }

        public static void Unlock()
        {
            _ffUnlock?.Invoke(_lockCtx);
        }

        /// <summary>
        /// 디코더용. 공유 디바이스를 감싼 FFmpeg ctx 에 av_buffer_ref 한 새 핸들을 돌려준다.
        /// codec ctx 가 소유권을 가져가 avcodec_free_context 시점에 unref 한다.
        /// 실패 시 IntPtr.Zero — 호출부는 HW 디코딩을 포기하고 에러를 띄워야 한다.
        /// </summary>
        public static IntPtr AcquireRef()
        {
            lock (_gate)
            {
                if (!EnsureCore())
                {
                    return IntPtr.Zero;
                }
                AVBufferRef* newRef = ffmpeg.av_buffer_ref(_shared);
                return (IntPtr)newRef;
            }
        }

        /// <summary>av_buffer_unref. 호출부가 자기 ref 를 반납한다.</summary>
        public static void ReleaseRef(IntPtr handle)
        {
            if (handle == IntPtr.Zero)
            {
                return;
            }
            AVBufferRef* r = (AVBufferRef*)handle;
            ffmpeg.av_buffer_unref(&r);
        }

        // _gate 를 잡은 상태에서 호출. 디바이스 + FFmpeg 래핑을 lazy 1회 생성.
        private static bool EnsureCore()
        {
            if (_initFailed)
            {
                return false;
            }
            if (_shared != null)
            {
                return true;
            }

            try
            {
                // 컴포지터가 스트림 시작보다 먼저 디바이스를 만들 수 있으므로 여기서 FFmpeg 등록 보장.
                FFmpegLibraryLoader.EnsureRegistered();

                FeatureLevel[] levels =
                {
                    FeatureLevel.Level_11_1,
                    FeatureLevel.Level_11_0
                };
                // BgraSupport: BGRA 렌더 타깃 + D3D9 공유 surface 호환. VideoSupport: HW 디코딩 필수.
                D3D11.D3D11CreateDevice(
                    null,
                    DriverType.Hardware,
                    DeviceCreationFlags.BgraSupport | DeviceCreationFlags.VideoSupport,
                    levels,
                    out D3D11Device? created).CheckError();

                D3D11Device device = created
                    ?? throw new InvalidOperationException("D3D11 디바이스 생성 실패");
                D3D11Context context = device.ImmediateContext;

                AVBufferRef* hwdev = ffmpeg.av_hwdevice_ctx_alloc(AVHWDeviceType.AV_HWDEVICE_TYPE_D3D11VA);
                if (hwdev == null)
                {
                    throw new InvalidOperationException("av_hwdevice_ctx_alloc 실패");
                }

                AVHWDeviceContext* dctx = (AVHWDeviceContext*)hwdev->data;
                AVD3D11VADeviceContext* d3d = (AVD3D11VADeviceContext*)dctx->hwctx;

                // FFmpeg 가 init/free 에서 자기 ref 를 잡고 풀므로 우리 쪽 ref 를 미리 늘려둔다.
                device.AddRef();
                context.AddRef();
                d3d->device = (global::FFmpeg.AutoGen.ID3D11Device*)device.NativePointer;
                d3d->device_context = (global::FFmpeg.AutoGen.ID3D11DeviceContext*)context.NativePointer;

                _device = device;
                _context = context;

                int ret = ffmpeg.av_hwdevice_ctx_init(hwdev);
                if (ret < 0)
                {
                    ffmpeg.av_buffer_unref(&hwdev);
                    throw new InvalidOperationException($"av_hwdevice_ctx_init 실패: {ret}");
                }

                _shared = hwdev;
                // init 이 설정한 기본 락을 컴포지터도 동일하게 사용.
                _ffLock = Marshal.GetDelegateForFunctionPointer<LockDelegate>(d3d->@lock.Pointer);
                _ffUnlock = Marshal.GetDelegateForFunctionPointer<LockDelegate>(d3d->unlock.Pointer);
                _lockCtx = d3d->lock_ctx;
                return true;
            }
            catch
            {
                _initFailed = true;
                _context?.Dispose();
                _context = null;
                _device?.Dispose();
                _device = null;
                return false;
            }
        }
    }
}
