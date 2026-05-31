using System;
using FFmpeg.AutoGen;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// HW(D3D11VA) 디코딩 결과 프레임. CPU 로 다운로드하지 않고 GPU 텍스처를 그대로 들고 다닌다.
    ///
    /// 내부적으로 디코더가 만든 D3D11 AVFrame 의 ref(av_frame_clone) 를 소유하므로,
    /// 이 객체가 살아 있는 동안 해당 디코드 surface 가 풀로 반납되지 않는다(컴포지터가 샘플 끝낼 때까지 보장).
    /// 반드시 Dispose 로 ref 를 반납해 디코드 풀 고갈을 막아야 한다.
    ///
    /// <see cref="Texture"/> 는 ID3D11Texture2D*(보통 Texture2DArray), <see cref="ArrayIndex"/> 는
    /// 그 배열에서 이 프레임이 들어 있는 슬라이스. 컴포지터는 (Texture, ArrayIndex) 로 SRV 를 만든다.
    /// </summary>
    public sealed unsafe class RtspD3D11Frame : IDisposable
    {
        private AVFrame* _frame;
        private bool _disposed;

        public RtspD3D11Frame(IntPtr clonedFrame, double ptsSeconds)
        {
            _frame = (AVFrame*)clonedFrame;
            Texture = (IntPtr)_frame->data[0];
            ArrayIndex = (int)_frame->data[1];
            Width = _frame->width;
            Height = _frame->height;
            PtsSeconds = ptsSeconds;
        }

        /// <summary>ID3D11Texture2D* (디코드 풀의 NV12 텍스처 배열).</summary>
        public IntPtr Texture { get; }

        /// <summary>텍스처 배열 내 이 프레임의 슬라이스 인덱스.</summary>
        public int ArrayIndex { get; }

        public int Width { get; }

        public int Height { get; }

        public double PtsSeconds { get; }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            if (_frame != null)
            {
                AVFrame* f = _frame;
                ffmpeg.av_frame_free(&f);
                _frame = null;
            }
        }
    }
}
