using System;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 수신 스레드가 디코딩해서 넘기는 한 프레임 단위.
    /// BgrPixels 는 OpenCV Mat 에서 그대로 복사한 BGR row-major 바이트 배열.
    /// </summary>
    public sealed class RtspFrame
    {
        public RtspFrame(byte[] bgrPixels, int width, int height, DateTime capturedAt, double ptsSeconds)
        {
            BgrPixels = bgrPixels;
            Width = width;
            Height = height;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
        }

        public byte[] BgrPixels { get; }

        public int Width { get; }

        public int Height { get; }

        public DateTime CapturedAt { get; }

        // PTS 가 NOPTS 또는 미지원이면 double.NaN.
        public double PtsSeconds { get; }
    }
}
