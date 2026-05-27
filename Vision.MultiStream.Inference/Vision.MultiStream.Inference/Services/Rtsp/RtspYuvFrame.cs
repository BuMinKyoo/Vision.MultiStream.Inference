using System;

namespace Vision.MultiStream.Inference.Services.Rtsp
{
    public sealed class RtspYuvFrame
    {
        public RtspYuvFrame(
            byte[] yPlane,
            byte[] uPlane,
            byte[] vPlane,
            int width,
            int height,
            int yStride,
            int uStride,
            int vStride,
            DateTime capturedAt,
            double ptsSeconds)
        {
            YPlane = yPlane;
            UPlane = uPlane;
            VPlane = vPlane;
            Width = width;
            Height = height;
            YStride = yStride;
            UStride = uStride;
            VStride = vStride;
            CapturedAt = capturedAt;
            PtsSeconds = ptsSeconds;
        }

        public byte[] YPlane { get; }

        public byte[] UPlane { get; }

        public byte[] VPlane { get; }

        public int Width { get; }

        public int Height { get; }

        public int YStride { get; }

        public int UStride { get; }

        public int VStride { get; }

        public DateTime CapturedAt { get; }

        public double PtsSeconds { get; }
    }
}
