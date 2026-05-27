using System;
using System.Collections.Concurrent;
using FFmpeg.AutoGen;

namespace Vision.MultiStream.Inference.Services.Rtsp.Pipeline
{
    /// <summary>
    /// 큐 경계에서 unmanaged 포인터를 해제하는 공용 헬퍼.
    /// 큐가 닫힌/취소된 상태로 들어온 포인터, Stop 시 큐에 남은 포인터를 free 한다.
    /// </summary>
    internal static unsafe class FfmpegNative
    {
        public static void FreePacket(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                return;
            }
            AVPacket* pkt = (AVPacket*)ptr;
            ffmpeg.av_packet_free(&pkt);
        }

        public static void FreeFrame(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                return;
            }
            AVFrame* frame = (AVFrame*)ptr;
            ffmpeg.av_frame_free(&frame);
        }

        // Stop 시 큐에 남아 있는 패킷/프레임 포인터를 전부 회수한다.
        public static void DrainAndFreePackets(BlockingCollection<IntPtr>? queue)
        {
            if (queue == null)
            {
                return;
            }
            while (queue.TryTake(out IntPtr ptr))
            {
                FreePacket(ptr);
            }
        }

        public static void DrainAndFreeFrames(BlockingCollection<IntPtr>? queue)
        {
            if (queue == null)
            {
                return;
            }
            while (queue.TryTake(out IntPtr ptr))
            {
                FreeFrame(ptr);
            }
        }

        // FFmpeg pts(time_base 단위) → 초 단위 double. NOPTS 면 NaN.
        public static double PtsToSeconds(long pts, AVRational timeBase)
        {
            if (pts == ffmpeg.AV_NOPTS_VALUE)
            {
                return double.NaN;
            }
            return pts * timeBase.num / (double)timeBase.den;
        }

        public static string FfErr(int err)
        {
            const int bufSize = 256;
            byte* buf = stackalloc byte[bufSize];
            ffmpeg.av_strerror(err, buf, (ulong)bufSize);
            return System.Runtime.InteropServices.Marshal.PtrToStringAnsi((IntPtr)buf) ?? err.ToString();
        }
    }

    /// <summary>
    /// 디먹서가 스트림을 연 뒤 디코더에게 넘기는 스트림 메타데이터.
    /// CodecParams 는 AVFormatContext 가 소유하므로, 디코더가 살아 있는 동안
    /// 디먹서(AVFormatContext) 도 살아 있어야 한다.
    /// </summary>
    internal struct RtspStreamInfo
    {
        public bool HasVideo;
        public int VideoStreamIndex;
        public IntPtr VideoCodecParams; // AVCodecParameters*
        public AVRational VideoTimeBase;
        public int Width;
        public int Height;

        public bool HasAudio;
        public int AudioStreamIndex;
        public IntPtr AudioCodecParams; // AVCodecParameters*
        public AVRational AudioTimeBase;
    }
}
