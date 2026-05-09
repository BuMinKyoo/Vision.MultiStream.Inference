using System;
using System.IO;
using FFmpeg.AutoGen;

namespace Vision.MultiStream.Inference.Services.Rtsp.FFmpeg
{
    /// <summary>
    /// FFmpeg.AutoGen 이 P/Invoke 로 부를 네이티브 DLL 의 위치를 알려주는 1회성 초기화기.
    /// 앱 시작 시 한 번만 EnsureRegistered() 호출.
    /// 다중 호출에도 안전하도록 lock 으로 보호.
    ///
    /// 동작:
    ///   - 출력 폴더(예: bin\Debug\net10.0-windows\)에 같이 복사된 avformat-*.dll 등을
    ///     P/Invoke 로드 경로로 등록.
    ///   - 등록 후 av_log_set_level 로 로그 레벨을 ERROR 까지만 노출 (콘솔에 노이즈 차단).
    /// </summary>
    public static class FFmpegLibraryLoader
    {
        private static readonly object _gate = new();
        private static bool _registered;

        public static void EnsureRegistered()
        {
            if (_registered)
            {
                return;
            }

            lock (_gate)
            {
                if (_registered)
                {
                    return;
                }

                // exe 가 위치한 폴더 — 여기에 .csproj 의 Native\win-x64\*.dll 이 복사됨
                string baseDir = AppContext.BaseDirectory;
                if (!Directory.Exists(baseDir))
                {
                    throw new DirectoryNotFoundException($"FFmpeg 네이티브 DLL 폴더를 찾을 수 없음: {baseDir}");
                }

                // FFmpeg.AutoGen 8.x : 네이티브 DLL 검색 경로 지정
                ffmpeg.RootPath = baseDir;

                // 로그 노이즈 줄이기 (RTSP 끊김 등 진짜 오류만 표시)
                ffmpeg.av_log_set_level(ffmpeg.AV_LOG_ERROR);

                _registered = true;
            }
        }
    }
}
