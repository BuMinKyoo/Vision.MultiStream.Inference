using System.IO;

namespace Vision.MultiStream.Inference.Models
{
    /// <summary>
    /// MP3 플레이어 플레이리스트의 한 곡 — 파일 경로와 표시 이름.
    /// </summary>
    public sealed class Mp3Track
    {
        public Mp3Track(string filePath)
        {
            FilePath = filePath;
            Name = Path.GetFileNameWithoutExtension(filePath);
        }

        public string FilePath { get; }
        public string Name { get; }
    }
}
