namespace Vision.MultiStream.Inference.Models
{
    /// <summary>
    /// YOLOv8 추론 결과 한 건. 좌표계는 원본 이미지 픽셀 기준 (letterbox 역변환 후).
    /// </summary>
    public sealed class Detection
    {
        public float X { get; init; }          // 박스 좌상단 X 픽셀 좌표
        public float Y { get; init; }          // 박스 좌상단 Y 픽셀 좌표
        public float Width { get; init; }      // 박스 너비 (픽셀)
        public float Height { get; init; }     // 박스 높이 (픽셀)

        public int ClassId { get; init; }      // COCO 클래스 인덱스 (0=person, 2=car ...)
        public string ClassName { get; init; } = string.Empty; // ClassId에 대응하는 라벨 문자열
        public float Confidence { get; init; } // 신뢰도 0~1 (0.25 미만은 ParseOutput에서 이미 걸러짐)

        public override string ToString()
        {
            return $"{ClassName}({ClassId}) {Confidence:P1} @ ({X:F0},{Y:F0} {Width:F0}x{Height:F0})";
        }
    }
}
