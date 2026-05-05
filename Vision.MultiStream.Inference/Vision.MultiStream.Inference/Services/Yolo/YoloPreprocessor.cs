using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Vision.MultiStream.Inference.Services.Yolo
{
    /// <summary>
    /// YOLOv8 입력 텐서 [1,3,640,640] 만들기. 책임 1개:
    /// "어떤 형태의 픽셀이 들어오든 letterbox + 정규화 + CHW 텐서로 변환".
    /// 진입점은 입력 형태별로 둘 (파일 경로 / 메모리 BGR byte[]).
    /// 4단계: (1) letterbox 리사이즈 (2) RGB 정렬 (3) 0~1 정규화 (4) HWC→CHW 차원 재배열.
    /// </summary>
    public static class YoloPreprocessor
    {
        public const int InputSize = 640;

        // YOLOv8 letterbox 표준 패딩 색 (회색)
        private const float PadValueNormalized = 114f / 255f;

        /// <summary>
        /// 디스크의 JPG/PNG 파일을 읽어 전처리. (Snapshot 도메인용)
        /// </summary>
        public static LetterboxResult Preprocess(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);
            return PreprocessCore(image);
        }

        /// <summary>
        /// 메모리상 BGR 픽셀(OpenCV Mat 기본 포맷)을 받아 전처리. (RTSP 프레임 도메인용)
        /// bgrPixels 길이 = width * height * 3, 채널 순서 B-G-R, row-major.
        /// </summary>
        public static LetterboxResult Preprocess(byte[] bgrPixels, int width, int height)
        {
            using var image = BgrBytesToImage(bgrPixels, width, height);
            return PreprocessCore(image);
        }

        private static Image<Rgb24> BgrBytesToImage(byte[] bgr, int width, int height)
        {
            var image = new Image<Rgb24>(width, height);

            // ProcessPixelRows + GetRowSpan: Span 기반 zero-copy 접근 (직접 인덱싱보다 빠름)
            image.ProcessPixelRows(accessor =>
            {
                int idx = 0;
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        // OpenCV는 BGR 순서, ImageSharp Rgb24는 RGB 순서 → B↔R 스왑
                        // 스왑 안 하면 모델이 색상을 반대로 인식해 검출 정확도 크게 저하
                        row[x] = new Rgb24(bgr[idx + 2], bgr[idx + 1], bgr[idx]);
                        idx += 3;
                    }
                }
            });
            return image;
        }

        private static LetterboxResult PreprocessCore(Image<Rgb24> image)
        {
            int origW = image.Width;
            int origH = image.Height;

            // 종횡비 유지: 가로/세로 중 더 큰 쪽이 640에 딱 맞도록 축소 비율 결정
            float scale = System.Math.Min(
                (float)InputSize / origW,
                (float)InputSize / origH);

            int newW = (int)System.Math.Round(origW * scale);
            int newH = (int)System.Math.Round(origH * scale);

            // 640×640 안에서 이미지를 중앙에 배치하기 위한 좌우/상하 패딩 크기
            int padX = (InputSize - newW) / 2;
            int padY = (InputSize - newH) / 2;

            image.Mutate(x => x.Resize(newW, newH));

            // ONNX 입력 텐서: [배치=1, 채널=3, 높이=640, 너비=640] (NCHW 형식)
            var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

            // 전체를 회색(114/255)으로 초기화 → letterbox 빈 영역 색 (YOLOv8 학습 시 표준)
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < InputSize; y++)
                {
                    for (int x = 0; x < InputSize; x++)
                    {
                        tensor[0, c, y, x] = PadValueNormalized;
                    }
                }
            }

            // 리사이즈된 이미지를 패딩 오프셋만큼 밀어서 텐서에 복사
            // 동시에 세 가지 변환 수행:
            //   HWC(Height×Width×Channel) → CHW(Channel×Height×Width): ONNX가 요구하는 차원 순서
            //   0~255 → 0~1 정규화
            //   (padX, padY) 오프셋으로 이미지를 중앙에 배치
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < newH; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < newW; x++)
                    {
                        Rgb24 px = row[x];
                        tensor[0, 0, y + padY, x + padX] = px.R / 255f; // R 채널
                        tensor[0, 1, y + padY, x + padX] = px.G / 255f; // G 채널
                        tensor[0, 2, y + padY, x + padX] = px.B / 255f; // B 채널
                    }
                }
            });

            // scale, padX, padY를 같이 반환 → 추론 후 박스 좌표를 원본 해상도로 역변환할 때 사용
            return new LetterboxResult(tensor, scale, padX, padY, origW, origH);
        }
    }
}
