namespace Vision.MultiStream.Inference.Services
{
    /// <summary>
    /// COCO 데이터셋 80개 클래스 이름.
    /// 인덱스(0~79)가 YOLOv8 출력 텐서의 클래스 채널 순서와 1:1 매칭됨.
    /// </summary>
    public static class CocoLabels
    {
        public static readonly string[] Names =
        [
            "person",         "bicycle",      "car",            "motorcycle",
            "airplane",       "bus",          "train",          "truck",
            "boat",           "traffic light","fire hydrant",   "stop sign",
            "parking meter",  "bench",        "bird",           "cat",
            "dog",            "horse",        "sheep",          "cow",
            "elephant",       "bear",         "zebra",          "giraffe",
            "backpack",       "umbrella",     "handbag",        "tie",
            "suitcase",       "frisbee",      "skis",           "snowboard",
            "sports ball",    "kite",         "baseball bat",   "baseball glove",
            "skateboard",     "surfboard",    "tennis racket",  "bottle",
            "wine glass",     "cup",          "fork",           "knife",
            "spoon",          "bowl",         "banana",         "apple",
            "sandwich",       "orange",       "broccoli",       "carrot",
            "hot dog",        "pizza",        "donut",          "cake",
            "chair",          "couch",        "potted plant",   "bed",
            "dining table",   "toilet",       "tv",             "laptop",
            "mouse",          "remote",       "keyboard",       "cell phone",
            "microwave",      "oven",         "toaster",        "sink",
            "refrigerator",   "book",         "clock",          "vase",
            "scissors",       "teddy bear",   "hair drier",     "toothbrush"
        ];

        public static int Count => Names.Length;

        public static string Get(int classId)
        {
            if (classId < 0 || classId >= Names.Length)
            {
                return $"unknown({classId})";
            }
            return Names[classId];
        }
    }
}
