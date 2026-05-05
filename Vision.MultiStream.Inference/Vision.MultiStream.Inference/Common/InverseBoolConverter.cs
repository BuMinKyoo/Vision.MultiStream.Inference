using System;
using System.Globalization;
using System.Windows.Data;

namespace Vision.MultiStream.Inference.Common
{
    // bool 값을 반전시키는 컨버터 (true → false, false → true)
    // 용도: RadioButton에서 UseGpu=false일 때 CPU 버튼이 선택되도록
    [ValueConversion(typeof(bool), typeof(bool))]
    public sealed class InverseBoolConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
            => value is bool b && !b;

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
            => value is bool b && !b;
    }
}
