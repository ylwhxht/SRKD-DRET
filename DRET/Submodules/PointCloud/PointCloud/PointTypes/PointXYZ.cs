using System.ComponentModel;
using PointCloud.PointTypes.Converters;

namespace PointCloud
{
    [TypeConverter(typeof(PointXYZConverter))]
    public class PointXYZ : PointXY
    {
        public PointXYZ() : base()
        {
            Z = 0;
        }

        public PointXYZ(float x, float y, float z) : base(x, y)
        {
            Z = z;
        }

        public float Z { get; set; }
    }
}
