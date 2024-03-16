using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;

namespace PointCloud.PointTypes.Converters
{
    /// <summary>
    /// Class that converts between PointXYZ objects and string objects.
    /// </summary>
    class PointXYZConverter : TypeConverter
    {
        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            if(sourceType == typeof(string))
            {
                return true;
            }

            return base.CanConvertFrom(context, sourceType);
        }

        public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
        {
            if(value is string)
            {
                String stringRep = (String) value;
                culture = new CultureInfo("en-US");
                String[] splitted = stringRep.Split(' ');
                return new PointXYZ(Convert.ToSingle(splitted[0], culture), Convert.ToSingle(splitted[1], culture),
                                    Convert.ToSingle(splitted[2], culture));
            }

            return base.ConvertFrom(context, culture, value);
        }

        public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
        {
            if(destinationType == typeof(string))
            {
                return true;
            }

            return base.CanConvertTo(context, destinationType);
        }

        public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
        {
            if(destinationType == typeof(string))
            {
                PointXYZ point = (PointXYZ) value;
                return point.X + " " + point.Y + " " + point.Z;
            }
            return base.ConvertTo(context, culture, value, destinationType);
        }
    }
}
