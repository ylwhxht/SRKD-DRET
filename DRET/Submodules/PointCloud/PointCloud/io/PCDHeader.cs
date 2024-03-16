using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PointCloud.io
{
    public class PCDHeader
    {
        private PCDVersion _version;
        private SortedDictionary<int, FieldDescription> _fields;
        private int _width;
        private int _height;

        public PCDHeader()
        {
            Fields = new SortedDictionary<int, FieldDescription>();
            Width = 0;
            Height = 0;
        }

        public int Points
        {
            get { return Width*Height; }
        }

        public PCDVersion Version
        {
            get { return _version; }
            set { _version = value; }
        }

        public SortedDictionary<int, FieldDescription> Fields
        {
            get { return _fields; }
            set { _fields = value; }
        }

        public int Width
        {
            get { return _width; }
            set { _width = value; }
        }

        public int Height
        {
            get { return _height; }
            set { _height = value; }
        }
    }
}
