/*
Pointcloud library for .NET
Copyright (C) 2013  M. Hofman

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

using System.Collections.Generic;
using System.Linq;
using PointCloud.Exceptions;

namespace PointCloud
{
    public class PointCloud<T> where T:PointT
    {
        private bool _isDense = true;
        private List<T> _points; 

        public PointCloud()
        {
            _points = new List<T>();        
        }

        public PointCloud(IEnumerable<T> collection)
        {
            _points = new List<T>(collection);
        }

        public ICollection<T> Points { get { return _points; }
        }

        public void Add(T Object)
        {
            Points.Add(Object);
        }

        public T At(int n)
        {
            return Points.ElementAt(n);
        }

        public T At(int column, int row)
        {
            if(Height > 1)
            {
                return Points.ElementAt(row*this.Width + column);
            }

            throw new IsNotDenseException("Can't use 2D indexing with an unorganized pointcloud.");
        }

        public bool IsOrganized()
        {
            return (Height != 1);
        }

        public void Clear()
        {
            Points.Clear();
            Width = 0;
            Height = 0;
        }

        public T this[int n]
        {
            get { return _points.ElementAt(n); }
            set { _points[n] = value; }
        }

        public int Size { get { return Points.Count; } }

        public bool Empty { get { return Points.Count == 0; } }

        public int Height { get; set; }

        public int Width { get; set; }

        public bool IsDense
        {
            get { return _isDense; }
            set { _isDense = value; }
        }
    }
}
