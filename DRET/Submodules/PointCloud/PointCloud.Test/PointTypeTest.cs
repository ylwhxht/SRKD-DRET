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

using System.ComponentModel;
using NUnit.Framework;

namespace PointCloud.Test
{
    [TestFixture]
    class PointTypeTest
    {
        [Test]
        public void PointConvertFromTest()
        {
            PointXYZ point = (PointXYZ) TypeDescriptor.GetConverter(typeof (PointXYZ)).ConvertFrom("1 2 3");

            Assert.AreEqual(1, point.X);
            Assert.AreEqual(2, point.Y);
            Assert.AreEqual(3, point.Z);
        }

        [Test]
        public void PointConvertToTest()
        {
            PointXYZ point = new PointXYZ(1, 2, 3);

            Assert.AreEqual("1 2 3", TypeDescriptor.GetConverter(point).ConvertTo(point, typeof(string)));
        }
    }
}
