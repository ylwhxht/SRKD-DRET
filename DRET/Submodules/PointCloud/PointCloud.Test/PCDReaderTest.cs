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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NUnit.Framework;
using PointCloud.io;

namespace PointCloud.Test
{
    [TestFixture]
    public class PCDReaderTest
    {
        [Test]
        public void FileReadVersionTest()
        {
            PCDReader<PointXYZ> reader = new PCDReader<PointXYZ>();

            PCDHeader header = reader.ReadHeader("test.pcd");

            Assert.AreEqual(PCDVersion.PCDv7, header.Version);
        }

        [Test]
        public void FileReadWidthTest()
        {
            PCDReader<PointXYZ> reader = new PCDReader<PointXYZ>();

            PCDHeader header = reader.ReadHeader("test.pcd");

            Assert.AreEqual(213, header.Width);
        }

        [Test]
        public void FileReadHeightTest()
        {
            PCDReader<PointXYZ> reader = new PCDReader<PointXYZ>();

            PCDHeader header = reader.ReadHeader("test.pcd");

            Assert.AreEqual(1, header.Height);
        }

        [Test]
        public void FileReadPointsTest()
        {
            PCDReader<PointXYZ> reader = new PCDReader<PointXYZ>();

            PCDHeader header = reader.ReadHeader("test.pcd");

            Assert.AreEqual(213, header.Points);
        }

        [Test]
        public void FileReadFieldsTest()
        {
            PCDReader<PointXYZ> reader = new PCDReader<PointXYZ>();

            PCDHeader header = reader.ReadHeader("test.pcd");

            Assert.AreEqual(4, header.Fields.Count);
            Assert.AreEqual(4, header.Fields[0].Size);
            Assert.AreEqual('F', header.Fields[0].Type);
            Assert.AreEqual(1, header.Fields[0].Count);
        }

        [Test]
        public void FileReadCompleteTest()
        {
            PCDReader<PointXYZ> reader = new PCDReader<PointXYZ>();
            PointCloud<PointXYZ> cloud = reader.Read("test.pcd");

            Assert.AreEqual(213, cloud.Points.Count);

            Assert.AreEqual(0.93773, cloud[0].X, 0.000001);
            Assert.AreEqual(0.33763, cloud[0].Y, 0.000001);
            Assert.AreEqual(0, cloud[0].Z, 0.000001);
        }

    }
}
