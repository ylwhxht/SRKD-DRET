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
using System.ComponentModel;
using System.IO;
using System.Linq;
using PointCloud.Exceptions;

namespace PointCloud.io
{
    public class PCDReader<T> where T:PointT
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public PointCloud<T> Read(String filename)
        {
            return Read(filename, 0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename">File to read from</param>
        /// <param name="offset">Offset to start reading from</param>
        /// <returns>Pointcloud of type T</returns>
        public PointCloud<T> Read(String filename, int offset)
        {
            PCDHeader header = ReadHeader(filename);
            List<T> pointList;

            // Start reading data
            using (StreamReader sr = new StreamReader(filename))
            {
                String line = sr.ReadLine();

                while (!line.Contains("DATA"))
                {
                    line = sr.ReadLine();
                }

                if(line.Contains("ascii"))
                {
                    pointList = readData(sr);    
                }
                else
                {
		    throw new PointCloudException("Data in file in unrecognized format.");
                    // throw new PCDReaderException("Data in file in unrecognized format.");

                }
            }

            PointCloud<T> cloud = new PointCloud<T>(pointList);
            cloud.Height = header.Height;
            cloud.Width = header.Width;

            return cloud;
        }

        private List<T> readData(StreamReader sr)
        {
            List<T> points = new List<T>();
            String line = sr.ReadLine();
            while(line != null)
            {
                points.Add((T)TypeDescriptor.GetConverter(typeof(T)).ConvertFrom(line));
                line = sr.ReadLine();
            }

            return points;
        }

        /// <summary>
        /// Reads a poincloud from the specified *.pcd file. Supports different version of the pcd file format.
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static PointCloud<T> LoadPCDFile(String filename)
        {
            PCDReader<T> reader = new PCDReader<T>();
            return reader.Read(filename, 0);
        }

        /// <summary>
        /// Reads the header of PCD file;
        /// </summary>
        /// <param name="filename">PCD File</param>
        /// <returns>PCDHeader object</returns>
        public PCDHeader ReadHeader(string filename)
        {
            PCDHeader header = new PCDHeader();

            using (StreamReader sr = new StreamReader(filename))
            {
                    String line = sr.ReadLine();

                // TODO: Code needs some refactoring. Lots of repeated code but it works.
                // Keep reading until Datasection is found
                while(line != null && !line.Contains("DATA"))
                {
                    if(line.Contains("VERSION"))
                    {
                        header.Version = getVersion(line);
                    }
                    if (line.Contains("FIELDS"))
                    {
                        // Read fieldsnames from the line expects no spaces in field name
                        string[] fields = line.Substring(7).Split(' ');

                        for(int i = 0; i < fields.Length; i++)
                        {
                            header.Fields.Add(i, new FieldDescription { Name = fields[i] });
                        }
                    }
                    if (line.Contains("SIZE"))
                    {
                        if(header.Fields.Count == 0)
                        {
                            throw new PointCloudException("No fields found in pcb file.");
                        }

                        string[] fields = line.Substring(7).Split(' ');

                        for (int i = 0; i < fields.Length; i++)
                        {
                            FieldDescription desc = header.Fields[i];
                            desc.Size = Convert.ToInt32(fields[i]);
                            header.Fields[i] = desc;
                        }
                    }
                    if (line.Contains("TYPE"))
                    {
                            if (header.Fields.Count == 0)
                        {
                            throw new PointCloudException("No fields found in pcb file.");
                        }

                        string[] fields = line.Substring(5).Split(' ');

                        for (int i = 0; i < fields.Length; i++)
                        {
                            FieldDescription desc = header.Fields[i];
                            desc.Type = fields[i].ToCharArray()[0];
                            header.Fields[i] = desc;
                        }     
                    }

                    if (line.Contains("COUNT"))
                    {
                        if (header.Fields.Count == 0)
                        {
                            throw new PointCloudException("No fields found in pcb file.");
                        }

                        string[] fields = line.Substring(6).Split(' ');

                        for (int i = 0; i < fields.Length; i++)
                        {
                            FieldDescription desc = header.Fields[i];
                            desc.Count = Convert.ToInt32(fields[i]);
                            header.Fields[i] = desc;
                        } 
                    }
                    if (line.Contains("WIDTH"))
                    {
                        header.Width = Convert.ToInt32(line.Substring(6));                  
                    }
                    if (line.Contains("HEIGHT"))
                    {
                        header.Height = Convert.ToInt32(line.Substring(7));
                    }

                    line = sr.ReadLine();
                }
            }

            return header;
        }

        /// <summary>
        /// Retrieves the version from a version String. String is in format VERSION 0.8.
        /// </summary>
        /// <param name="line">String</param>
        /// <returns>PCD Version</returns>
        private PCDVersion getVersion(String line)
        {
            PCDVersion returnval = PCDVersion.PCDv5;
            string value = new string(line.Skip(8).ToArray());
            if (value == ".7" || value == "0.7")
            {
                returnval = PCDVersion.PCDv7;
            }
            else if (value == ".6" || value == "0.6")
            {
                returnval = PCDVersion.PCDv6;
            }
            else if (value == ".5" || value == "0.5")
            {
                returnval = PCDVersion.PCDv5;
            }
            else
            {
                throw new PointCloudException("No version information found in PCD file.");
            }

            return returnval;
        }
    }
}
