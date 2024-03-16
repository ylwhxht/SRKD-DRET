using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PointCloud;
using PointCloud.io;
using LPCSS;

public class LabelVehicle : MonoBehaviour
{
    private DetectionResult m_detectionResult = null;
    string m_pcdFilePath = null;
    PCDReader<PointXYZ> m_pcdReader = new PCDReader<PointXYZ>();
    private int m_idx = 0;
    private string m_originPCDPath = "";

    private void Awake()
    {
        m_originPCDPath = GlobalSetting.OutputPCDPath;
        
        // Example, see the description below
        Run("segment-13830510593707564159_5575_000_5595_000_with_camera_labels",  "seg138_renamed");
    }

    private void Run(string l0, string l1)
    {
        m_idx = 0;
        // first line the json file, second line the ego-vehicle velocity
        StreamReader jsonFileReader = new StreamReader(
            GlobalSetting.SprayDetectionJSONPath + l0 + ".json");
        m_detectionResult = JsonUtility.FromJson<DetectionResult>(jsonFileReader.ReadToEnd());

        // WE CAN IGNORE THE SECOND LINE HERE AS IT'S NOT IMPORTANT FOR LAEBLING
        GlobalSetting.OutputPCDPath = m_originPCDPath + "LabelVehicleFromRealSpray/" + l0 + "/";

        
        System.IO.Directory.CreateDirectory(GlobalSetting.OutputPCDPath);
        
        // Remove the last entry in Data
        Array.Resize(ref m_detectionResult.Data, m_detectionResult.Data.Length - 1);
        // Remove the last entry in all Boxes array
        foreach (var e in m_detectionResult.Data)
        {
            Array.Resize(ref e.Boxes, e.Boxes.Length - 1);
        }

        // Core Loop : 
        // 1. For each point cloud in the directory, read the point cloud
        // 2. From the corresponding Boxes list, determine whether each point is in any of the box
        // 3. Output labelled point cloud

        // After : Use the labelled point cloud for PCD_MergePointCloud

        DirectoryInfo dir = new DirectoryInfo(
            GlobalSetting.NotSegmentedDataPath + l1);
        FileInfo[] info = dir.GetFiles("*.*");
        // Sort the files by name
        Array.Sort(info, delegate(FileInfo f1, FileInfo f2) {
            return f1.Name.CompareTo(f2.Name);
        });

        foreach (FileInfo f in info) 
        {
            // 1.
            PointCloud<PointXYZ> pointCloud = ReadPCDFile(f.FullName);
            PointCloud<PointXYZ> vehiclePointCloud = new PointCloud<PointXYZ>();
            PointCloud<PointXYZ> otherPointCloud = new PointCloud<PointXYZ>();

            // 2.
            SingleFrame frame = m_detectionResult.Data[m_idx];
            foreach (PointXYZ point in pointCloud.Points)
            {
                bool isVehiclePoint = false;
                // if in any, break the loop for the next point
                foreach(SingleBoundingBox box in frame.Boxes)
                {
                    if(IsPointInsideBox(
                        new Vector3(point.X, point.Y, point.Z), // point
                        box.Position,// center
                        box.Scale * 0.5f, // extends (half the scale)
                        box.Orientation// rotation
                    ) == true)
                    {
                        // TODO : mark point as vehicle
                        isVehiclePoint = true;
                        break;
                    }
                }

                if(isVehiclePoint == true)
                {
                    vehiclePointCloud.Add(point);
                }
                else
                {
                    otherPointCloud.Add(point);
                }
            }

            ++m_idx;

            ExportXYZIPointCloud(f.Name, vehiclePointCloud, otherPointCloud);
        }
    }

    private PointCloud<PointXYZ> ReadPCDFile(string pcdFilePath)
    {
        if(File.Exists(pcdFilePath) == false)
        {
            Debug.LogWarning(string.Format("PCD File : {0} does not exist!", pcdFilePath));
            return null;
        }

        m_pcdFilePath = pcdFilePath;
        PointCloud<PointXYZ> cloud = m_pcdReader.Read(m_pcdFilePath);
        return cloud;
    }

    // Ref : https://answers.unity.com/questions/1753765/check-if-point-is-inside-obb-without-collider.html
    private static bool IsPointInsideBox(Vector3 p, Vector3 center, Vector3 extends, Quaternion rotation)
    {
        Matrix4x4 m = Matrix4x4.TRS(center, rotation, Vector3.one);
        p = m.inverse.MultiplyPoint3x4(p);
        //p = rotation * p - center;
        return p.x <= extends.x && p.x > -extends.x 
            && p.y <= extends.y && p.y > -extends.y
            && p.z <= extends.z && p.z > -extends.z;
    }

    public static void ExportXYZIPointCloud(
            string fileName,
            PointCloud<PointXYZ> vehiclePoints,
            PointCloud<PointXYZ> otherPoints)
        {
            StreamWriter writer = new StreamWriter(
                GlobalSetting.OutputPCDPath + fileName);

            int validPointCount = vehiclePoints.Size + otherPoints.Size;

            writer.WriteLine("# .PCD v0.7 - Point Cloud Data file format");
            // Write info as water mark
            writer.WriteLine("# Labelled using Unity");
            //
            writer.WriteLine("VERSION 0.7");
            writer.WriteLine("FIELDS x y z intensity");
            writer.WriteLine("SIZE 4 4 4 4");
            writer.WriteLine("TYPE F F F F");
            writer.WriteLine("COUNT 1 1 1 1");

            writer.WriteLine("WIDTH " + validPointCount);
            writer.WriteLine("HEIGHT 1");
            writer.WriteLine("VIEWPOINT 0 0 0 1 0 0 0");
            writer.WriteLine("POINTS " + validPointCount);
            writer.WriteLine("DATA ascii");

            foreach(var point in vehiclePoints.Points)
            {
                writer.WriteLine(
                    string.Format("{0} {1} {2} {3}", 
                    point.X, point.Y, point.Z,
                    1.0f
                    ));
            }

            foreach(var point in otherPoints.Points)
            {
                writer.WriteLine(
                    string.Format("{0} {1} {2} {3}", 
                    point.X, point.Y, point.Z,
                    0.0f
                    ));
            }

            writer.Close();
        }

    [Serializable]

    private class DetectionResult
    {
        public SingleFrame [] Data = null;
    }

    [Serializable]
    private class SingleFrame
    {
        // public UInt64 TimeStamp = 0;
        public string TimeStamp = "";
        public SingleBoundingBox [] Boxes = null;
    }

    [Serializable]
    private class SingleBoundingBox
    {
        public int ID = 0;
        public Vector3 Position = Vector3.zero;
        public Quaternion Orientation = Quaternion.identity;
        public Vector3 Scale = Vector3.zero;
    }
}
