using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LPCSS;
// Binray formatter
using System.Runtime.Serialization.Formatters.Binary;

namespace LPCSS.ExportUtil
{
    public static class LidarPointCloudExporter
    {
        public static void ExportPointCloud(
            string fileName,
            ComputeBufferRaycastResultInfoStruct[] raycastResultInfoDataArray,
            string simulationParameterInfo = "")
        {
            StreamWriter writer = new StreamWriter(GlobalSetting.OutputPCDPath + fileName);

            // Count the number of valid points
            int validPointCount = 0;
            foreach(var entry in raycastResultInfoDataArray)
            {
                Vector3 transformed = GlobalSetting.TransformPointToWaymoSpace(entry.hitPosition);

                if( GlobalSetting.IsPointInComparingBoundingBox(transformed) == false )
                    continue;

                ++validPointCount;
            }

            writer.WriteLine("# .PCD v0.7 - Point Cloud Data file format");
            // Write info of simulation parameter as water mark
            writer.WriteLine("# Simulation parameters");
            writer.WriteLine("# " + simulationParameterInfo);
            writer.WriteLine("# End of simulation parameters");
            //
            writer.WriteLine("VERSION 0.7");
            
            writer.WriteLine("FIELDS x y z genTime initVelX initVelY initVelZ initPosX initPosY initPosZ genCode genIndex");
            writer.WriteLine("SIZE 4 4 4 4 4 4 4 4 4 4 4 4");
            writer.WriteLine("TYPE F F F I F F F F F F I I");
            writer.WriteLine("COUNT 1 1 1 1 1 1 1 1 1 1 1 1");

            writer.WriteLine("WIDTH " + validPointCount);
            writer.WriteLine("HEIGHT 1");
            writer.WriteLine("VIEWPOINT 0 0 0 1 0 0 0");
            writer.WriteLine("POINTS " + validPointCount);
            writer.WriteLine("DATA ascii");

            foreach(var entry in raycastResultInfoDataArray)
            {
                Vector3 transformed = GlobalSetting.TransformPointToWaymoSpace(entry.hitPosition);
                if( GlobalSetting.IsPointInComparingBoundingBox(transformed) == false )
                    continue;

                writer.WriteLine(
                    string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}", 
                    transformed.x, transformed.y, transformed.z,
                    entry.hitParticleGenerationTimeStamp,
                    entry.hitParticleInitialVelocity.x, entry.hitParticleInitialVelocity.y, entry.hitParticleInitialVelocity.z,
                    entry.hitParticleInitialPosition.x, entry.hitParticleInitialPosition.y, entry.hitParticleInitialPosition.z,
                    entry.hitParticleGenerationCode,
                    entry.hitParticleGenerationIndex
                    ));
            }

            writer.Close();
        }

        // This method exports sorted raycast result data for future filtering purposes.
        // The "filterHint" argument is a string that will be write to the file as well as a comment.
        public static void ExportFilter(
            string fileName,
            ComputeBufferRaycastResultInfoStruct[] raycastResultInfoDataArray,
            string simulationParameterInfo = "")
        {
            StreamWriter writer = new StreamWriter(GlobalSetting.OutputFilterPath + fileName);
            Debug.Log(GlobalSetting.OutputFilterPath + fileName);
            writer.WriteLine("# " + simulationParameterInfo);
            Array.Sort(raycastResultInfoDataArray, delegate(
                ComputeBufferRaycastResultInfoStruct lfs, ComputeBufferRaycastResultInfoStruct rhs)
            {
                return lfs.hitParticleGenerationTimeStamp.CompareTo(rhs.hitParticleGenerationTimeStamp);
            });

            foreach(var entry in raycastResultInfoDataArray)
            {
                if(entry.hitParticleGenerationTimeStamp < 0.0f)
                    continue;
                
                Vector3 transformed = GlobalSetting.TransformPointToWaymoSpace(entry.hitPosition);
                if( GlobalSetting.IsPointInComparingBoundingBox(transformed) == false )
                    continue;

                writer.WriteLine(
                    string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8}", 
                    entry.hitParticleGenerationTimeStamp,
                    entry.hitParticleGenerationCode,
                    entry.hitParticleGenerationIndex,
                    DatasackFormatting.FloatToHexString(entry.hitParticleInitialVelocity.x),
                    DatasackFormatting.FloatToHexString(entry.hitParticleInitialVelocity.y),
                    DatasackFormatting.FloatToHexString(entry.hitParticleInitialVelocity.z),
                    DatasackFormatting.FloatToHexString(entry.hitParticleInitialPosition.x),
                    DatasackFormatting.FloatToHexString(entry.hitParticleInitialPosition.y),
                    DatasackFormatting.FloatToHexString(entry.hitParticleInitialPosition.z)
                    ));

             
                // Remove duplicate lines from a file but leave 1 occurrence
                // Ref : https://unix.stackDexchange.com/questions/444795/remove-duplicate-lines-from-a-file-but-leave-1-occurrence
                // cat -n stuff.txt | sort -uk2 | sort -nk1 | cut -f2-
            }
            writer.Close();
        }
        public static void ExportPosition(
            string fileName,
            List<ParticleInfo> points,
            string simulationParameterInfo = "")
        {
            ComputeBufferRaycastResultInfoStruct[] raycastResultInfoDataArray = new ComputeBufferRaycastResultInfoStruct[points.Count];
            for (int rayIndex = 0; rayIndex < points.Count; ++rayIndex)
            {
                ComputeBufferRaycastResultInfoStruct raycastResultInfo = new ComputeBufferRaycastResultInfoStruct
                {
                    distance = 0.0f,
                    hitPosition = points[rayIndex].position,
                    hitParticleGenerationTimeStamp = points[rayIndex].generationTimeStamp
                };
                raycastResultInfoDataArray[rayIndex] = raycastResultInfo;
            }
            StreamWriter writer_pos = new StreamWriter("C:\\Users\\ylwhxht\\Desktop\\output\\infoposition\\" + fileName);
            Debug.Log("C:\\Users\\ylwhxht\\Desktop\\output\\infoposition\\" + fileName);
            Array.Sort(raycastResultInfoDataArray, delegate (
                ComputeBufferRaycastResultInfoStruct lfs, ComputeBufferRaycastResultInfoStruct rhs)
            {
                return lfs.hitParticleGenerationTimeStamp.CompareTo(rhs.hitParticleGenerationTimeStamp);
            });

            foreach (var entry in raycastResultInfoDataArray)
            {
                if (entry.hitParticleGenerationTimeStamp < 0.0f)
                    continue;

                Vector3 transformed = GlobalSetting.TransformPointToWaymoSpace(entry.hitPosition);
                if (GlobalSetting.IsPointInComparingBoundingBox(transformed) == false)
                    continue;

                writer_pos.WriteLine(string.Format("{0} {1} {2}", transformed.x, transformed.y, transformed.z));

                // Remove duplicate lines from a file but leave 1 occurrence
                // Ref : https://unix.stackexchange.com/questions/444795/remove-duplicate-lines-from-a-file-but-leave-1-occurrence
                // cat -n stuff.txt | sort -uk2 | sort -nk1 | cut -f2-
            }
            writer_pos.Close();
        }
    }
}