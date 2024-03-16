using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace LPCSS
{
    public static class GlobalSetting
    {
        public static string OutputPCDPath;
        public static string OutputFilterPath;
        public static string SimAugmentConfig;
        public static string SprayDetectionJSONPath;
        public static string NotSegmentedDataPath;
        public static string ManipulationCheckConfig;
        public static string ReconstructRefPath; // OtherVeh_SpraySegmentation/";
        public static string IntegralPath;
        public static string IntegralJSONPath;
        public static string DefaultFilter;
        public static float GetGroundPlaneHeight()
        {
            return -1.0f;
        }

        public static float GetLidarScanTimeStep()
        {
            return 0.1f;
        }

        public static Vector3 TransformPointToWaymoSpace(Vector3 point)
        {
            return new Vector3(
                // 1.353f = -1.154f - (-2.507f) = rear_lidar_pos_in_waymo - unity_car_back
                point.z + 1.353f,
                -1.0f * point.x,
                point.y - GetGroundPlaneHeight());
        }

        public static Vector3 TransformPointFromWaymoSpace(Vector3 point)
        {
            return new Vector3(
                -1.0f * point.y,
                point.z + GetGroundPlaneHeight(),
                point.x - 1.353f);
        }

        public static Vector3 TransformDirectionToWaymoSpace(Vector3 direction)
        {
            // Waymo orientation to Unity orientation
            // -y > x, z > y, x > z
            return new Vector3(
                direction.z,
                -1.0f * direction.x,
                direction.y);
        }

        public static Vector3 TransformDirectionFromWaymoSpace(Vector3 direction)
        {
            return new Vector3(
                -1.0f * direction.y,
                direction.z,
                direction.x);
        }

        public static Quaternion TransformQuaternionToWaymoSpace(Quaternion quat)
        {
            // Waymo orientation to Unity orientation
            // -y > x, z > y, x > z
            return new Quaternion(
                quat.z,
                -1.0f * quat.x,
                quat.y,
                -quat.w);
        }

        public static Quaternion TransformQuaternionFromWaymoSpace(Quaternion quat)
        {
            return new Quaternion(
                -1.0f * quat.y,
                quat.z,
                quat.x,
                -quat.w);
        }

        public static Vector3 TransformScaleToWaymoSpace(Vector3 scale)
        {
            return new Vector3(
                scale.z,
                scale.x,
                scale.y);
        }

        public static Vector3 TransformScaleFromWaymoSpace(Vector3 scale)
        {
            return new Vector3(
                scale.y,
                scale.z,
                scale.x);
        }

        // Given a bounding box of an arbitrary vehicle in Waymo data, return a position
        // where when place a simulating vehicle, its back will align with the back of
        // the given bounding box. Notice that here we expect data in Waymo space.
        public static Vector3 AlignBackOfBoundingBox(Vector3 position, Vector3 scale)
        {
            position.x += 2.507f - scale.x * 0.5f;
            return position;
        }

        public static Vector3 CancelEgomotionOffset(Vector3 point, float egoVehicleVelocity)
        {
            // Waymo Top Lidar Transform
            // 
            // translation: 
            //     x: 1.43
            //     y: 0.0
            //     z: 2.184
            

            Vector3 lidarPositionInWaymo = TransformPointFromWaymoSpace(new Vector3(
                1.43f,
                0.0f,
                2.184f
            ));
            
            Vector3 direction = point - lidarPositionInWaymo;
            float azimuth = Mathf.Atan2(direction.x, direction.z);

            point.z += 
                -1.0f * GetLidarScanTimeStep() * (azimuth / (2.0f * Mathf.PI)) * egoVehicleVelocity;

            return point;
        }

        public static bool IsPointConsideredGround(Vector3 waymoSpacePoint)
        {
            return (waymoSpacePoint.z < 0.2f);
        }

        // Please notice that this API expects the input point "transformed" to Waymo space, not in
        // Unity world space
        public static bool IsPointInComparingBoundingBox(Vector3 transformed)
        {
            if (transformed.z <= -2.0f)
                return false;

            // Different car with different size(i.e. bounding box)
            // if(transformed.y > 8.5f || transformed.y < -8.5f)
            //     return false;
            // // front and back constaint
            // if(transformed.x > 10.0f || transformed.x < -75.0f)
            //     return false;

            // if(transformed.z < 0.2f)
            // {
            //     // left and right constraint
            //     if(transformed.y > 8.5f || transformed.y < -8.5f)
            //         return false;
            //     // front and back constaint
            //     if(transformed.x > 10.0f || transformed.x < -75.0f)
            //         return false;
            // }
            // else
            // {
            //     // left and right constraint
            //     if(transformed.y > 2.5f || transformed.y < -2.5f)
            //         return false;
            //     // front and back constaint
            //     if(transformed.x > 10.0f || transformed.x < -75.0f)
            //         return false;
            // }

            return true;
        }
        private static string DotEnvPath = "./Assets/Scripts/.env";
        static GlobalSetting()
        {
            foreach (var line in File.ReadAllLines(GlobalSetting.DotEnvPath))
            {
                var parts = line.Split('=', (char) StringSplitOptions.RemoveEmptyEntries);

                if (parts.Length != 2)
                    continue;

                switch(parts[0])
                {
                    case "OutputPCDPath":
                        OutputPCDPath = parts[1];
                        break;
                    case "OutputFilterPath":
                        OutputFilterPath = parts[1];
                        break;
                    case "SimAugmentConfig":
                        SimAugmentConfig = parts[1];
                        break;
                    case "SprayDetectionJSONPath":
                        SprayDetectionJSONPath = parts[1];
                        break;
                    case "NotSegmentedDataPath":
                        NotSegmentedDataPath = parts[1];
                        break;
                    case "ManipulationCheckConfig":
                        ManipulationCheckConfig = parts[1];
                        break;
                    case "ReconstructRefPath":
                        ReconstructRefPath = parts[1];
                        break;
                    case "IntegralPath":
                        IntegralPath = parts[1];
                        break;
                    case "IntegralJSONPath":
                        IntegralJSONPath = parts[1];
                        break;
                    case "DefaultFilter":
                        DefaultFilter = parts[1];
                        break;
                    default: break;                    
                }
            }
        }

    }
}