using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LPCSS;
public class DetectionResultParser : MonoBehaviour
{
    // [SerializeField]
    // private TextAsset m_DetectionJson = null;
    [SerializeField]
    private GameObject m_VehiclePrefab = null;
    [SerializeField]
    private Lidar m_LidarReference = null;
    [SerializeField]
    private WindFieldLoader m_WindFieldLoader = null;
    [SerializeField]
    private CloseToRealData m_CloseToRealData = null;

    private DetectionResult m_detectionResult = null;
    private Dictionary<int, Transform> m_vehicles = new Dictionary<int, Transform>();
    private Dictionary<int, Vector3> m_nextPos = new Dictionary<int, Vector3>();
    private Dictionary<int, Quaternion> m_nextOri = new Dictionary<int, Quaternion>();
    private float m_timer = 0.0f;
    private int m_idx = 0;
    private float egoVel = 0.0f;

    private void Awake()
    {
        // Read the config
        StreamReader reader = new StreamReader(GlobalSetting.SimAugmentConfig);
        string allText = reader.ReadToEnd();
        string[] lines = allText.Split('\n');
        // first line the json file, second line the ego-vehicle velocity
        StreamReader jsonFileReader = new StreamReader(
            GlobalSetting.SprayDetectionJSONPath + lines[0] + ".json");
        m_detectionResult = JsonUtility.FromJson<DetectionResult>(jsonFileReader.ReadToEnd());
        // second line the ego-vehicle velocity
        Debug.Log(m_detectionResult.Data.Length);
        if(float.TryParse(lines[1], out egoVel) == false)
        {
            Debug.LogError("float.TryParse Failed for string : " + lines[1]);
        }
        m_LidarReference.EgoVehicleVelocity = egoVel;
        // Lastly, override the output directory and create the directory
        GlobalSetting.OutputPCDPath += lines[0] + "/";
        System.IO.Directory.CreateDirectory(GlobalSetting.OutputPCDPath);
        
        // Remove the last entry in Data
        Array.Resize(ref m_detectionResult.Data, m_detectionResult.Data.Length - 1);
        // Remove the last entry in all Boxes array
        //foreach (var e in m_detectionResult.Data)
        //{
         //   Array.Resize(ref e.Boxes, e.Boxes.Length - 1);
        //}
        FixedUpdate();
        //UpdateToNextFrame();
    }

    private void FixedUpdate()
    {
        // Wait until the lidar hits the 15th point cloud, so we actually start updating
        //if(m_LidarReference.CurrentPointCloudIndex < 16) return;

        // we lerp from m_idx to m_idx+1, so if m_idx is the last element, we stop lerping
        if (m_idx >= m_detectionResult.Data.Length - 1) return;

        float ratio = m_timer / GlobalSetting.GetLidarScanTimeStep();
        // Perform Lerp on position and orientation
        foreach(var e in m_detectionResult.Data[m_idx].Boxes)
        {
            if(m_vehicles.ContainsKey(e.ID) == false) continue;

            Transform trans = m_vehicles[e.ID];
            trans.position = Vector3.Lerp(
                GlobalSetting.CancelEgomotionOffset(
                    GlobalSetting.TransformPointFromWaymoSpace(
                        GlobalSetting.AlignBackOfBoundingBox(e.Position, e.Scale)), m_LidarReference.EgoVehicleVelocity),
                m_nextPos[e.ID],
                ratio
            );

            trans.position = new Vector3(trans.position.x, Math.Max(trans.position.y, 0.0f), trans.position.z);

            trans.rotation = Quaternion.Slerp(
                GlobalSetting.TransformQuaternionFromWaymoSpace(e.Orientation),
                m_nextOri[e.ID],
                ratio
            );
            // trans.localScale = GlobalSetting.TransformScaleFromWaymoSpace(e.Scale);
        }

        if(m_timer >= GlobalSetting.GetLidarScanTimeStep())
        {
            m_timer = 0.0f;
            ++m_idx;

            UpdateToNextFrame();
        }
        m_timer += Time.fixedDeltaTime;
    }

    private void UpdateToNextFrame()
    {
        
        if (m_idx >= m_detectionResult.Data.Length - 1) return;

        foreach(var e in m_detectionResult.Data[m_idx].Boxes)
        {
            //filter all other vehicles
            //if(e.ID != 1953445495) continue;

            if(m_vehicles.ContainsKey(e.ID) == false)
            {
                // Transform all the transformations from Waymo space to Unity space
                // GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                GameObject cube = Instantiate(m_VehiclePrefab);
                cube.name = e.ID.ToString();
                cube.transform.position = GlobalSetting.CancelEgomotionOffset(
                                            GlobalSetting.TransformPointFromWaymoSpace(
                                                GlobalSetting.AlignBackOfBoundingBox(e.Position, e.Scale)), m_LidarReference.EgoVehicleVelocity);
                cube.transform.position = new Vector3(cube.transform.position.x, Math.Max(cube.transform.position.y, 0.0f), cube.transform.position.z);
                cube.transform.rotation = GlobalSetting.TransformQuaternionFromWaymoSpace(e.Orientation);
                // cube.transform.localScale = GlobalSetting.TransformScaleFromWaymoSpace(e.Scale);

                // if the orientation is almost perpendicular to ego, ignore the sprayModel
                if(Math.Abs(Vector3.Dot(cube.transform.forward, Vector3.forward)) <= 0.2f)
                {
                    Destroy(cube);
                    continue;
                }

                cube.GetComponent<SprayModel>().SetWindFieldLoader(
                    m_WindFieldLoader
                );
                cube.GetComponent<SprayModel>().SetCloseToRealData(
                    m_CloseToRealData
                );
                m_LidarReference.RegisterSprayModel(cube.GetComponent<SprayModel>());
                m_vehicles.Add(e.ID, cube.transform);
            }
            else
            {
                Transform trans = m_vehicles[e.ID];
                trans.position = GlobalSetting.CancelEgomotionOffset(
                                    GlobalSetting.TransformPointFromWaymoSpace(
                                        GlobalSetting.AlignBackOfBoundingBox(e.Position, e.Scale)), m_LidarReference.EgoVehicleVelocity);
                trans.position = new Vector3(trans.position.x, Math.Max(trans.position.y, 0.0f), trans.position.z);
                trans.rotation = GlobalSetting.TransformQuaternionFromWaymoSpace(e.Orientation);
                // trans.localScale = GlobalSetting.TransformScaleFromWaymoSpace(e.Scale);
            }
        }

        // Collect IDs of all vehicles that should be removed 
        // (the vehicles that does not have a next frame transform)
        // This need to be after the above stage, otherwise the removed vehicle is immediately
        // added back to the collection.
        List<int> removingVehicleIDs = new List<int>();
        foreach(var vehicleID in m_vehicles.Keys)
        {
            bool isStillExisting = false;
            foreach(var e in m_detectionResult.Data[m_idx + 1].Boxes)
            {
                if(e.ID == vehicleID)
                {
                    isStillExisting = true;
                    break;
                }
            }
            if(isStillExisting == false)
                removingVehicleIDs.Add(vehicleID);
        }
        // Actuall remove the vehicle
        foreach (var vehicleID in removingVehicleIDs)
        {
            // TODO : We need to also un-register the newly instantiated gameObject to the Lidar
            m_LidarReference.UnregisterSprayModel(m_vehicles[vehicleID].GetComponent<SprayModel>());

            // Remember to destroy the gameObject before removing the entry from the dictionary
            Destroy(m_vehicles[vehicleID].gameObject);
            m_vehicles.Remove(vehicleID);
        }

        // Update the cached m_nextPos and m_nextOri
        m_nextPos.Clear();
        m_nextOri.Clear();
        foreach(var e in m_detectionResult.Data[m_idx + 1].Boxes)
        {
            
            m_nextPos.Add(e.ID, GlobalSetting.CancelEgomotionOffset(
                GlobalSetting.TransformPointFromWaymoSpace(
                    GlobalSetting.AlignBackOfBoundingBox(e.Position, e.Scale)), m_LidarReference.EgoVehicleVelocity)
            );
            m_nextOri.Add(e.ID, GlobalSetting.TransformQuaternionFromWaymoSpace(e.Orientation));
        }
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
