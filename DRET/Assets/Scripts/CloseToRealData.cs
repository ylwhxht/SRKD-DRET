using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using PointCloud;
using PointCloud.io;

using DataStructures.ViliWonka.KDTree;

using LPCSS;

public class CloseToRealData : MonoBehaviour
{
    private KDTree m_kdtree = null;
    private List<Vector3> m_pointList = new List<Vector3>();
    private List<Vector3> m_offsetedPointList = new List<Vector3>();
    public List<Vector3> PointList
    {
        get { return m_pointList; }
    }

    string m_pcdFilePath = null;
    PCDReader<PointXYZ> m_pcdReader = new PCDReader<PointXYZ>();

    // this can be called externally to fit multiple pcd files in one single simulation run
    public void InitializeWithPCDFile(string pcdFilePath, float egoVehicleVelocity)
    {
        pcdFilePath = "C:\\Users\\ylwhxht\\Desktop\\SpraySimulation\\seg-pc\\self\\"+Random.Range(0, 2100).ToString()+".pcd";
        Debug.Log(pcdFilePath);
        if(File.Exists(pcdFilePath) == false)
        {
            Debug.LogError(string.Format("PCD File : {0} does not exist!", pcdFilePath));
            return;
        }
        
        m_pcdFilePath = pcdFilePath;
        
        PointCloud<PointXYZ> cloud = m_pcdReader.Read(m_pcdFilePath);

        m_pointList.Clear();
        m_offsetedPointList.Clear();
        m_kdtree = null;

        foreach (var entry in cloud.Points)
        {
            Vector3 point = new Vector3(entry.X, entry.Y, entry.Z);





            m_pointList.Add( 
                GlobalSetting.CancelEgomotionOffset(
                    GlobalSetting.TransformPointFromWaymoSpace(point), egoVehicleVelocity));

            m_offsetedPointList.Add(
                    GlobalSetting.TransformPointFromWaymoSpace(point));
        }
        
        m_kdtree = new KDTree(m_pointList.ToArray());
    }

    public string GetCurrentFittingPCDFilePath()
    {
        return m_pcdFilePath;
    }

    public bool IsCloseToRealPoint(Vector3 position, KDQuery query, List<int> queryResults)
    {
        if(m_kdtree == null)
        {
            Debug.LogWarning("KD Tree not set yet!");
            return false;
        }

        queryResults.Clear();

        // further particles can have larger tolerance
        // float radius = Mathf.Lerp(0.1f, 0.5f, Mathf.Abs(position.z / 15.0f));
        float radius = 0.2f;

        query.Radius(m_kdtree, position, radius, queryResults);
        if(queryResults.Count > 1)
            return true;
        
        return false;
    }
}
