using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DataStructures.ViliWonka.KDTree;

namespace LPCSS
{
    public class WindFieldLoader : MonoBehaviour
    {
        [SerializeField]
        private TextAsset m_WindFieldData = null;

        private Dictionary<Vector3, Vector3> m_positionToWindVelocityMap = new Dictionary<Vector3, Vector3>();
        private List<Vector3> m_dataPositions = new List<Vector3>();
        private KDTree m_kdtree = null;
        /*
        private static Vector3 size = new Vector3(0.05f, 0.05f, 0.05f);
        private static List<int> m_queryResults = new List<int>();
        private static KDQuery m_query = new KDQuery();
        private void OnDrawGizmos()
        {
            Color color = Color.grey;
            for (int i = 0; i < m_dataPositions.Count; i++)
            {
                Gizmos.DrawCube(m_dataPositions[i], size);
            }

            if (m_kdtree != null)
            {
                m_queryResults.Clear();
                Vector3 pos = transform.position;
                if (pos.x > 0) pos.x *= -1.0f;
                m_query.ClosestPoint(m_kdtree, pos, m_queryResults);
            }

            if (m_queryResults.Count > 0)
            {
                Gizmos.color = Color.blue;
                Gizmos.DrawCube(m_dataPositions[m_queryResults[0]], size * 5.0f);
                Debug.Log(m_positionToWindVelocityMap[m_dataPositions[m_queryResults[0]]]);

                m_query.DrawLastQuery();
            }
        }
        */
        
        // Add utility to show the imported data as point cloud
        private void Awake()
        {
            // Load the .csv file exported from SimFlow
            string allText = m_WindFieldData.text;
            string[] lines = allText.Split('\n');

            // Skip idx 0 as it is the header line in the .csv file
            for (int idx = 1; idx < lines.Length - 1; ++idx)
            {
                string[] data = lines[idx].Split(',');
                Vector3 position, windVelocity;
                ParseEntry(data, out position, out windVelocity);

                // According to Ref (https://docs.microsoft.com/zh-tw/dotnet/api/system.collections.generic.dictionary-2.add?view=netcore-3.1)
                // Operator [] will overwrite old value if the given key already exists in the Dictionary
                // Add() will throw an exception instead
                m_positionToWindVelocityMap.Add(position, windVelocity);
                m_dataPositions.Add(position);
            }

            // Initialize KDTree
            m_kdtree = new KDTree(m_dataPositions.ToArray());
        }
        
        public Vector3 GetWindVelocity(KDQuery query, List<int> queryResult, Vector3 position)
        {
            queryResult.Clear();
            
            bool isXFlipped = false;
            if (position.x > 0.0f)
            {
                position.x *= -1.0f;
                isXFlipped = true;
            }
            query.ClosestPoint(m_kdtree, position, queryResult);
            
            Vector3 closestPoint = m_dataPositions[queryResult[0]];
            Vector3 windVelocity = m_positionToWindVelocityMap[closestPoint];
            if (isXFlipped == true)
                windVelocity.x *= -1.0f;
            
            return windVelocity;
        }

        static private void ParseEntry(string[] data, out Vector3 position, out Vector3 windVelocity)
        {
            // WindVelocity at 0, 1, 2
            // Position at 7, 8, 9
            float posX;
            if (float.TryParse(data[7], out posX) != true)
                Debug.LogError(string.Format("Error when parsing .csv file for string : {0}", data));
            float posY;
            if (float.TryParse(data[8], out posY) != true)
                Debug.LogError(string.Format("Error when parsing .csv file for string : {0}", data));
            float posZ;
            if (float.TryParse(data[9], out posZ) != true)
                Debug.LogError(string.Format("Error when parsing .csv file for string : {0}", data));

            float windVelX;
            if (float.TryParse(data[0], out windVelX) != true)
                Debug.LogError(string.Format("Error when parsing .csv file for string : {0}", data));
            float windVelY;
            if (float.TryParse(data[1], out windVelY) != true)
                Debug.LogError(string.Format("Error when parsing .csv file for string : {0}", data));
            float windVelZ;
            if (float.TryParse(data[2], out windVelZ) != true)
                Debug.LogError(string.Format("Error when parsing .csv file for string : {0}", data));

            position.x = posX;
            position.y = posY;
            position.z = posZ;

            // position.x = posX;
            // position.y = posY + GlobalSetting.GetGroundPlaneHeight();
            // position.z = posZ;

            windVelocity.x = windVelX;
            windVelocity.y = windVelY;
            windVelocity.z = windVelZ;
        }
    }
}
