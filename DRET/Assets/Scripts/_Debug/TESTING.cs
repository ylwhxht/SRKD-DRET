using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class TESTING : MonoBehaviour
{
    [SerializeField]
    private List<Transform> m_Targets = null;

    private void OnDrawGizmos()
    {
        Color color = Color.blue;
        color.a = 1.0f;
        Gizmos.color = color;

        // Gizmos.DrawCube(pos), pos is the center of the cube
        // Gizmos.DrawCube(Vector3.up * 2.0f, size);

        Vector3[] vertices = new Vector3[]
        {
            new Vector3( 0.5f,  0.5f, 0.0f),
            new Vector3(-0.5f,  0.5f, 0.0f),
            new Vector3(-0.5f, -0.5f, 0.0f),
            new Vector3( 0.5f, -0.5f, 0.0f)
        };

        Vector3 scale = new Vector3(0.195f, 0.16f, 1.0f);
        Quaternion rotation = Quaternion.Euler(-60.0f, 0.0f, 0.0f);
        // Quaternion rotation = Quaternion.AngleAxis(-60.0f, Vector3.right);
        Vector3 translation = new Vector3(0.0f, 0.04f, -0.2f);

        string singleLog = "";

        foreach (Transform trans in m_Targets)
        {
            singleLog += trans.name + "\n { ";
            foreach (Vector3 vertex in vertices)
            {
                Matrix4x4 m = Matrix4x4.TRS(translation, rotation, scale);

                var worldPosVertex = trans.TransformPoint(Vector3.zero);
                worldPosVertex += m.MultiplyPoint3x4(vertex);

                Gizmos.DrawSphere(worldPosVertex, 0.01f);

                // Matrix4x4 mtmp = Matrix4x4.TRS(translation, Quaternion.identity, scale);
                // Gizmos.DrawCube(trans.TransformPoint(mtmp.MultiplyPoint3x4(vertex)), size);
                // Debug.Log(worldPosVertex.ToString("F6"));
                singleLog += worldPosVertex.ToString("F6") + " , \n";
            }
            singleLog += " } \n";
        }

        // Debug.Log(singleLog);
        // Gizmos.DrawSphere(new Vector3(-0.7365f, -0.796413f, -1.652282f), 0.01f);
    }
}
