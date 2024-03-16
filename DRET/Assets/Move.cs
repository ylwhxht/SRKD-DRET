using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Move : MonoBehaviour
{
    [SerializeField]
    private Vector3 m_Velocity = Vector3.zero;
    [SerializeField]
    private float m_AngularVelocity = 0.0f;

    private void FixedUpdate()
    {
        transform.position += transform.TransformDirection(m_Velocity) * Time.fixedDeltaTime;
        transform.Rotate(Vector3.up, m_AngularVelocity * Time.fixedDeltaTime);
    }
}
