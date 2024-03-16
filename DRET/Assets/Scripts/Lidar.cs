using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using LPCSS.ExportUtil;

// Reference : https://github.com/vwaurich/VelodyneLidarUnitySimulation/blob/master/Scripts/Lidar.cs

namespace LPCSS
{
    using ComputeBufferParticleInfoStruct = ParticleInfo;

    public class Lidar : MonoBehaviour
    {
        [Header("SprayModelReference")]
        [SerializeField]
        private SprayModel m_EgoVehicleSprayModel = null;
        [SerializeField]
        private List<SprayModel> m_SprayModels = null;

        [Header("RaycastMeshColliderReference")]
        [SerializeField]
        private List<MeshCollider> m_MeshColliderList = null;

        [Header("RealDataComparatorReference")]
        [Tooltip("Lidar script should update this object to read different pcd files every time lidar updates")]
        [SerializeField]
        private CloseToRealData m_CloseToRealData = null;

        [Header("Lidar Parameters")]
        [SerializeField]
        private TextAsset m_InclinationData = null;
        [SerializeField]
        private int m_NumberOfCircularSegments = 360;
        [SerializeField]
        private float m_MaxRange = 75.0f;
        [SerializeField]
        [Range(0.0f, 1.0f)]
        private float m_GeneralDropOffRate = 0.45f;
        [SerializeField]
        [Tooltip("Control whether the simulator fits to a single pcd file or multiple pcd files")]
        private PCDFittingConfig m_pcdFittingConfig = PCDFittingConfig.FitToMultiplePCD;
        [SerializeField]
        [Tooltip("Simulator starts to fit to real data at this particular frame.")]
        private int m_fittingOffset = 15;

        [Header("Render Entities")]
        [SerializeField]
        private Material m_RenderMaterial = null;
        [SerializeField]
        private Mesh m_PointCloudEntityMesh = null;

        [Header("Compute Shaders")]
        [SerializeField]
        private ComputeShader m_RaycastComputeShader = null;

        [Header("Export Config")]
        [SerializeField]
        private int m_StoppingFrameCount = 220;
        [SerializeField]
        private bool m_IsExportingPointCloud = true;
        [SerializeField]
        private bool m_IsExportingFilter = false;

        private bool m_IsExportPosition = true;

        // Lidar simulation data
        private float m_azimuthIncrementAngle;
        private List<float> m_inclinationAngles = new List<float>();

        private const int particleInfoSize = 50000 * 200;
        private const int triangleInfoSize = 50000 * 200;

        private int m_rayCount = 0;
        private int m_triangleCount = 0;

        private ComputeBuffer m_particleInfoBuffer = null;
        private ComputeBuffer m_triangleInfoBuffer = null;
        private ComputeBuffer m_raycastInfoBuffer = null;
        private ComputeBuffer m_raycastResultInfoBuffer = null;


        ComputeBufferParticleInfoStruct[] m_particleInfoDataArray = null;
        ComputeBufferTriangleInfoStruct[] m_triangleInfoDataArray = null;
        ComputeBufferRaycastInfoStruct[] m_raycastInfoDataArray = null;
        ComputeBufferRaycastResultInfoStruct[] m_raycastResultInfoDataArray = null;
        //

        // Rendering point cloud
        private List<Vector3> m_pointCloudPositions = new List<Vector3>();
        private ComputeBuffer argsBuffer;
        private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };

        // Exporting point cloud

        private int m_currentPointCloudIndex = 0;
        public int CurrentPointCloudIndex { get { return m_currentPointCloudIndex; } }

        //

        public float EgoVehicleVelocity
        {
            get { return m_EgoVehicleSprayModel.VehicleVelocity; }
            set { m_EgoVehicleSprayModel.VehicleVelocity = value; }
        }

        public void RegisterSprayModel(SprayModel sprayModel)
        {
            m_SprayModels.Add(sprayModel);
            m_MeshColliderList.Add(sprayModel.transform.Find("FamilyCarChassis").GetComponent<MeshCollider>());
        }

        public void UnregisterSprayModel(SprayModel sprayModel)
        {
            m_SprayModels.Remove(sprayModel);
            m_MeshColliderList.Remove(sprayModel.transform.Find("FamilyCarChassis").GetComponent<MeshCollider>());
        }

        private void Awake()
        {
            m_azimuthIncrementAngle = (360.0f / m_NumberOfCircularSegments);

            InitializeInclinationAngles();
            InitializeComputeBuffer();
        }

        private void OnDestroy()
        {
            if (m_particleInfoBuffer != null)
                m_particleInfoBuffer.Release();
            if (m_triangleInfoBuffer != null)
                m_triangleInfoBuffer.Release();
            if (m_raycastInfoBuffer != null)
                m_raycastInfoBuffer.Release();
            if (m_raycastResultInfoBuffer != null)
                m_raycastResultInfoBuffer.Release();
            if (argsBuffer != null)
                argsBuffer.Release();
        }

        private float m_UpdateLidarTimer = 0.0f;
        // private float m_UpdateLidarTimer = -0.05f;
        private void FixedUpdate()
        {
            if (m_UpdateLidarTimer >= GlobalSetting.GetLidarScanTimeStep())
            {
                IRUpdate();
                m_UpdateLidarTimer = 0.0f;
            }
            m_UpdateLidarTimer += Time.fixedDeltaTime;
        }

        private void IRUpdate()
        {
            if (m_EgoVehicleSprayModel.VisibleParticleConfig == VisibleParticleConfig.CloseToRealData)
            {
                if (m_pcdFittingConfig == PCDFittingConfig.FitToMultiplePCD)
                {
                    // m_CloseToRealData.InitializeWithPCDFile(
                    //     string.Format("../../Desktop/Research/LidarDataset/PCD_Ascii_Renamed/NotSegmented/" +
                    //         "segment-13830510593707564159_renamed/segment-13830510593707564159_{0}.pcd",
                    //         (m_currentPointCloudIndex - m_fittingOffset).ToString("D4")),
                    //         m_EgoVehicleSprayModel.VehicleVelocity
                    // );

                    m_CloseToRealData.InitializeWithPCDFile(
                        string.Format(GlobalSetting.ReconstructRefPath + "/segment-13830510593707564159.pcd",
                            (m_currentPointCloudIndex - m_fittingOffset).ToString("D4")),
                        m_EgoVehicleSprayModel.VehicleVelocity
                    );
                }
            }

            UpdateLidarDetection();

            m_RenderMaterial.SetBuffer("raycastResultInfoBuffer", m_raycastResultInfoBuffer);

            args[0] = (uint)m_PointCloudEntityMesh.GetIndexCount(0);
            args[1] = (uint)m_raycastResultInfoDataArray.Length;
            args[2] = (uint)m_PointCloudEntityMesh.GetIndexStart(0);
            args[3] = (uint)m_PointCloudEntityMesh.GetBaseVertex(0);
            argsBuffer.SetData(args);

            Graphics.DrawMeshInstancedIndirect(
                m_PointCloudEntityMesh,
                0,
                m_RenderMaterial,
                new Bounds(Vector3.zero,
                new Vector3(100.0f, 100.0f, 100.0f)),
                argsBuffer);

            // Export lidar detection result as .pcd files
            if (m_IsExportingPointCloud == true)
            {
                LidarPointCloudExporter.ExportPointCloud(
                    string.Format("result{0}.pcd", m_currentPointCloudIndex.ToString("D4")),
                    m_raycastResultInfoDataArray,
                    m_EgoVehicleSprayModel.GetParameterInfoString());
            }

            if (m_IsExportingFilter == true)
            {
                if (m_currentPointCloudIndex >= m_fittingOffset)
                {
                    LidarPointCloudExporter.ExportFilter(
                        string.Format("filter{0}.txt", m_currentPointCloudIndex.ToString("D4")),
                        m_raycastResultInfoDataArray,
                        m_EgoVehicleSprayModel.GetParameterInfoString());
                }
            }

            if (m_currentPointCloudIndex >= m_StoppingFrameCount)
            {
                Application.Quit();
            }

            ++m_currentPointCloudIndex;
        }

        // This loop will update the distance array
        private void UpdateLidarDetection()
        {
            // Update collider info buffers, particles and triangle mesh 
            List<ParticleInfo> points = new List<ParticleInfo>();
            points.AddRange(m_EgoVehicleSprayModel.AllParticleInfo);
            foreach (var sprayModel in m_SprayModels)
            {
                //foreach(var point in sprayModel.AllParticleInfo)
                //{
                //    Debug.Log(point.generationCode);
                //}
                //Debug.Log(sprayModel.m_TireSpraySourceList.Length);
                points.AddRange(sprayModel.AllParticleInfo);
            }
            
            UpdateParticleBuffer(points);
            int triangleCount = UpdateTriangleBuffer();

            // Initialize the particleInfo compute buffer data
            m_RaycastComputeShader.SetFloat("maxRange", m_MaxRange + 0.005f);
            m_RaycastComputeShader.SetFloat("time", Time.time);
            m_RaycastComputeShader.SetFloat("generalDropOffRate", m_GeneralDropOffRate);
            m_RaycastComputeShader.SetFloat("vehicleVelocity", m_EgoVehicleSprayModel.VehicleVelocity);
            // This should match the rotation frequency of the simulating lidar, which in waymo is 10 hz,
            // which makes the time spent on a single lidar scan 0.1 seconds.
            m_RaycastComputeShader.SetFloat("lidarScanTime", GlobalSetting.GetLidarScanTimeStep());
            m_RaycastComputeShader.SetInt("triangleCount", triangleCount);
            m_RaycastComputeShader.SetInt("particleCount", points.Count);

            if (m_IsExportPosition == true)
            {
                if (m_currentPointCloudIndex >= m_fittingOffset)
                {
                    LidarPointCloudExporter.ExportPosition(
                        string.Format("position{0}.txt", m_currentPointCloudIndex.ToString("D4")),
                        points,
                        m_EgoVehicleSprayModel.GetParameterInfoString());
                }
            }

            

            //////////////////// UPDATE : for mani check, remove if using for recon or synthesis
            // m_RaycastComputeShader.SetFloat("particleCollideRadius", 0.035f);
            m_RaycastComputeShader.SetFloat("particleCollideRadius", m_EgoVehicleSprayModel.ParticleSize / 30.0f * 0.035f );
            ////////////////////

            // Set the random offsets needed for the gaussrand in compute shader
            m_RaycastComputeShader.SetVector("offsets", new Vector4(
                Random.Range(-1.0f, 1.0f),
                Random.Range(-1.0f, 1.0f),
                Random.Range(-1.0f, 1.0f),
                Random.Range(-1.0f, 1.0f)
            ));
            m_RaycastComputeShader.SetFloat("groundHeight", GlobalSetting.GetGroundPlaneHeight());

            // Call compute shader, assign data to m_distances
            int computeShaderEntry = m_RaycastComputeShader.FindKernel("CSMain");
            m_RaycastComputeShader.SetBuffer(computeShaderEntry, "particleInfoBuffer", m_particleInfoBuffer);
            if (m_triangleInfoBuffer != null)
            {
                m_RaycastComputeShader.SetBuffer(computeShaderEntry, "triangleInfoBuffer", m_triangleInfoBuffer);
            }
            m_RaycastComputeShader.SetBuffer(computeShaderEntry, "raycastInfoBuffer", m_raycastInfoBuffer);
            m_RaycastComputeShader.SetBuffer(computeShaderEntry, "raycastResultInfoBuffer", m_raycastResultInfoBuffer);

            m_RaycastComputeShader.Dispatch(computeShaderEntry, m_rayCount / 64, 1, 1);

            // Update computed data
            m_raycastResultInfoBuffer.GetData(m_raycastResultInfoDataArray);
        }

        private void InitializeInclinationAngles()
        {
            string allText = m_InclinationData.text;
            string[] lines = allText.Split('\n');

            // Skip idx 0 as it is the header line in the .csv file
            for (int idx = 0; idx < lines.Length; ++idx)
            {
                string data = lines[idx];
                float inclination = -1.0f;
                if (float.TryParse(data, out inclination) != true)
                {
                    Debug.LogError(string.Format("Error when parsing inclination file for string : {0}", data));
                }
                else
                {
                    float inclinationInDeg = Mathf.Rad2Deg * inclination;
                    m_inclinationAngles.Add(inclinationInDeg);
                }
            }
        }

        private void InitializeComputeBuffer()
        {
            argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);

            // Compute how many rays we need
            m_rayCount = 0;
            List<Vector3> directions = new List<Vector3>();
            for (int incr = 0; incr < m_NumberOfCircularSegments; incr++)
            {
                for (int layer = 0; layer < m_inclinationAngles.Count; layer++)
                {
                    int index = layer + incr * m_inclinationAngles.Count;
                    float angle = m_inclinationAngles[layer];
                    float azimuth = incr * m_azimuthIncrementAngle;
                    Vector3 dir = transform.rotation * Quaternion.Euler(-angle, azimuth, 0) * transform.forward;

                    // Skip all fowward directions
                    //if (Vector3.Angle(transform.forward, dir) < 120.0f)
                    //    continue;
                    // if(angle < -13.0f && Mathf.Abs(azimuth - 180) < 20.0f) continue;

                    ++m_rayCount;
                    directions.Add(dir);
                }
            }

            m_particleInfoBuffer = new ComputeBuffer(particleInfoSize, 48);
            m_triangleInfoBuffer = new ComputeBuffer(triangleInfoSize, 36);
            if (m_rayCount > 0)
            {
                m_raycastInfoBuffer = new ComputeBuffer(m_rayCount, 24);
                m_raycastResultInfoBuffer = new ComputeBuffer(m_rayCount, 52);
            }

            // Initialize the raycastInfo compute buffer data
            m_particleInfoDataArray = new ComputeBufferParticleInfoStruct[particleInfoSize];
            m_triangleInfoDataArray = new ComputeBufferTriangleInfoStruct[triangleInfoSize];
            if (m_rayCount > 0)
            {
                m_raycastInfoDataArray = new ComputeBufferRaycastInfoStruct[m_rayCount];
                m_raycastResultInfoDataArray = new ComputeBufferRaycastResultInfoStruct[m_rayCount];
            }

            for (int rayIndex = 0; rayIndex < m_rayCount; ++rayIndex)
            {
                ComputeBufferRaycastInfoStruct raycastInfo = new ComputeBufferRaycastInfoStruct
                {
                    origin = transform.position,
                    direction = directions[rayIndex]
                };
                m_raycastInfoDataArray[rayIndex] = raycastInfo;

                ComputeBufferRaycastResultInfoStruct raycastResultInfo = new ComputeBufferRaycastResultInfoStruct
                {
                    distance = 0.0f,
                    hitPosition = Vector3.zero
                };
                m_raycastResultInfoDataArray[rayIndex] = raycastResultInfo;
            }
            m_raycastInfoBuffer.SetData(m_raycastInfoDataArray);
            m_raycastResultInfoBuffer.SetData(m_raycastResultInfoDataArray);

            // Initialize the particleInfo compute buffer data
            for (int particleIndex = 0; particleIndex < particleInfoSize; ++particleIndex)
            {
                ComputeBufferParticleInfoStruct particleInfo = new ComputeBufferParticleInfoStruct
                {
                    position = Vector3.zero,
                    initialVelocity = Vector3.zero,
                    generationTimeStamp = -1,
                    initialPosition = Vector3.zero,
                    generationCode = 0,
                    particleGenerationIndex = 0
                };
                m_particleInfoDataArray[particleIndex] = particleInfo;
            }
            m_particleInfoBuffer.SetData(m_particleInfoDataArray);

            // Initialize the triangleInfo compute buffer data
            for (int triangleIndex = 0; triangleIndex < m_triangleCount; ++triangleIndex)
            {
                m_triangleInfoDataArray[triangleIndex] = new ComputeBufferTriangleInfoStruct
                {
                    pointA = Vector3.zero,
                    pointB = Vector3.zero,
                    pointC = Vector3.zero
                };
            }
            if (m_triangleInfoBuffer != null)
                m_triangleInfoBuffer.SetData(m_triangleInfoDataArray);
        }

        private void UpdateParticleBuffer(List<ParticleInfo> points)
        {
            //Debug.Log(points.Count);
            if (points.Count >= m_particleInfoDataArray.Length)
            {
                Debug.LogError(
                    "UpdateParticleBuffer : points.Count >= m_particleInfoDataArray.Count : "
                    + "points.Count = " + points.Count.ToString()
                    + "m_particleInfoDataArray.Count = " + m_particleInfoDataArray.Length.ToString()
                    );
            }

            for (int particleIndex = 0; particleIndex < points.Count; ++particleIndex)
            {
                m_particleInfoDataArray[particleIndex] = points[particleIndex];
            }
            m_particleInfoBuffer.SetData(m_particleInfoDataArray);
        }

        private int UpdateTriangleBuffer()
        {
            // Initialize the triangleInfo compute buffer data
            int overallTriangleIndex = 0;
            
            for (int meshIndex = 0; meshIndex < m_MeshColliderList.Count; ++meshIndex)
            {
                Transform targetTransform = m_MeshColliderList[meshIndex].transform;
                Vector3[] vertices = m_MeshColliderList[meshIndex].sharedMesh.vertices;
                int[] triangleIndexArray = m_MeshColliderList[meshIndex].sharedMesh.triangles;
                
                

                for (int index = 0; index < triangleIndexArray.Length; index += 3)
                {
                    int indexA = triangleIndexArray[index + 0];
                    int indexB = triangleIndexArray[index + 1];
                    int indexC = triangleIndexArray[index + 2];

                    Vector3 pointA = targetTransform.TransformPoint(vertices[indexA]);
                    Vector3 pointB = targetTransform.TransformPoint(vertices[indexB]);
                    Vector3 pointC = targetTransform.TransformPoint(vertices[indexC]);

                    m_triangleInfoDataArray[overallTriangleIndex].pointA = pointA;
                    m_triangleInfoDataArray[overallTriangleIndex].pointB = pointB;
                    m_triangleInfoDataArray[overallTriangleIndex].pointC = pointC;

                    ++overallTriangleIndex;
                }
            }
            if (m_triangleInfoBuffer != null)
                m_triangleInfoBuffer.SetData(m_triangleInfoDataArray);

            return overallTriangleIndex;
        }
    }

    // size = size of 3 Vector3 = 4*3 * 3 = 36
    struct ComputeBufferTriangleInfoStruct
    {
        // All these points should be in world space
        public Vector3 pointA;
        public Vector3 pointB;
        public Vector3 pointC;
    }

    // size = size of 2 Vector3 = 4*3 * 2 = 24
    struct ComputeBufferRaycastInfoStruct
    {
        public Vector3 origin;
        public Vector3 direction;
    }

    // size = size of 3 Vector3 + size of 2 float + 2 int = 4*3*3 + 4*2 + 4*2 = 52
    public struct ComputeBufferRaycastResultInfoStruct
    {
        public float distance;
        public Vector3 hitPosition;
        public int hitParticleGenerationTimeStamp;
	    public Vector3 hitParticleInitialVelocity;
        public Vector3 hitParticleInitialPosition;
        public int hitParticleGenerationCode;
        public int hitParticleGenerationIndex;
    }

    public enum PCDFittingConfig : byte
    {
        FitToSinglePCD,
        FitToMultiplePCD
    }
}
