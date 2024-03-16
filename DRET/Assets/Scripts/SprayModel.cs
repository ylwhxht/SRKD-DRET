using DataStructures.ViliWonka.KDTree;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using System.IO;
using Random = UnityEngine.Random;
using System;

// Lidar Point Cloud in Spray and Splash (LPCSS)
namespace LPCSS
{
    //perlin noise to simulate wind field（randomly）
    public class Perlin
    {

        public int repeat;

        public Perlin(int repeat = -1)
        {
            this.repeat = repeat;
            p = new int[512];
            for (int x = 0; x < 512; x++)
            {
                p[x] = permutation[(x) % 256];
            }
        }

        public float OctavePerlin(float t, int octaves, float persistence, float max)
        {
            float total = 0;
            float frequency = 1;
            float amplitude = 1;
            float maxValue = 0;            // Used for normalizing result to 0.0 - 1.0
            for (int i = 0; i < octaves; i++)
            {
                total += perlin(t * frequency) * amplitude;

                maxValue += amplitude;

                amplitude *= persistence;
                frequency *= 2;
            }

            return MathF.Tanh(total / maxValue) * max;//;
        }

        private static int[] permutation = { 151,160,137,91,90,15,					// Hash lookup table as defined by Ken Perlin.  This is a randomly
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,	// arranged array of all numbers from 0-255 inclusive.
		190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };

        public int[] p;                                                    // Doubled permutation to avoid overflow
        //随机打乱数组元素
        
        public float perlin(float t)
        {
            if (repeat > 0)
            {                                   // If we have any repeat on, change the coordinates to their "local" repetitions
                t = t % repeat;
            }

            int ti = (int)t & 255;                              // Calculate the "unit cube" that the point asked will be located in                             // plus 1.  Next we calculate the location (from 0.0 to 1.0) in that cube.
            float tf = t - (int)t;
            float u = fade(tf);


            int a, b;
            a = p[(p[ti & 255] + p[ti & 255] * 2) * 2 & 255] * 2 - 255;
            b = p[(p[ti & 255] * p[ti & 255]) & 255] * 2 - 255;



            float x, y;
            x = grad(a, tf);
            y = grad(b, tf - 1);
            //System.Console.WriteLine(a);
            //System.Console.WriteLine(b);

            return lerp(x, y, u);
        }

        public static float grad(int hash, float t)
        {
            return (hash & 31) * t;
        }

        public static float fade(float t)
        {
            // Fade function as defined by Ken Perlin.  This eases coordinate values
            // so that they will "ease" towards integral values.  This ends up smoothing
            // the final output.

            return t * t * t * (t * (t * 6 - 15) + 10);         // 6t^5 - 15t^4 + 10t^3
        }

        public static float lerp(float a, float b, float x)
        {
            return a + x * (b - a);
        }
    }

    public class SprayModel : MonoBehaviour
    {
        [Header("References")]
        [SerializeField]
        private WindFieldLoader m_WindFieldLoader = null;
        [SerializeField]
        public TireSpraySource[] m_TireSpraySourceList = null;
        [SerializeField]
        private CloseToRealData m_CloseToRealData = null;
        [SerializeField]
        [TextArea]
        private string m_FilterFilePath = GlobalSetting.DefaultFilter;
        private float dely = 0;
        private float delx = 0;
        private float delz = 0;
        private float times = 0;
        private float alltimes = 0;
        Perlin perlinx = new Perlin();
        Perlin perliny = new Perlin();
        Perlin perlinz = new Perlin();
        [Header("Non-optimizaing Parameters")]
        [SerializeField]
        [Range(3000.0f, 50000.0f)]
        private float m_ParticleCountMultiplier = 5000.0f;
        [SerializeField]
        [Tooltip("Recommended options as in reference : 2, 20, 30")]
        [Range(2.0f, 1000.0f)]
        private float m_ParticleDiameter = 20.0f;
        [SerializeField]
        private TimeStepConfig m_TimeStepConfig = TimeStepConfig.Precise;
        [SerializeField]
        private WindFieldConfig m_WindFieldConfig = WindFieldConfig.ExportedData;
        [SerializeField]
        private VisibleParticleConfig m_VisibleParticleConfig = VisibleParticleConfig.All;
        [SerializeField]
        private SprayInitialStatusConfig m_SprayInitialStatusConfig = SprayInitialStatusConfig.Random;

        public VisibleParticleConfig VisibleParticleConfig
        {
            get
            {
                return m_VisibleParticleConfig;
            }
        }

        [Header("Optimizing Parameters")]
        [SerializeField]
        [Range(0.0f, 100.0f)]
        private float m_Velocity = 10.0f;
        [SerializeField]
        [Range(0.0f, 0.0005f)]
        private float m_WaterFilmThickness = 0.0001f; // same range as in Splash and Spray Final Report
        [SerializeField]
        [Tooltip("Now perturbation is applied to all emitters")]
        private float m_PerturbationStrength = 8.0f;

        private ISprayVelocityInputDriver m_sprayVelocityInputDriver = null;

        // Parameters referencing "Simulation of Impact of Water-Film Spray on Visibility"
        // and also Splash and Spray Final Report, p.48
        // also https://www.tiipublications.ie/training/ST14/Review-of-Geometric-Design.pdf
        private const float tireWidth = 0.195f; // meters
        // K = factor that indicates the proportion of the tire width that is not a roove.
        private const float K = 0.77f;

        private const float density = 1000.0f; // density of water, 1000kg/m3, actuall 997 but close enough

        private Dictionary<ParticleSystem, ParticleSystem.Particle[]> m_particleArrayCache
            = new Dictionary<ParticleSystem, ParticleSystem.Particle[]>();
        private Dictionary<ParticleSystem, KDQuery> m_kdtreeQueryCache
            = new Dictionary<ParticleSystem, KDQuery>();

        private List<ParticleInfo> m_allParticleInfo = new List<ParticleInfo>();
        public List<ParticleInfo> AllParticleInfo
        {
            get
            {
                m_allParticleInfo.Clear();
                m_allParticleInfo.Capacity = 50000;
                UpdateAllParticleInfo();
                return m_allParticleInfo;
            }
        }

        public int MaxParticleCount
        {
            get
            {
                int count = 0;
                foreach (var e in m_TireSpraySourceList)
                {
                    count += e.treadPickup.main.maxParticles;
                    count += e.bowWave.main.maxParticles;
                    count += e.sideWaveLeft.main.maxParticles;
                    count += e.sideWaveRight.main.maxParticles;
                }
                return count;
            }
        }

        public float VehicleVelocity
        {
            get{ return m_Velocity; } 
            set{ m_Velocity = value; }
        }

        public float ParticleSize
        {
            get{ return m_ParticleDiameter; } 
            set{ m_ParticleDiameter = value; }
        }

        public float WaterFilmThickness
        {
            get{ return m_WaterFilmThickness; } 
            set{ m_WaterFilmThickness = value; }
        }

        public float PerturbationStrength
        {
            get{ return m_PerturbationStrength; } 
            set{ m_PerturbationStrength = value; }
        }

        public void SetWindFieldLoader(WindFieldLoader loader)
        {
            m_WindFieldLoader = loader;
        }

        public void SetCloseToRealData(CloseToRealData realData)
        {
            m_CloseToRealData = realData;
        }
        
        private float getperlin_noise(Perlin perlin, float time, float aml, float max)
        {
            float noise = perlin.OctavePerlin(time, 10, aml, max);
            return noise;
        }

        private void Awake()
        {
            // Reference : https://docs.unity3d.com/ScriptReference/Random.InitState.html
            Random.InitState(42);


            for (int j = 0;j<3;j++)
            {
                Perlin perlin = new Perlin();
                
                for (int i = 0; i < perlin.p.Length; i++)
                {
                    int index = Random.Range(0, perlin.p.Length);//随机获得0（包括0）到arr.Length（不包括arr.Length）的索引
                                                          //Console.WriteLine("index={0}", index);//查看index的值
                    int temp = perlin.p[i];  //当前元素和随机元素交换位置
                    perlin.p[i] = perlin.p[index];
                    perlin.p[index] = temp;
                }

                if (j == 0) perlinx = perlin;
                else if (j == 1) perliny = perlin;
                else perlinz = perlin;
            }
            // Adjust fixed delta time
            if (m_TimeStepConfig == TimeStepConfig.Precise)
                Time.fixedDeltaTime = 0.001f;
            else if (m_TimeStepConfig == TimeStepConfig.Rough)
                Time.fixedDeltaTime = 0.02f;
            else
                Debug.LogError("TimeStepConfig : Case not considered!");

            // Determine spray velocity input driver
            if (m_SprayInitialStatusConfig == SprayInitialStatusConfig.Random)
            {
                m_sprayVelocityInputDriver = new RandomSprayVelocityInputDriver();
            }
            else if(m_SprayInitialStatusConfig == SprayInitialStatusConfig.Filtered)
            {
                m_sprayVelocityInputDriver = new FilteredSprayVelocityInputDriver(m_FilterFilePath);
            }
            else
            {
                Debug.LogError("SprayInitialStatusConfig : Case not considered!");
            }

            Time.maximumDeltaTime = 0.1f;
            Debug.Log("Time.maximumDeltaTime has been set to 0.1f!");

            foreach (var e in m_TireSpraySourceList)
            {
                e.treadPickup.Pause();
                e.bowWave.Pause();
                e.sideWaveLeft.Pause();
                e.sideWaveRight.Pause();
            }

            Debug.Log("Water Depth = " + m_WaterFilmThickness.ToString(), this);
        }

        private int m_TimeStamp = 0;
        private void FixedUpdate()
        {
            float V = Mathf.Max(0.0f, m_Velocity);

            //init different noise between different vehicles
            if (alltimes == 0)
            {
                Debug.Log("init for a vehicle");
                for (int j = 0; j < 3; j++)
                {
                    Perlin perlin = new Perlin();
                    for (int i = 0; i < perlin.p.Length; i++)
                    {
                        int index = Random.Range(0, perlin.p.Length);//随机获得0（包括0）到arr.Length（不包括arr.Length）的索引
                                                                     //Console.WriteLine("index={0}", index);//查看index的值
                        int temp = perlin.p[i];  //当前元素和随机元素交换位置
                        perlin.p[i] = perlin.p[index];
                        perlin.p[index] = temp;
                    }

                    if (j == 0) perlinx = perlin;
                    else if (j == 1) perliny = perlin;
                    else perlinz = perlin;
                }
            }
            times += Time.fixedDeltaTime;
            alltimes += (float)(Time.fixedDeltaTime*12.3);
            
            if (times>= Time.maximumDeltaTime)
            {
                // x->waymo.z, y->waymo.y, z->waymo.x
                times = 0;
                //delx = (float)0.0;
                delx = getperlin_noise(perlinx, alltimes, (float)1.1, 10);
                //dely = (float)0.0;
                dely = getperlin_noise(perliny, alltimes, (float)1.1, 70); 
                delz = getperlin_noise(perlinz, alltimes, (float)1.1, 70);
                //delz = (float)0.0;
                Debug.Log(alltimes.ToString("0.00")+' '+delx.ToString()+' '+dely.ToString()+' '+delz.ToString());
            }
            List<Thread> allThreads = new List<Thread>();
            Dictionary<ParticleSystem, ParticleArrayData> psToDataMap
                = new Dictionary<ParticleSystem, ParticleArrayData>();
            
            for (int i = 0; i < m_TireSpraySourceList.Length; ++i)
            {

                TireSpraySource e = m_TireSpraySourceList[i];
                int tpGenCode  = i * 10 + 1;
                int bwGenCode  = i * 10 + 2;
                int swlGenCode = i * 10 + 3;
                int swrGenCode = i * 10 + 4;

                if(m_GenCodeToMaxIndexMap.ContainsKey(tpGenCode) == false)
                    m_GenCodeToMaxIndexMap.Add(tpGenCode, 0);
                if(m_GenCodeToMaxIndexMap.ContainsKey(bwGenCode) == false)
                    m_GenCodeToMaxIndexMap.Add(bwGenCode, 0);
                if(m_GenCodeToMaxIndexMap.ContainsKey(swlGenCode) == false)
                    m_GenCodeToMaxIndexMap.Add(swlGenCode, 0);
                if(m_GenCodeToMaxIndexMap.ContainsKey(swrGenCode) == false)
                    m_GenCodeToMaxIndexMap.Add(swrGenCode, 0);
                
                if(m_InitialStateMap.ContainsKey(tpGenCode) == false)
                    m_InitialStateMap.Add(tpGenCode, new Dictionary<int, ParticleInitialState>());
                if(m_InitialStateMap.ContainsKey(bwGenCode) == false)
                    m_InitialStateMap.Add(bwGenCode, new Dictionary<int, ParticleInitialState>());
                if(m_InitialStateMap.ContainsKey(swlGenCode) == false)
                    m_InitialStateMap.Add(swlGenCode, new Dictionary<int, ParticleInitialState>());
                if(m_InitialStateMap.ContainsKey(swrGenCode) == false)
                    m_InitialStateMap.Add(swrGenCode, new Dictionary<int, ParticleInitialState>());

                SprayInitialVelocityParameter tpParameter = new SprayInitialVelocityParameter
                {
                    generationTimeStamp = m_TimeStamp,
                    generationCode = tpGenCode,
                    perturbationStrength = m_PerturbationStrength,
                    vehicleVelocity = m_Velocity
                };

                SprayInitialVelocityParameter bwParameter = new SprayInitialVelocityParameter
                {
                    generationTimeStamp = m_TimeStamp,
                    generationCode = bwGenCode,
                    perturbationStrength = m_PerturbationStrength,
                    vehicleVelocity = m_Velocity
                };

                SprayInitialVelocityParameter swlParameter = new SprayInitialVelocityParameter
                {
                    generationTimeStamp = m_TimeStamp,
                    generationCode = swlGenCode,
                    perturbationStrength = m_PerturbationStrength,
                    vehicleVelocity = m_Velocity
                };

                SprayInitialVelocityParameter swrParameter = new SprayInitialVelocityParameter
                {
                    generationTimeStamp = m_TimeStamp,
                    generationCode = swrGenCode,
                    perturbationStrength = m_PerturbationStrength,
                    vehicleVelocity = m_Velocity
                };
                
                UpdateEmissionRate(e);

                List<ParticleInitialState> tpInitialStates  = m_sprayVelocityInputDriver.GetTreadPickupInitialVelocity(tpParameter);
                List<ParticleInitialState> bwInitialStates  = m_sprayVelocityInputDriver.GetBowWaveInitialVelocity(bwParameter);
                List<ParticleInitialState> swlInitialStates = m_sprayVelocityInputDriver.GetSideWaveLeftInitialVelocity(swlParameter);
                List<ParticleInitialState> swrInitialStates = m_sprayVelocityInputDriver.GetSideWaveRightInitialVelocity(swrParameter);

                // Update : Transform the initPos in the initial states to the world space with
                // regard of the vehicle's transform
                // foreach (var initialState in tpInitialStates)
                Matrix4x4 local2WorldPointMatrix = 
                    Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
                
                for(int idx = 0; idx < tpInitialStates.Count; ++idx)
                {
                    tpInitialStates[idx].initialPosition = local2WorldPointMatrix.MultiplyPoint3x4(tpInitialStates[idx].initialPosition);
                    tpInitialStates[idx].initialVelocity = transform.TransformDirection(tpInitialStates[idx].initialVelocity);
                }
                for(int idx = 0; idx < bwInitialStates.Count; ++idx)
                {
                    bwInitialStates[idx].initialPosition = local2WorldPointMatrix.MultiplyPoint3x4(bwInitialStates[idx].initialPosition);
                    bwInitialStates[idx].initialVelocity = transform.TransformDirection(bwInitialStates[idx].initialVelocity);
                }
                for(int idx = 0; idx < swlInitialStates.Count; ++idx)
                {
                    swlInitialStates[idx].initialPosition = local2WorldPointMatrix.MultiplyPoint3x4(swlInitialStates[idx].initialPosition);
                    swlInitialStates[idx].initialVelocity = transform.TransformDirection(swlInitialStates[idx].initialVelocity);
                }
                for(int idx = 0; idx < swrInitialStates.Count; ++idx)
                {
                    swrInitialStates[idx].initialPosition = local2WorldPointMatrix.MultiplyPoint3x4(swrInitialStates[idx].initialPosition);
                    swrInitialStates[idx].initialVelocity = transform.TransformDirection(swrInitialStates[idx].initialVelocity);
                }

                // Obtain data and update particles systems accordingly
                ParticleArrayData tpData = GetCachedParticleArrayData(e.treadPickup);
                ParticleArrayData bwData = GetCachedParticleArrayData(e.bowWave);
                ParticleArrayData swlData = GetCachedParticleArrayData(e.sideWaveLeft);
                ParticleArrayData swrData = GetCachedParticleArrayData(e.sideWaveRight);

                KDQuery tpQuery = GetCachedkdtreeQuery(e.treadPickup);
                KDQuery bwQuery = GetCachedkdtreeQuery(e.bowWave);
                KDQuery swlQuery = GetCachedkdtreeQuery(e.sideWaveLeft);
                KDQuery swrQuery = GetCachedkdtreeQuery(e.sideWaveRight);

                float fdt = Time.fixedDeltaTime;
                Matrix4x4 local2WorldDirectionMatrix = 
                    Matrix4x4.TRS(Vector3.zero, transform.rotation, Vector3.one);
                Matrix4x4 world2LocalPointMatrix = 
                    // Matrix4x4.TRS(-transform.position, Quaternion.Inverse(transform.rotation), Vector3.one);
                    local2WorldPointMatrix.inverse;

                var thread0 = new Thread(
                    () => UpdateAerodynamicDragAndInitialVelocity(
                        tpData, tpQuery, tpGenCode, tpInitialStates, fdt, m_TimeStamp, 
                        local2WorldDirectionMatrix, world2LocalPointMatrix));
                thread0.Start();
                
                var thread1 = new Thread(
                    () => UpdateAerodynamicDragAndInitialVelocity(
                        bwData, bwQuery, bwGenCode, bwInitialStates, fdt, m_TimeStamp, 
                        local2WorldDirectionMatrix, world2LocalPointMatrix));
                thread1.Start();

                var thread2 = new Thread(
                    () => UpdateAerodynamicDragAndInitialVelocity(
                        swlData, swlQuery, swlGenCode, swlInitialStates, fdt, m_TimeStamp, 
                        local2WorldDirectionMatrix, world2LocalPointMatrix));
                thread2.Start();

                var thread3 = new Thread(
                    () => UpdateAerodynamicDragAndInitialVelocity(
                        swrData, swrQuery, swrGenCode, swrInitialStates, fdt, m_TimeStamp, 
                        local2WorldDirectionMatrix, world2LocalPointMatrix));
                thread3.Start();

                // Done : apply initial velocity to newly generated particles
                // Do not rely on velocityOverLifetime Module

                allThreads.Add(thread0);
                allThreads.Add(thread1);
                allThreads.Add(thread2);
                allThreads.Add(thread3);

                psToDataMap.Add(e.treadPickup, tpData);
                psToDataMap.Add(e.bowWave, bwData);
                psToDataMap.Add(e.sideWaveLeft, swlData);
                psToDataMap.Add(e.sideWaveRight, swrData);
            }

            foreach (var thread in allThreads)
            {
                thread.Join();
            }

            foreach (var pair in psToDataMap)
            {
                pair.Key.SetParticles(pair.Value.particles, pair.Value.numParticlesAlive);
            }

            foreach (var e in m_TireSpraySourceList)
            {
                e.treadPickup.Simulate(Time.fixedDeltaTime, false, false, true);
                e.bowWave.Simulate(Time.fixedDeltaTime, false, false, true);
                e.sideWaveLeft.Simulate(Time.fixedDeltaTime, false, false, true);
                e.sideWaveRight.Simulate(Time.fixedDeltaTime, false, false, true);
            }

            m_TimeStamp++;
        }

        private void UpdateEmissionRate(TireSpraySource e)
        {
            float V = Mathf.Max(0.0f, m_Velocity);

            float MassFlowRate = density * m_WaterFilmThickness * tireWidth;
            float totalEmissionRate = m_ParticleCountMultiplier * MassFlowRate * 10.0f;

            float b = tireWidth;
            float Hfilm = (m_WaterFilmThickness < 0.0001f) ? m_WaterFilmThickness : 0.0001f;
            float Hgroove = (m_WaterFilmThickness < 0.01f) ? m_WaterFilmThickness : 0.01f;

            float MRTP = V * b * (1 - K) * Hgroove * density;
            float MRBW = 0.5f * density * b * V * (m_WaterFilmThickness - K * Hfilm - (1 - K) * Hgroove);
            float MRSW = 0.5f * density * b * V * (m_WaterFilmThickness - K * Hfilm - (1 - K) * Hgroove);
            float sum = MassFlowRate;

            // Update emission rate and (X)initial velocity for each particle systems
            if (e.treadPickup != null)
            {
                var tpEmission = e.treadPickup.emission;
                tpEmission.rateOverTime =
                    Mathf.Max(0.0f, (MRTP / sum) * totalEmissionRate);
            }
            if (e.bowWave != null)
            {
                var bwEmission = e.bowWave.emission;
                bwEmission.rateOverTime =
                    Mathf.Max(0.0f, (MRBW / sum) * totalEmissionRate);
            }
            if (e.sideWaveLeft != null)
            {
                var swEmission = e.sideWaveLeft.emission;
                swEmission.rateOverTime =
                    0.5f *
                    Mathf.Max(0.0f, (MRSW / sum) * totalEmissionRate);
            }
            if (e.sideWaveRight != null)
            {
                var swEmission = e.sideWaveRight.emission;
                swEmission.rateOverTime =
                    0.5f *
                    Mathf.Max(0.0f, (MRSW / sum) * totalEmissionRate);
            }
        }

        private ParticleArrayData GetCachedParticleArrayData(ParticleSystem particleSystem)
        {
            if (m_particleArrayCache.ContainsKey(particleSystem) == false)
            {
                m_particleArrayCache[particleSystem] = new ParticleSystem.Particle[particleSystem.main.maxParticles];
            }
            int numParticlesAlive = particleSystem.GetParticles(m_particleArrayCache[particleSystem]);
            ParticleSystem.Particle[] particles = m_particleArrayCache[particleSystem];

            return new ParticleArrayData { particles = particles, numParticlesAlive = numParticlesAlive };
        }

        private KDQuery GetCachedkdtreeQuery(ParticleSystem particleSystem)
        {
            if (m_kdtreeQueryCache.ContainsKey(particleSystem) == false)
            {
                m_kdtreeQueryCache[particleSystem] = new KDQuery();
            }

            return m_kdtreeQueryCache[particleSystem];
        }

        // genCode to max index
        private volatile Dictionary<int, int> m_GenCodeToMaxIndexMap = new Dictionary<int, int>();
        // m_InitialStateMap [genCode][particleIndex] = initialStateOfParticle
        private volatile Dictionary<int, Dictionary<int, ParticleInitialState>> m_InitialStateMap = new Dictionary<int, Dictionary<int, ParticleInitialState>>();

        private void UpdateAerodynamicDragAndInitialVelocity(
            ParticleArrayData particleArrayData, KDQuery kdtreeQuery,
            int generationCode,
            List<ParticleInitialState> initialStates,
            float fixedDeltaTime, int passedGameTimeStamp,
            Matrix4x4 local2WorldDirectionMatrix,
            Matrix4x4 world2LocalPointMatrix
            )
        {
            int numParticlesAlive = particleArrayData.numParticlesAlive;
            ParticleSystem.Particle[] particles = particleArrayData.particles;

            float particleDiameter = m_ParticleDiameter * 1e-6f; // µm to m
            // assumed to be perfect sphere
            float particleVolume = (4.0f / 3.0f) * Mathf.PI * particleDiameter * particleDiameter * particleDiameter;
            float particleMass = particleVolume * density;

            // compute aerodynamic drag base on air velocity (which is identical to vehicle velocity)
            // particle velocity (use "totalVelocity" to also account noise and velocity over lifetime module)
            // drag coefficient (here we represent with a constant)
            const float dragCoef = 0.2f;
            const float airDensity = 1.225f; // air density is about 1.225 kg/m³
            // Ad = pi * r * r = 0.25 * pi * d * d
            float Ad = 0.25f * Mathf.PI * particleDiameter * particleDiameter;

            Vector3 airVelocity = new Vector3(0, 0, -m_Velocity);
            List<int> queryResult = new List<int>();

            int initialStateIdx = 0;
            bool isCovered = false;
            for (int i = 0; i < numParticlesAlive; ++i)
            {
                // Apply initial velocity to newly generated particles, which are those have passed life time
                // lower than a single step of the particle system, which is fixedDeltaTime
                if (particles[i].startLifetime - particles[i].remainingLifetime <= fixedDeltaTime)
                {
                    // initialize the particle
                    particles[i].velocity = initialStates[initialStateIdx].initialVelocity;

                    if(initialStates[initialStateIdx].isInitialPositionControlled == true)
                    {
                        particles[i].position = initialStates[initialStateIdx].initialPosition;
                        if(isCovered == true)
                            particles[i].remainingLifetime = -1.0f;
                    }

                    initialStateIdx++;
                    if(initialStateIdx >= initialStates.Count)
                    {
                        initialStateIdx -= initialStates.Count;
                        isCovered = true;
                    }
                    // end of - initialize the particle

                    // assign and increment the max index of the corresponding genCode
                    int particleIndex = m_GenCodeToMaxIndexMap[generationCode];
                    m_GenCodeToMaxIndexMap[generationCode] += 1;
                    particles[i].rotation3D = new Vector3((float)particleIndex, passedGameTimeStamp, 0.0f);

                    // Record the initial state
                    m_InitialStateMap[generationCode].Add(particleIndex, new ParticleInitialState
                    {
                        isInitialPositionControlled = true, // this does not matter
                        initialPosition = world2LocalPointMatrix.MultiplyPoint3x4(particles[i].position),
                        initialVelocity = particles[i].velocity
                    });
                }

                // Determine the appropriate wind field source according to config, then apply it to the particles
                if (m_WindFieldConfig == WindFieldConfig.ExportedData)
                    airVelocity = m_WindFieldLoader.GetWindVelocity(
                        kdtreeQuery, queryResult, world2LocalPointMatrix.MultiplyPoint3x4(particles[i].position));
                else if (m_WindFieldConfig == WindFieldConfig.None)
                    airVelocity = Vector3.zero;

                // airVelocity is in local space, transform to global space direction
                airVelocity = local2WorldDirectionMatrix.MultiplyVector(airVelocity);
                
                Vector3 deltaVel = airVelocity - particles[i].totalVelocity;
                Vector3 force = 0.5f * dragCoef * airDensity * Ad * deltaVel * deltaVel.magnitude;
                Vector3 acc = force / particleMass;

                particles[i].velocity += acc * fixedDeltaTime;



                Vector3 change = new Vector3(dely, delx, delz) * fixedDeltaTime;
                particles[i].velocity += change;
                // Remove particles that already touches the ground, which is at (0, -1, 0)
                if(particles[i].position.y <= -0.98f && particles[i].totalVelocity.y <= 0.0f)
                {
                    // Kill the particle by setting its remaining life time to a negative value
                    particles[i].remainingLifetime = -1.0f;
                }
            }
        }

        private void UpdateAllParticleInfo()
        {
            // The generation code is based on the index in the tire spray source list, and mechanism

            for(int i = 0; i < m_TireSpraySourceList.Length; ++i)
            {
                
                UpdateParticleInfo(m_allParticleInfo, m_TireSpraySourceList[i].treadPickup,   i*10 + 1);
                UpdateParticleInfo(m_allParticleInfo, m_TireSpraySourceList[i].bowWave,       i*10 + 2);
                UpdateParticleInfo(m_allParticleInfo, m_TireSpraySourceList[i].sideWaveLeft,  i*10 + 3);
                UpdateParticleInfo(m_allParticleInfo, m_TireSpraySourceList[i].sideWaveRight, i*10 + 4);
            }
        }

        private void UpdateParticleInfo(
            List<ParticleInfo> allParticleInfo,
            ParticleSystem particleSystem,
            int generationCode)
        {
            var data = GetCachedParticleArrayData(particleSystem);
            
            List<Vector4> particleCustomData1 = new List<Vector4>();
            particleSystem.GetCustomParticleData(particleCustomData1, ParticleSystemCustomData.Custom1);
            List<Vector4> particleCustomData2 = new List<Vector4>();
            particleSystem.GetCustomParticleData(particleCustomData2, ParticleSystemCustomData.Custom2);
            
            KDQuery query = new KDQuery();
            
            List<int> queryResults = new List<int>();
            for (int i = 0; i < data.numParticlesAlive; ++i)
            {
                
                // if we only want the particles that are close to real data, perform the test
                if (m_VisibleParticleConfig == VisibleParticleConfig.CloseToRealData && 
                   m_CloseToRealData.IsCloseToRealPoint(
                       data.particles[i].position, query, queryResults) == false)
                {

                    continue;
                }
                
                ParticleInfo info;
                info.position = data.particles[i].position;
                info.generationTimeStamp = (int)data.particles[i].rotation3D.y;
                // Obtain the particle index from the rotation3D field,
                // and look up in the maintained map
                int particleIndex = (int)data.particles[i].rotation3D.x;

                ParticleInitialState state = m_InitialStateMap[generationCode][particleIndex];

                info.initialVelocity = state.initialVelocity;
                info.initialPosition = state.initialPosition;
                info.generationCode = generationCode;
                info.particleGenerationIndex = particleIndex;
                //System.Console.WriteLine(info.generationCode);
                allParticleInfo.Add(info);
            }
        }

        public string GetParameterInfoString()
        {
            string infoString = "";
            infoString += string.Format("m_ParticleCountMultiplier = {0}, \n", m_ParticleCountMultiplier);
            infoString += string.Format("# m_ParticleDiameter = {0} (micrometers), \n", m_ParticleDiameter);
            infoString += string.Format("# m_TimeStepConfig = {0}, Time.fixedDeltaTime = {1}, \n", m_TimeStepConfig, Time.fixedDeltaTime);
            infoString += string.Format("# m_WindFieldConfig = {0}, \n", m_WindFieldConfig);
            infoString += string.Format("# m_Velocity = {0} (m/s), \n", m_Velocity);
            infoString += string.Format("# m_WaterFilmThickness = {0} (meters), \n", m_WaterFilmThickness);
            infoString += string.Format("# m_PerturbationStrength = {0}, \n", m_PerturbationStrength);
            infoString += string.Format("# Time.time = {0}, \n", Time.time);
            infoString += string.Format("# m_VisibleParticleConfig = {0}, \n", m_VisibleParticleConfig);
            infoString += string.Format("# m_SprayInitialStatusConfig = {0}, \n", m_SprayInitialStatusConfig);
            infoString += string.Format("# CurrentFittingPCDFilePath = {0},", m_CloseToRealData.GetCurrentFittingPCDFilePath());

            return infoString;
        }
    }

    [System.Serializable]
    public struct TireSpraySource
    {
        public ParticleSystem treadPickup;
        public ParticleSystem bowWave;
        public ParticleSystem sideWaveLeft;
        public ParticleSystem sideWaveRight;
    }

    [System.Serializable]
    public enum TimeStepConfig : byte
    {
        Precise, // Time.fixedDeltaTime = 0.001f;
        Rough
    }

    [System.Serializable]
    public enum WindFieldConfig : byte
    {
        None, // (0, 0, 0)
        Const, // (0, 0, -V)
        ExportedData // Exported data from SimFlow
    }

    [System.Serializable]
    public enum SprayInitialStatusConfig : byte
    {
        Random,
        Filtered
    }

    [System.Serializable]
    public enum VisibleParticleConfig : byte
    {
        All,
        CloseToRealData
    }

    public struct ParticleArrayData
    {
        public ParticleSystem.Particle[] particles;
        public int numParticlesAlive;
    }

    public struct ParticleInfo
    {
        public Vector3 position;
        public Vector3 initialVelocity;
        public int generationTimeStamp;
        public Vector3 initialPosition;
        // this code encodes which tire and mechanism the particle is generated by
        public int generationCode;
        public int particleGenerationIndex;
    }
}
