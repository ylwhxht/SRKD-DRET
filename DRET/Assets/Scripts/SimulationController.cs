// using System.Collections;
// using System.Collections.Generic;
// using System.IO;
// using UnityEngine;

// namespace LPCSS
// {
//     /**
//      *  This class reads external config files and update the simulation parameters accordingly.
//      *  It also shuts down the simulation when exceeding a certain time length, meaning we have
//      *  enough synthetic data for optimization.
//      */
//     public class SimulationController : MonoBehaviour
//     {
//         [SerializeField]
//         private SprayModel m_SprayModel = null;

//         private void Awake()
//         {
//             // Read file
//             string path = "../../Desktop/Research/SimConfig/OptmizingWindFieldsConfig.txt";
//             StreamReader reader = new StreamReader(path);

//             /*
//             struct ControlWindFieldData
//             {
//                 public Vector3 position;
//                 public Vector3 windVelocity;
//                 public float alpha;
//             }

//             struct ControlSineWaveData
//             {
//                 public float magnitude;
//                 public float frequency;
//                 public float offset;
//             }
//             */

//             List<ControlWindFieldData> windFields = new List<ControlWindFieldData>();
//             List<ControlSineWaveData> sineWaveData = new List<ControlSineWaveData>();
            
//             string allText = reader.ReadToEnd();
//             string[] lines = allText.Split('\n');

//             ControlWindFieldData currentWindFieldData = new ControlWindFieldData();
//             ControlSineWaveData currentSineWaveData = new ControlSineWaveData();
//             for (int idx = 0; idx < lines.Length; ++idx)
//             {
//                 string data = lines[idx];
//                 // ignore comments
//                 if(data.StartsWith("#")) continue;

//                 // Read sine wave data
//                 if(data.StartsWith("SWD"))
//                 {
//                     string[] sineWaveDataStrings = data.Split(' ');
//                     float magnitude;
//                     if (float.TryParse(sineWaveDataStrings[1], out magnitude) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));
//                     float frequency;
//                     if (float.TryParse(sineWaveDataStrings[2], out frequency) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));
//                     float offset;
//                     if (float.TryParse(sineWaveDataStrings[3], out offset) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));

//                     currentSineWaveData.magnitude = magnitude;
//                     currentSineWaveData.frequency = frequency;
//                     currentSineWaveData.offset = offset;
//                     sineWaveData.Add(currentSineWaveData);
//                 }

//                 if(data.StartsWith("END"))
//                     windFields.Add(currentWindFieldData);
//                 if(data.StartsWith("POS"))
//                 {
//                     string[] positionStrings = data.Split(' ');
//                     float posX;
//                     if (float.TryParse(positionStrings[1], out posX) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));
//                     float posY;
//                     if (float.TryParse(positionStrings[2], out posY) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));
//                     float posZ;
//                     if (float.TryParse(positionStrings[3], out posZ) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));

//                     currentWindFieldData.position.x = posX;
//                     currentWindFieldData.position.y = posY;
//                     currentWindFieldData.position.z = posZ;
//                 }
//                 if(data.StartsWith("VEL"))
//                 {
//                     string[] windVelocityStrings = data.Split(' ');
//                     float velX;
//                     if (float.TryParse(windVelocityStrings[1], out velX) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));
//                     float velY;
//                     if (float.TryParse(windVelocityStrings[2], out velY) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));
//                     float velZ;
//                     if (float.TryParse(windVelocityStrings[3], out velZ) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));

//                     currentWindFieldData.windVelocity.x = velX;
//                     currentWindFieldData.windVelocity.y = velY;
//                     currentWindFieldData.windVelocity.z = velZ;
//                 }
//                 if(data.StartsWith("ALP"))
//                 {
//                     string[] alphaStrings = data.Split(' ');
//                     float alpha;
//                     if (float.TryParse(alphaStrings[1], out alpha) != true)
//                         Debug.LogError(string.Format("Error when parsing ExternalContrlWindFieldsConfig file for string : {0}", data));

//                     currentWindFieldData.alpha = alpha;
//                 }
//             }

//             m_SprayModel.SetExternalContrlWindFields(windFields);
//             m_SprayModel.SetExternalControlSineWavePerturbation(sineWaveData);
//         }

// #if !UNITY_EDITOR
//         private void Update()
//         {
//             if(Time.time >= 2.0f)
//                 Application.Quit();
//         }
// #endif

//     }
// }
