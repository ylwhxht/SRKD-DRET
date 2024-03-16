using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LPCSS;

/**
 *   In order to use this script, remember to disable DetectionReseultParser.cs in the scene.
 *   Also, the sprayModel should have less particle count multiplier ( around 10k ),
 *   and use all particles with random inputVelocityDriverConfig.
**/

public class ManipulationCheck : MonoBehaviour
{
    // [SerializeField]
    // private TextAsset m_DetectionJson = null;
    [SerializeField]
    private SprayModel m_SprayVehicle = null;

    private void Awake()
    {
        // line 0 : particle size
        // line 1 : water film thickness
        // line 2 : perturbation strength
        // line 3 : vehicle velocity

        // Read the config
        StreamReader reader = new StreamReader(GlobalSetting.ManipulationCheckConfig);
        string allText = reader.ReadToEnd();
        string[] lines = allText.Split('\n');
        
        float value;
        {
            if(float.TryParse(lines[0], out value) == false)
            {
                Debug.LogError("float.TryParse Failed for string : " + lines[0]);
            }
            m_SprayVehicle.ParticleSize = value;
        }
        {
            if(float.TryParse(lines[1], out value) == false)
            {
                Debug.LogError("float.TryParse Failed for string : " + lines[1]);
            }
            m_SprayVehicle.WaterFilmThickness = value;
        }
        {
            if(float.TryParse(lines[2], out value) == false)
            {
                Debug.LogError("float.TryParse Failed for string : " + lines[2]);
            }
            m_SprayVehicle.PerturbationStrength = value;
        }
        {
            if(float.TryParse(lines[3], out value) == false)
            {
                Debug.LogError("float.TryParse Failed for string : " + lines[3]);
            }
            m_SprayVehicle.VehicleVelocity = value;
        }
        
        string hash = 
            "PS" + m_SprayVehicle.ParticleSize.ToString() + 
            "_WFT" + m_SprayVehicle.WaterFilmThickness.ToString() + 
            "_PertStr" + m_SprayVehicle.PerturbationStrength.ToString() + 
            "_VV" + m_SprayVehicle.VehicleVelocity.ToString();

        GlobalSetting.OutputPCDPath += hash + "/";
        System.IO.Directory.CreateDirectory(GlobalSetting.OutputPCDPath);
    }

}
