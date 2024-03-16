using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
// Binray formatter
using System.Runtime.Serialization.Formatters.Binary;
using LPCSS.ExportUtil;

namespace LPCSS
{
    public class FilteredSprayVelocityInputDriver : ISprayVelocityInputDriver
    {
        // Use generation time and code to look up the filter table, ignore perturbation and vehicle velocity

        private string m_filterFilePath = null;
            
        // initialize this class upon first requested
        private bool m_isInitialized = false;
        private Dictionary<LookUpKey, List<ParticleInitialState>> m_LookUpTable = new Dictionary<LookUpKey, List<ParticleInitialState>>();

        public FilteredSprayVelocityInputDriver(string filterFilePath)
        {
            m_filterFilePath = filterFilePath;
        }

        private void Init()
        {
            m_isInitialized = true;

            StreamReader reader = new StreamReader(m_filterFilePath);
            string allText = reader.ReadToEnd();
            string[] lines = allText.Split('\n');
            
            for (int idx = 0; idx < lines.Length; ++idx)
            {
                string data = lines[idx];
                // ignore comments
                if(data.StartsWith("#") || string.IsNullOrWhiteSpace(data)) continue;

                // Parse key-value pair : "time code idx initVelX initVelY initVelZ initPosX initPosY initPosZ"
                // Ex. : 1139 21 14294 0x40CE061E 0x4131858D 0xC1D011A0 0xBF685348 0xBF770B82 0xBFE51312

                LookUpKey key;
                string[] keyValueStrings = data.Split(' ');
                int genTimeStamp;
                if (int.TryParse(keyValueStrings[0], out genTimeStamp) != true)
                    Debug.LogError(string.Format("Error when parsing FilteredSprayVelocityInputDriver file for string : {0}", data));
                int genCode;
                if (int.TryParse(keyValueStrings[1], out genCode) != true)
                    Debug.LogError(string.Format("Error when parsing FilteredSprayVelocityInputDriver file for string : {0}", data));

                // normalize particle generation time with respect to Time.fixedDeltaTime
                key.generationTimeStamp = genTimeStamp;
                key.generationCode = genCode;
	string tmp = "";
	for (int i = 0; i < keyValueStrings[8].Length-1; i++)
    		tmp+=keyValueStrings[8][i];
	keyValueStrings[8] = tmp;
                float initPosZ = DatasackFormatting.FloatFromHexString(keyValueStrings[8]);
                float initVelX = DatasackFormatting.FloatFromHexString(keyValueStrings[3]);
                float initVelY = DatasackFormatting.FloatFromHexString(keyValueStrings[4]);
                float initVelZ = DatasackFormatting.FloatFromHexString(keyValueStrings[5]);
                float initPosX = DatasackFormatting.FloatFromHexString(keyValueStrings[6]);
                float initPosY = DatasackFormatting.FloatFromHexString(keyValueStrings[7]);
                

                ParticleInitialState valueState = new ParticleInitialState();
                valueState.isInitialPositionControlled = true;
                valueState.initialPosition = new Vector3(initPosX, initPosY, initPosZ);
                valueState.initialVelocity = new Vector3(initVelX, initVelY, initVelZ);

                if(m_LookUpTable.ContainsKey(key) == false)
                {
                    m_LookUpTable.Add(key, new List<ParticleInitialState>());
                }
                m_LookUpTable[key].Add(valueState);
            }
        }

        private List<ParticleInitialState> LookUpByParameter(SprayInitialVelocityParameter parameter)
        {
            if(m_isInitialized == false)
            {
                Init();
            }

            LookUpKey key;
            key.generationTimeStamp = parameter.generationTimeStamp;
            key.generationCode = parameter.generationCode;
            if(m_LookUpTable.ContainsKey(key) == false)
            {
                return new List<ParticleInitialState>{ new ParticleInitialState
                    {
                        isInitialPositionControlled = true,
                        initialPosition = Vector3.down * 300.0f,
                        initialVelocity = Vector3.down * 300.0f
                    }};
            }
            else
            {
                return m_LookUpTable[key];
            }
        }

        public List<ParticleInitialState> GetTreadPickupInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            return LookUpByParameter(parameter);
        }

        public List<ParticleInitialState> GetBowWaveInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            return LookUpByParameter(parameter);
        }

        public List<ParticleInitialState> GetSideWaveLeftInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            return LookUpByParameter(parameter);
        }

        public List<ParticleInitialState> GetSideWaveRightInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            return LookUpByParameter(parameter);
        }

        struct LookUpKey
        {
            public int generationTimeStamp;
            public int generationCode;
        }
    }
}