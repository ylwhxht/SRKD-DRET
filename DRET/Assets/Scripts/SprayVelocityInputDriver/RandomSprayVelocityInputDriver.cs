using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace LPCSS
{
    public class RandomSprayVelocityInputDriver : ISprayVelocityInputDriver
    {
        public List<ParticleInitialState> GetTreadPickupInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            // Update : Add perturbation based on Sin-wave to emulate the waving effect found in real-data (Waymo)
            // Updtae : Change from 10.0f degree to 30.0f degree, Reference paper : 
            // Simulation method for vehicle tire water spraying behavior [Internationales Stuttgarter Symposium, 2018]
                
            float V = parameter.vehicleVelocity;

            float tpAngle = Random.Range(10.0f, 30.0f);
            Vector3 tpInitialVelocity = new Vector3(
                0.0f,
                V * Mathf.Sin(Mathf.Deg2Rad * tpAngle),
                -1.0f * V * Mathf.Cos(Mathf.Deg2Rad * tpAngle)
            );

            // Update : Use actual random range instead of sin wave based perturbation
            // Reference : https://docs.unity3d.com/ScriptReference/Random.Range.html
            // The Random.Range distribution is uniform.
            tpInitialVelocity.x += parameter.perturbationStrength * Random.Range(-1.0f, 1.0f);

            tpInitialVelocity.Normalize();
            tpInitialVelocity *= V;

            ParticleInitialState state = new ParticleInitialState
            {
                isInitialPositionControlled = false,
                initialPosition = Vector3.zero,
                initialVelocity = tpInitialVelocity
            };

            return new List<ParticleInitialState>{state};
        }

        public List<ParticleInitialState> GetBowWaveInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            float V = parameter.vehicleVelocity;

            Vector3 bwInitialVelocity = new Vector3(
                0.0f,
                0.2f * V * Mathf.Sin(Mathf.Deg2Rad * 35.0f),
                0.2f * V * Mathf.Cos(Mathf.Deg2Rad * 35.0f)
            );

            bwInitialVelocity.x += parameter.perturbationStrength * Random.Range(-1.0f, 1.0f);

            ParticleInitialState state = new ParticleInitialState
            {
                isInitialPositionControlled = false,
                initialPosition = Vector3.zero,
                initialVelocity = bwInitialVelocity
            };

            return new List<ParticleInitialState>{state};
        }

        public List<ParticleInitialState> GetSideWaveLeftInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            float V = parameter.vehicleVelocity;

            float angleN = Random.Range(15.0f, 20.0f);
            float angleL = Random.Range(10.0f, 15.0f);
            Vector3 swlInitialVelocity = new Vector3(
                -1.0f * V * Mathf.Tan(Mathf.Deg2Rad * angleN),
                V * Mathf.Tan(Mathf.Deg2Rad * angleL),
                -1.0f * V
            );

            swlInitialVelocity.x += parameter.perturbationStrength * Random.Range(-1.0f, 1.0f);

            ParticleInitialState state = new ParticleInitialState
            {
                isInitialPositionControlled = false,
                initialPosition = Vector3.zero,
                initialVelocity = swlInitialVelocity
            };

            return new List<ParticleInitialState>{state};
        }

        public List<ParticleInitialState> GetSideWaveRightInitialVelocity(SprayInitialVelocityParameter parameter)
        {
            float V = parameter.vehicleVelocity;

            float angleN = Random.Range(15.0f, 20.0f);
            float angleL = Random.Range(10.0f, 15.0f);
            Vector3 swrInitialVelocity = new Vector3(
                V * Mathf.Tan(Mathf.Deg2Rad * angleN),
                V * Mathf.Tan(Mathf.Deg2Rad * angleL),
                -1.0f * V
            );

            swrInitialVelocity.x += parameter.perturbationStrength * Random.Range(-1.0f, 1.0f);

            ParticleInitialState state = new ParticleInitialState
            {
                isInitialPositionControlled = false,
                initialPosition = Vector3.zero,
                initialVelocity = swrInitialVelocity
            };

            return new List<ParticleInitialState>{state};
        }

    }

}