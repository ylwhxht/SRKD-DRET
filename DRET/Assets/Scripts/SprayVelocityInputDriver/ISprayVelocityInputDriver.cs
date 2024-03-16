using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace LPCSS
{
    public interface ISprayVelocityInputDriver
    {
        // Return possible initial state for the particles
        List<ParticleInitialState> GetTreadPickupInitialVelocity(SprayInitialVelocityParameter parameter);
        List<ParticleInitialState> GetBowWaveInitialVelocity(SprayInitialVelocityParameter parameter);
        List<ParticleInitialState> GetSideWaveLeftInitialVelocity(SprayInitialVelocityParameter parameter);
        List<ParticleInitialState> GetSideWaveRightInitialVelocity(SprayInitialVelocityParameter parameter);
    }

    public struct SprayInitialVelocityParameter
    {
        public int generationTimeStamp;
        public int generationCode;
        public float perturbationStrength;
        public float vehicleVelocity;
    }

    public class ParticleInitialState
    {
        public bool isInitialPositionControlled;
        public Vector3 initialPosition;
        public Vector3 initialVelocity;
    }
}