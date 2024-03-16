Shader "Instanced/InstancedShader"
{
	Properties
	{
		_MainColor("Albedo (RGB)", Color) = (1, 1, 1, 1)
	}

	SubShader
	{
		Pass
		{
			Tags {"LightMode" = "ForwardBase"}

			ZTest Always

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fwdbase nolightmap nodirlightmap nodynlightmap novertexlight
			#pragma target 4.5

			#include "UnityCG.cginc"
			#include "UnityLightingCommon.cginc"
			#include "AutoLight.cginc"

			fixed4 _MainColor;

			struct ComputeBufferRaycastResultInfoStruct
			{
				float distance;
				float3 hitPosition;
				int hitParticleGenerationTimeStamp;
				float3 hitParticleInitialVelocity;
				float3 hitParticleInitialPosition;
				int hitParticleGenerationCode;
				int hitParticleGenerationIndex;
			};

		#if SHADER_TARGET >= 45
			StructuredBuffer<ComputeBufferRaycastResultInfoStruct> raycastResultInfoBuffer;
		#endif

			struct v2f
			{
				float4 pos : SV_POSITION;
			};

			v2f vert(appdata_base v, uint instanceID : SV_InstanceID)
			{
			#if SHADER_TARGET >= 45
				ComputeBufferRaycastResultInfoStruct data = raycastResultInfoBuffer[instanceID];
			#else
				ComputeBufferRaycastResultInfoStruct data = 0;
			#endif

				float3 localPosition = v.vertex.xyz * 0.03f;
				float3 worldPosition = data.hitPosition.xyz + localPosition;

				v2f o;
				o.pos = mul(UNITY_MATRIX_VP, float4(worldPosition, 1.0f));
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				return _MainColor;
			}

			ENDCG
		}
	}
}
