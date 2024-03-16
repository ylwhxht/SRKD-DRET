Shader "Instanced/InstancedShader_OnlyPos"
{
	Properties
	{
		_MainColor("Albedo (RGB)", Color) = (1, 1, 1, 1)
		_PointSize("Point Size", float) = 0.1
	}

	SubShader
	{
		Pass
		{
			Tags {"LightMode" = "ForwardBase"}

			//ZTest Always

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fwdbase nolightmap nodirlightmap nodynlightmap novertexlight
			#pragma target 4.5

			#include "UnityCG.cginc"
			#include "UnityLightingCommon.cginc"
			#include "AutoLight.cginc"

			fixed4 _MainColor;
			float _PointSize;
			float4 _Offset;

		#if SHADER_TARGET >= 45
			StructuredBuffer<float3> positionInfoBuffer;
		#endif

			struct v2f
			{
				float4 pos : SV_POSITION;
				float3 worldPos : TEXCOORD0;
			};

			v2f vert(appdata_base v, uint instanceID : SV_InstanceID)
			{
			#if SHADER_TARGET >= 45
				float3 data = positionInfoBuffer[instanceID];
			#else
				float3 data = 0;
			#endif

				float3 localPosition = v.vertex.xyz * _PointSize;
				float3 worldPosition = data + localPosition;

				v2f o;
				o.pos = mul(UNITY_MATRIX_VP, float4(worldPosition + _Offset.xyz, 1.0f));
				o.worldPos = worldPosition;
				return o;
			}

			float3 hsv2rgb(float3 c)
			{
				float3 rgb = clamp(abs(fmod(c.x * 6.0 + float3(0.0, 4.0, 2.0), 6) - 3.0) - 1.0, 0, 1);
				rgb = rgb * rgb * (3.0 - 2.0 * rgb);
				return c.z * lerp(float3(1, 1, 1), rgb, c.y);
			}

			fixed4 frag(v2f i) : SV_Target
			{
				//return _MainColor;
				float3 rgb = hsv2rgb(float3( (i.worldPos.y + 1.0f) / 4.0f, 1.0f, 1.0f));
				return fixed4(rgb, 1.0f);
			}

			ENDCG
		}
	}
}
