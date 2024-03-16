Shader "Unlit/ColorBar"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
				float3 worldPos : TEXCOORD0;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul (unity_ObjectToWorld, v.vertex);
                // o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                // UNITY_TRANSFER_FOG(o,o.vertex);
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
				// float3 rgb = hsv2rgb(float3( (i.worldPos.x + 1.0f) / 4.0f, 1.0f, 1.0f));
				float3 rgb = hsv2rgb(float3( (i.worldPos.x) / 4.0f, 1.0f, 1.0f));
				return fixed4(rgb, 1.0f);
			}

            // fixed4 frag (v2f i) : SV_Target
            // {
            //     // sample the texture
            //     // fixed4 col = tex2D(_MainTex, i.uv);
            //     // apply fog
            //     // UNITY_APPLY_FOG(i.fogCoord, col);
            //     return col;
            // }
            ENDCG
        }
    }
}
