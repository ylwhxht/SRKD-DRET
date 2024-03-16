﻿/*
	The following license supersedes all notices in the source code.

	Copyright (c) 2018 Kurt Dekker/PLBM Games All rights reserved.

	http://www.twitter.com/kurtdekker

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are
	met:

	Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.

	Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.

	Neither the name of the Kurt Dekker/PLBM Games nor the names of its
	contributors may be used to endorse or promote products derived from
	this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
	IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
	TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
	HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public static class DatasackFormatting
{
	public static string FloatToHexString(float f)
	{
		var bytes = BitConverter.GetBytes(f);
		var i = BitConverter.ToInt32(bytes, 0);
		return "0x" + i.ToString("X8");
	}

	public static float FloatFromHexString(string s)
	{
		if (s == null || s.Length == 0) return 0;
		if (s.StartsWith( "0x"))
		{
			s = s.Substring(2);
			var i = Convert.ToInt32(s, 16);
			var bytes = BitConverter.GetBytes(i);
			return BitConverter.ToSingle(bytes, 0);
		}
		float result = 0.0f;
		if (s.EndsWith( "f")) s = s.Substring( 0, s.Length - 1);
		if (s.EndsWith( "d")) s = s.Substring( 0, s.Length - 1);
		if (float.TryParse( s, out result))
		{
			return result;
		}
		Debug.LogWarning( "DatasackFormatting.FloatFromHexString(): invalid string for float (must be IEEE754-format 0x12345678 or else 1234.5678[f])");
		return 0.0f;
	}

	public static string DoubleToHexString(double f)
	{
		var bytes = BitConverter.GetBytes(f);
		var i = BitConverter.ToInt64(bytes, 0);
		return "0x" + i.ToString("X16");
	}

	public static double DoubleFromHexString(string s)
	{
		if (s == null || s.Length == 0) return 0;
		if (s.StartsWith( "0x"))
		{
			s = s.Substring(2);
			var i = Convert.ToInt64(s, 16);
			var bytes = BitConverter.GetBytes(i);
			return BitConverter.ToDouble(bytes, 0);
		}
		float result = 0.0f;
		if (s.EndsWith( "f")) s = s.Substring( 0, s.Length - 1);
		if (s.EndsWith( "d")) s = s.Substring( 0, s.Length - 1);
		if (float.TryParse( s, out result))
		{
			return result;
		}
		Debug.LogWarning( "DatasackFormatting.DoubleFromHexString(): invalid string for double (must be IEEE754-format 0x12345678012345678 or else 1234.5678[f/d])");
		return 0.0f;
	}
}
