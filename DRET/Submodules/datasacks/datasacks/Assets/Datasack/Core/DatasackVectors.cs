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

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public partial class Datasack
{
	public	Vector2	v2Value
	{
		get
		{
			string[] parts = Value.Split( ',');
			if (parts.Length == 2)
			{
				return new Vector2(
					DatasackFormatting.FloatFromHexString( parts[0]),
					DatasackFormatting.FloatFromHexString( parts[1]));
			}
			return Vector2.zero;
		}
		set
		{
			Value = DatasackFormatting.FloatToHexString( value.x) + "," +
				DatasackFormatting.FloatToHexString( value.y);
		}
	}

	public	Vector3	v3Value
	{
		get
		{
			string[] parts = Value.Split( ',');
			if (parts.Length == 3)
			{
				return new Vector3(
					DatasackFormatting.FloatFromHexString( parts[0]),
					DatasackFormatting.FloatFromHexString( parts[1]),
					DatasackFormatting.FloatFromHexString( parts[2]));
			}
			return Vector3.zero;
		}
		set
		{
			Value = DatasackFormatting.FloatToHexString( value.x) + "," +
					DatasackFormatting.FloatToHexString( value.y) + "," +
					DatasackFormatting.FloatToHexString( value.z);
		}
	}

	public	Quaternion qValue
	{
		get
		{
			string[] parts = Value.Split( ',');
			if (parts.Length == 4)
			{
				return new Quaternion(
					DatasackFormatting.FloatFromHexString( parts[0]),
					DatasackFormatting.FloatFromHexString( parts[1]),
					DatasackFormatting.FloatFromHexString( parts[2]),
					DatasackFormatting.FloatFromHexString( parts[3]));
			}
			return Quaternion.identity;
		}
		set
		{
			Value = DatasackFormatting.FloatToHexString( value.x) + "," +
				DatasackFormatting.FloatToHexString( value.y) + "," +
				DatasackFormatting.FloatToHexString( value.z) + "," +
				DatasackFormatting.FloatToHexString( value.w);
		}
	}

	public	Color colorValue
	{
		get
		{
			string[] parts = Value.Split( ',');
			if (parts.Length == 4)
			{
				return new Color(
					DatasackFormatting.FloatFromHexString( parts[0]),
					DatasackFormatting.FloatFromHexString( parts[1]),
					DatasackFormatting.FloatFromHexString( parts[2]),
					DatasackFormatting.FloatFromHexString( parts[3]));
			}
			if (parts.Length == 3)
			{
				return new Color(
					DatasackFormatting.FloatFromHexString( parts[0]),
					DatasackFormatting.FloatFromHexString( parts[1]),
					DatasackFormatting.FloatFromHexString( parts[2]));
			}
			return Color.magenta;
		}
		set
		{
			Value = DatasackFormatting.FloatToHexString( value.r) + "," +
				DatasackFormatting.FloatToHexString( value.g) + "," +
				DatasackFormatting.FloatToHexString( value.b) + "," +
				DatasackFormatting.FloatToHexString( value.a);
		}
	}

	public	Rect rValue
	{
		get
		{
			string[] parts = Value.Split( ',');
			if (parts.Length == 4)
			{
				return new Rect(
					DatasackFormatting.FloatFromHexString( parts[0]),
					DatasackFormatting.FloatFromHexString( parts[1]),
					DatasackFormatting.FloatFromHexString( parts[2]),
					DatasackFormatting.FloatFromHexString( parts[3]));
			}
			return new Rect();
		}
		set
		{
			Value = DatasackFormatting.FloatToHexString( value.x) + "," +
				DatasackFormatting.FloatToHexString( value.y) + "," +
				DatasackFormatting.FloatToHexString( value.width) + "," +
				DatasackFormatting.FloatToHexString( value.height);
		}
	}
}
