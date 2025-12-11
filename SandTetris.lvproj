<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="25008000">
	<Property Name="NI.LV.All.SaveVersion" Type="Str">25.0</Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="bgm.wav" Type="Document" URL="../bgm.wav"/>
		<Item Name="next_piece.ctl" Type="VI" URL="../next_piece.ctl"/>
		<Item Name="sandtetris.vi" Type="VI" URL="../sandtetris.vi"/>
		<Item Name="sandtetris_type.ctl" Type="VI" URL="../sandtetris_type.ctl"/>
		<Item Name="server.py" Type="Document" URL="../server.py"/>
		<Item Name="tetris_ai.py" Type="Document" URL="../tetris_ai.py"/>
		<Item Name="tetris_core.py" Type="Document" URL="../tetris_core.py"/>
		<Item Name="tetris_remote.py" Type="Document" URL="../tetris_remote.py"/>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
