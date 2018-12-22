import bpy
import bmesh
from math import degrees
from mathutils import Vector, kdtree
import mathutils
from bpy_extras import view3d_utils
from bpy.props import StringProperty
import blf
import bgl
from math import sqrt
import numpy as np
from mathutils.geometry import intersect_line_plane
import os
from bpy.props import IntProperty, FloatProperty

from bpy.types import Operator
from math import sin, cos, pi, radians
from time import perf_counter
from bpy.props import (
	BoolProperty,
	FloatProperty,
	EnumProperty, )

bl_info = {
	"name": "Destructive Extrude :)",
	"location": "View3D > Add > Mesh > Destructive Extrude,",
	"description": "Extrude how SketchUp.",
	"author": "Vladislav Kindushov",
	"version": (1, 0, 0),
	"blender": (2, 80, 0),
	"category": "Mesh",
}


def CursorPosition(self,context, is_Set = False):
	if is_Set and self.CursorLocation != 'NONE':
		context.scene.cursor_location = self.CursorLocation
	else:
		self.CursorLocation = context.scene.cursor_location

def CreateNewObject(self,context):
	# ________Duplicate Object________#
	bpy.ops.mesh.duplicate_move()
	bpy.ops.mesh.separate(type='SELECTED')
	bpy.ops.object.mode_set(mode='OBJECT')
	self.ExtrudeObject = context.selected_objects[-1]
	# ________Clear Modifiers________#
	while len(self.ExtrudeObject.modifiers) != 0:
		self.ExtrudeObject.modifiers.remove(self.ExtrudeObject.modifiers[0])

def GetVisualSetings(self,context, isSet=False):
	if isSet:
		context.active_object.show_all_edges = self.ShowAllEdges
		context.active_object.show_wire = self.ShowAllEdges
	else:
		self.ShowAllEdges = context.active_object.show_all_edges
		self.ShowAllEdges = context.active_object.show_wire

def SetVisualSetings(self,context):
	self.MainObject.show_all_edges = True
	self.MainObject.show_wire = True

	self.ExtrudeObject.display_type = 'WIRE'

def GetVisualModifiers(self,context, isSet=False):
	if isSet:
		for i in self.MainObject.modifiers:
			if i.name in self.VisibilityModifiers:
				i.show_viewport = True
	else:
		for i in self.MainObject.modifiers:
			if i.show_viewport:
				self.VisibilityModifiers.append(i.name)
				i.show_viewport = False

def CreateModifier(self,context):
	#________Set Boolean________#
	context.view_layer.objects.active = self.MainObject
	self.bool = context.object.modifiers.new('DestructiveBoolean', 'BOOLEAN')
	bpy.context.object.modifiers["DestructiveBoolean"].operation = 'DIFFERENCE'
	bpy.context.object.modifiers["DestructiveBoolean"].object = self.ExtrudeObject
	bpy.context.object.modifiers["DestructiveBoolean"].show_viewport = True
	# ________Set Solidify________#
	context.view_layer.objects.active = self.ExtrudeObject
	context.object.modifiers.new('DestructiveSolidify', 'SOLIDIFY')
	context.object.modifiers['DestructiveSolidify'].use_even_offset = True
	context.object.modifiers['DestructiveSolidify'].offset = -0.99959

def GetMouseLocation(self, event, context):
	region = bpy.context.region
	rv3d = bpy.context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
	ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
	V_a = ray_origin_mouse + view_vector_mouse
	V_b = rv3d.view_rotation @ Vector((0.0,0.0,-1.0))
	pointLoc = intersect_line_plane(ray_origin_mouse,V_a,
									context.object.location,V_b )

	loc = (self.GeneralNormal @ pointLoc) * -1
	return loc

def SetSolidifyValue(self,context, value):
	self.ExtrudeObject.modifiers[-1].thickness = value

def CalculateNormal(self,context):
	for i in self.ExtrudeObject.data.polygons:
		self.GeneralNormal += i.normal.copy()

def TransformObject(self, context):
	selObj = context.selected_objects
	bpy.ops.object.select_all(action='DESELECT')
	self.ExtrudeObject.select_set(True)
	context.view_layer.objects.active = self.ExtrudeObject
	bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
	bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
	bpy.ops.view3d.snap_cursor_to_selected()

	bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'
	#bpy.ops.transform.resize(value=(1.000001, 1.000001, 1.000001))
	self.ExtrudeObject.scale = Vector((1.001, 1.001, 1.001))

	for i in selObj:
		i.select_set(True)
	context.view_layer.objects.active = self.MainObject

def GetFaceNormal(self,context):
	for i in self.ExtrudeObject.data.polygons:
		self.FaceNormal.append(i.normal.copy())

def GetMainVertsIndex(self, context):
	for i in self.ExtrudeObject.data.vertices:
		self.MainVertsIndex.append(i.index)

def SetForAxis(self, context):
	GetMainVertsIndex(self, context)
	context.view_layer.objects.active = self.ExtrudeObject
	for i in range(0, len(self.MainVertsIndex) - 1):
		self.StartVertsPos.append(self.ExtrudeObject.data.vertices[i].co.copy())
	index = []
	for f in self.ExtrudeObject.data.polygons:
		normal = f.normal
		for v in f.vertices:
			if v not in index:
				self.ExtrudeObject.data.vertices[v].co = normal * 0.02 + self.ExtrudeObject.data.vertices[v].co.copy()
				index.append(v)


	self.ExtrudeObject.modifiers[0].thickness = 0.00
	self.ExtrudeObject.modifiers[0].offset = 0
	bpy.ops.object.modifier_apply(apply_as = 'DATA', modifier = self.ExtrudeObject.modifiers[0].name)
	for i in range(len(self.MainVertsIndex) - 1, len(self.ExtrudeObject.data.vertices)):
		self.StartVertsPos.append(self.ExtrudeObject.data.vertices[i].co.copy())
		context.view_layer.objects.active = self.MainObject

def ReturnStartPosition(self, context):
	for i in range(len(self.MainVertsIndex) - 1, len(self.ExtrudeObject.data.vertices)):
		self.ExtrudeObject.data.vertices[i].co = self.StartVertsPos[i]

def AxisMove(self, context, value):
	if self.AxisMove == 'X':
		axis = Vector((-1.0, 0.0, 0.0))
	elif self.AxisMove == 'Y':
		axis = Vector((0.0, -1.0, 0.0))
	elif self.AxisMove == 'Z':
		axis = Vector((0.0, 0.0, -1.0))

	for i in range(len(self.MainVertsIndex), len(self.ExtrudeObject.data.vertices)):
		vertPos = ((axis * value) + self.StartVertsPos[i])
		self.ExtrudeObject.data.vertices[i].co = vertPos

def Cansel(self, context):
	bpy.data.objects.remove(self.ExtrudeObject)
	context.view_layer.objects.active = self.MainObject
	bpy.ops.object.modifier_remove(apply_as='DATA', modifier='DestructiveBoolean')
	GetVisualSetings(self, context, True)
	GetVisualModifiers(self, context, True)
	bpy.ops.object.mode_set(mode='EDIT')

def Finish(self, context, BevelUpdate=False):
	if self.NormalMove:
		context.view_layer.objects.active = self.ExtrudeObject
		GetMainVertsIndex(self, context)
		bpy.ops.object.modifier_apply(apply_as='DATA', modifier=self.ExtrudeObject.modifiers[0].name)
		bpy.ops.object.mode_set(mode='EDIT')
		bpy.ops.object.mode_set(mode='OBJECT')

	context.view_layer.objects.active = self.MainObject
	bpy.ops.object.modifier_apply(apply_as='DATA', modifier='DestructiveBoolean')
	bpy.context.scene.update()
	context.active_object.data.update()
	context.active_object.data.update(calc_edges=False)
	context.active_object.update_tag(refresh={'OBJECT', 'DATA', 'TIME'})
	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.object.mode_set(mode='OBJECT')

	for f in self.ExtrudeObject.data.polygons:
		lose = False
		for v in f.vertices:
			if v in self.MainVertsIndex:
				lose = True
				break
		if lose:
			continue
		else:
			center = self.MainObject.matrix_world.inverted() @ f.center
			normal = self.MainObject.matrix_world.inverted() @ f.normal
			result, location, normal, index = self.MainObject.ray_cast(center,normal)
			if result:
				self.MainObject.data.polygons[index].select = True
	bpy.data.objects.remove(self.ExtrudeObject)
	GetVisualSetings(self, context, True)
	GetVisualModifiers(self, context, True)
	bpy.ops.object.mode_set(mode='EDIT')






class DestuctiveExtrude(bpy.types.Operator):
	bl_idname = "mesh.destuctive_extrude"
	bl_label = "Destructive Extrude"
	bl_options = {"REGISTER", "UNDO", "GRAB_CURSOR", "BLOCKING"}

	@classmethod
	def poll(cls, context):
		return (context.mode == "EDIT_MESH")

	def modal(self, context, event):
		if event.type == 'MOUSEMOVE':
			value = GetMouseLocation(self, event, context) - self.StartMouseLocation
			if self.NormalMove:
				SetSolidifyValue(self,context, value)
			else:
				AxisMove(self, context, value)


		if event.type == 'X':
			if self.NormalMove:
				SetForAxis(self, context)
				self.NormalMove = False
			ReturnStartPosition(self, context)
			self.AxisMove = 'X'

		if event.type == 'Y':
			if self.NormalMove:
				SetForAxis(self, context)
				self.NormalMove = False
			ReturnStartPosition(self, context)
			self.AxisMove = 'Y'

		if event.type == 'Z':
			if self.NormalMove:
				SetForAxis(self, context)
				self.NormalMove = False
			ReturnStartPosition(self, context)
			self.AxisMove = 'Z'

		if event.type == 'LEFTMOUSE':
			Finish(self, context, BevelUpdate=False)
			return {'FINISHED'}

		if event.type in {'RIGHTMOUSE', 'ESC'}:
			Cansel(self, context)
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self,context, event):
		if context.space_data.type == 'VIEW_3D':
			self.MainVertsIndex = []
			self.AxisMove = 'Z'
			self.StartVertsPos = []
			self.NormalMove = True
			self.GeneralNormal = Vector((0.0,0.0,0.0))
			self.FaceNormal = []
			self.ShowAllEdges = None
			self.ShowWire = None
			self.CursorLocation = None
			self.VisibilityModifiers=[]
			self.MainObject = context.active_object
			self.ExtrudeObject = None
			self.SaveSelectFaceForCansel = None

			#________For Axis Move________#


			GetVisualModifiers(self,context)
			GetVisualSetings(self,context)
			CursorPosition(self,context)
			CreateNewObject(self,context)
			CreateModifier(self,context)
			SetVisualSetings(self, context)
			TransformObject(self, context)
			CalculateNormal(self,context)
			self.StartMouseLocation = GetMouseLocation(self,event, context)
			print('StartMouseLocation', self.StartMouseLocation)

			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "is't 3dview")
			return {'CANCELLED'}

classes = (DestuctiveExtrude)

def register():
	bpy.utils.register_class(classes)


def unregister():
	bpy.utils.unregister_class(classes)


if __name__ == "__main__":
	register()
