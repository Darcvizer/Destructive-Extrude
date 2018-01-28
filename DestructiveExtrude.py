import bpy
import bmesh
from math import degrees
from mathutils import Vector, kdtree
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
	"version": (0, 9, 0),
	"blender": (2, 7, 8),
	"category": "Mesh",
}


def draw_callback_px(self, context):
	width = None
	for region in bpy.context.area.regions:
		if region.type == "TOOLS":
			width = region.width
			break
	
	font_id = 0
	
	bgl.glColor4f(1, 1, 1, 0.5)
	blf.position(font_id, width + 15, 60, 0)
	blf.size(font_id, 20, 72)
	blf.draw(font_id, "Offset: ")
	blf.position(font_id, width + 85, 60, 0)
	blf.size(font_id, 20, 72)
	if self.draw:
		blf.draw(font_id, self.key)
	else:
		if self.var[1].modifiers['Solidify'].thickness < 0.0:
			blf.draw(font_id, self.key[:6])
		else:
			blf.draw(font_id, self.key[:5])
	
	# restore opengl defaults
	bgl.glDisable(bgl.GL_BLEND)
	bgl.glColor4f(0.0, 0.0, 0.0, 1.0)


def RayCast(self, context, event, ray_max=1000.0):
	"""Run this function on left mouse, execute the ray cast"""
	
	def visible_objects_and_duplis():
		"""Loop over (object, matrix) pairs (mesh only)"""
		for obj in context.visible_objects:
			if obj.type == 'MESH':
				yield (obj, obj.matrix_world.copy())
			if obj.dupli_type != 'NONE':
				obj.dupli_list_create(scene)
				for dob in obj.dupli_list:
					obj_dupli = dob.object
					if obj_dupli.type == 'MESH':
						yield (obj_dupli, dob.matrix.copy())
			obj.dupli_list_clear()
	
	def obj_ray_cast(obj, matrix):
		"""Wrapper for ray casting that moves the ray into object space"""
		# get the ray relative to the object
		matrix_inv = matrix.inverted()
		ray_origin_obj = matrix_inv * ray_origin
		ray_target_obj = matrix_inv * ray_target
		ray_direction_obj = ray_target_obj - ray_origin_obj
		d = ray_direction_obj.length
		ray_direction_obj.normalize()
		success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)
		if face_index != -1:
			return location, normal, face_index
		else:
			return None, None, None
	
	# get the context arguments
	scene = context.scene
	region = context.region
	rv3d = context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	# get the ray from the viewport and mouse
	view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()
	ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
	ray_target = ray_origin + view_vector
	# cast rays and find the closest object
	best_length_squared = -1.0
	best_obj = None
	best_matrix = None
	best_face = None
	best_hit = None
	
	for obj, matrix in visible_objects_and_duplis():
		if obj.name == self.d_obj.d_obj.name:
			continue
		if obj.type == 'MESH':
			hit, normal, face_index = obj_ray_cast(obj, matrix)
			if hit is not None:
				hit_world = matrix * hit
				scene.cursor_location = hit_world
				length_squared = (hit_world - ray_origin).length_squared
				if best_obj is None or length_squared < best_length_squared:
					best_length_squared = length_squared
					best_obj = obj
					best_matrix = matrix
					best_face = face_index
					best_hit = hit
					break
	
	def run(best_obj, best_matrix, best_face, best_hit):
		best_distance = float("inf")  # use float("inf") (infinity) to have unlimited search range
		mesh = best_obj.data
		best_matrix = best_obj.matrix_world
		for vert_index in mesh.polygons[best_face].vertices:
			vert_coord = mesh.vertices[vert_index].co
			distance = (vert_coord - best_hit).magnitude
			if distance < best_distance:
				best_distance = distance
				scene.cursor_location = best_matrix * vert_coord
		for v0, v1 in mesh.polygons[best_face].edge_keys:
			p0 = mesh.vertices[v0].co
			p1 = mesh.vertices[v1].co
			p = (p0 + p1) / 2
			distance = (p - best_hit).magnitude
			if distance < best_distance:
				best_distance = distance
				scene.cursor_location = best_matrix * p
		face_pos = Vector(mesh.polygons[best_face].center)
		distance = (face_pos - best_hit).magnitude
		if distance < best_distance:
			best_distance = distance
			scene.cursor_location = best_matrix * face_pos
	
	try:
		run(best_obj, best_matrix, best_face, best_hit)
	except:
		pass


class M_Object:
	def __init__(self, context):
		self.m_Obj = context.active_object  # main object
		self.center_selection = None
		self.direction = Vector((0.0, 0.0, 0.0))
		self.n_offset, self.w_offset = self.__offset(context)
		# bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})
		bpy.ops.object.mode_set(mode='OBJECT')
		self.show_all_edges = self.m_Obj.show_all_edges
		self.show_wire = self.m_Obj.show_wire
		self.m_Obj.show_wire = True
		self.m_Obj.show_all_edges = True
		self.u_modifier = []  # save a on user modifier
		self.__Off_All_Modifier(context)
		self.index_bool_modifier = self.__Create_Boolean_Modifier(context)  # Index boolean modifiers
	
	def __Off_All_Modifier(self, context):
		'''get vertex for offset'''
		for i in self.m_Obj.modifiers:
			if i.show_viewport:
				self.u_modifier.append(i)
				i.show_viewport = False
	
	def __Create_Boolean_Modifier(self, context):
		'''Create Booleain Modifier and Return Modifier Index'''
		bpy.ops.object.modifier_add(type='BOOLEAN')
		for j, i in enumerate(self.m_Obj.modifiers):
			if i.type == 'BOOLEAN' and i.show_viewport:
				i.operation = 'DIFFERENCE'
				i.object = context.selected_objects[0]
				i.solver = 'CARVE'
				return j
	
	def __offset(self, context):
		bm = bmesh.from_edit_mesh(self.m_Obj.data)
		
		distance = 0.000002
		#distance = 0.2
		coordNof = []
		coordWof = []
		center = Vector((0.0, 0.0, 0.0))
		
		for i in bm.select_history:
			for edge in i.edges:
				if edge.link_faces[0].select and edge.link_faces[1].select:
					continue
				
				angle = degrees(edge.calc_face_angle_signed())
				if angle > 89.999:
					nor = [i.normal for i in edge.link_faces if not i.select]
					v1, v2 = edge.verts
					coordNof.append(v1.co.copy())
					coordNof.append(v2.co.copy())
					
					coordWof.append(v1.co.copy() + nor[0] * distance)
					coordWof.append(v2.co.copy() + nor[0] * distance)
			center += i.calc_center_median()
			self.direction += self.m_Obj.matrix_world.to_3x3() * i.normal.copy()
		
		self.center_selection = self.m_Obj.matrix_world * (center / len(bm.select_history))
		bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})
		bpy.ops.mesh.separate(type='SELECTED')
		return coordNof, coordWof
	
	def GetBool(self):
		return self.m_Obj[self.index_bool_modifier]


class D_Object:
	def __init__(self, context, n_offset, w_offset):
		# Use in normal move
		self.d_obj = context.selected_objects[0]  # extrude object
		self.i_offset = []  # index offset vertex
		self.n_offset = [context.active_object.matrix_world * i for i in n_offset]  # coordinate with not offset
		self.w_offset = [context.active_object.matrix_world * i for i in w_offset]  # coordinate with offset
		# .__GetIndexForOffsetSolidify(context)
		# self.__CreateSolidifityModifier(context)
		# self.d_obj.hide = True
		self.d_obj.draw_type = 'WIRE'
		# Use in axis mode
		self.vertx_for_move = []  # these vertices will move
		self.state_solydifity = 'DIFFERENCE'
		self.index_for_axis = None
		print('n_offset', self.n_offset[:])
		print('n_offset', self.w_offset[:])
		# self.i_For_Comp_Axis = []  # these vertices responsible for maintaining proportions
		# self.save_Coord = []  # save coordinate for change axis
		# self.constrain_axis = False  # Looks after whether or not to prepare the object for movement along the axis
		# self.i_offset2 = []
		# self.axis = False
		# self.KDT = self.__KDTree(self.d_obj.data)
		# self.KDA = None
		# self.temp_verts = None
		# self.S_val = 0
		#
		# self.i_For_Comp_Axis2 = []
		#
		self.__GetIndexForOffsetSolidify(context)
		
		self.__CreateSolidifityModifier(context)
	
	def __GetIndexForOffsetSolidify(self, context):
		for i in self.n_offset:
			for j in self.d_obj.data.vertices:
				if self.d_obj.matrix_world * j.co == i:
					self.i_offset.append(j.index)
		self.SwapCoordinate(context, self.w_offset)
	
	def __CreateSolidifityModifier(self, context):
		for i in self.d_obj.modifiers:
			self.d_obj.modifiers.remove(i)
		
		self.d_obj.modifiers.new('Solidify', 'SOLIDIFY')
		self.d_obj.modifiers[0].use_even_offset = True
		self.d_obj.modifiers[0].use_quality_normals = True
		self.d_obj.modifiers[0].thickness = 0.1
	
	def SetValSolidifity(self, context, value):
		self.d_obj.modifiers[0].thickness = value
	
	def SwapCoordinate(self, context, mode):
		for j, i in enumerate(self.i_offset):
			self.d_obj.data.vertices[i].co = mode[j]
	
	def SetObjForAxisMove(self, context):
		bpy.ops.object.select_all(action='DESELECT')
		indexV = [i.index for i in self.d_obj.data.vertices]
		indexF = [i.index for i in self.d_obj.data.polygons]
		self.d_obj.modifiers[0].thickness = 0.0
		act = context.active_object
		context.scene.objects.active = self.d_obj
		context.active_object.select = True
		bpy.ops.object.convert(target='MESH')
		for i in indexF:
			self.d_obj.data.polygons[i].select = True
		bpy.ops.object.mode_set(mode='EDIT')
		bpy.ops.mesh.select_more()
		bpy.ops.mesh.select_all(action='INVERT')
		bpy.ops.object.mode_set(mode='OBJECT')
		coord = [i.center.copy() for i in self.d_obj.data.polygons if i.select]
		# bpy.ops.object.select_all(action='DESELECT')
		# context.active_object.select = True
		# bpy.ops.object.convert(target='MESH')
		# bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
		context.scene.objects.active = act
		
		self.index_for_axis = []
		for i in self.d_obj.data.vertices:
			if i.index not in indexV:
			#if i.select:
				self.index_for_axis.append(i.index)
				#i.select = True
	
	def Move(self, context, point, axis):
		
		len_vert = len(self.index_for_axis)
		center = Vector((0.0, 0.0, 0.0))
		
		for i in self.index_for_axis:
			center = self.d_obj.data.vertices[i].co.copy() + center
		for j,n in enumerate(axis[:]):
			if n:
				for i in self.index_for_axis:
					pos = (self.d_obj.data.vertices[i].co.copy()-center/len_vert) + point
					self.d_obj.data.vertices[i].co[j] = pos[j]




class Util:
	def __init__(self, context, event, obj, modifer, auto_merge, center, dir):
		self.center = center
		self.direction = dir
		self.starMousePosPoint, self.starMousePosValuem = self.__StarPosMouse(context, event)
		self.lasetPosMouse = None
		self.starMousePosForAxis = False
		self.modifier = modifer
		self.auto_snap = bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge
		bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = auto_merge
		self.show_wire = obj.show_wire
		self.show_all_edges = obj.show_all_edges
		obj.show_wire = True
		obj.show_all_edges = True
	
	def __StarPosMouse(self, context, event):
		region = bpy.context.region
		rv3d = bpy.context.region_data
		coord = event.mouse_region_x, event.mouse_region_y
		view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
		ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
		
		pointLoc = intersect_line_plane(ray_origin_mouse, ray_origin_mouse + view_vector_mouse,
		                                self.center, rv3d.view_rotation * Vector((0.0, 0.0, -1.0)), False)
		
		return pointLoc, (self.direction * pointLoc) * -1
	
	def EventMouseNormal(self, context, event):
		region = bpy.context.region
		rv3d = bpy.context.region_data
		coord = event.mouse_region_x, event.mouse_region_y
		view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
		ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
		
		pointLoc = intersect_line_plane(ray_origin_mouse, ray_origin_mouse + view_vector_mouse,
		                                self.center, rv3d.view_rotation * Vector((0.0, 0.0, -1.0)), False)
		pointLoc = pointLoc
		self.lasetPosMouse = pointLoc
		context.scene.cursor_location= pointLoc
		return pointLoc, (self.direction * pointLoc) * -1

	def Finish(self, context, m_obj, d_obj,wire,all_edge, mode, u_modifier):
		bpy.ops.object.select_all(action='DESELECT')
		
		bpy.ops.object.modifier_apply(modifier=m_obj.modifiers[-1].name)
		context.scene.objects.active = d_obj.d_obj
		
		d_obj.SwapCoordinate(context, d_obj.n_offset)
		
		
		if not mode:
			index = [i.index for i in d_obj.d_obj.data.polygons]
			bpy.ops.object.modifier_apply(modifier=d_obj.d_obj.modifiers[-1].name)
			context.active_object.select = True
			bpy.ops.object.convert(target='MESH')
			for i in index:
				d_obj.d_obj.data.polygons[i].select= True
			bpy.ops.object.mode_set(mode='EDIT')
			bpy.ops.mesh.select_more()
			bpy.ops.mesh.select_all(action='INVERT')
			bpy.ops.object.mode_set(mode='OBJECT')
			coord = [i.center.copy() for i in d_obj.d_obj.data.polygons if i.select]
		else:
			coord = [i.center.copy() for i in d_obj.d_obj.data.polygons if i.select]
		for i in m_obj.data.polygons:
			if i.center in coord:
				i.select = True
		m_obj.show_wire =  wire
		m_obj.show_all_edges = all_edge
		
		for i in m_obj.modifiers:
			if i in u_modifier:
				i.show_viewport = True
		bpy.data.objects.remove(d_obj.d_obj)
		context.scene.objects.active = m_obj
		bpy.ops.object.mode_set(mode='EDIT')
		
	def Cancel(self, context, m_obj, d_obj, wire, all_edge, mode, u_modifier):
		
		m_obj.modifiers.remove(m_obj.modifiers[-1])
		bpy.data.objects.remove(d_obj.d_obj)
		#bpy.context.scene.objects.unlink(d_obj.d_obj)
		m_obj.show_wire = wire
		m_obj.show_all_edges = all_edge
		
		for i in m_obj.modifiers:
			if i in u_modifier:
				i.show_viewport = True
				

		context.scene.objects.active = m_obj
		bpy.ops.object.mode_set(mode='EDIT')
		
		
class DestructiveExtrude(bpy.types.Operator):
	bl_idname = "mesh.destructive_extrude"
	bl_label = "Destructive Extrude"
	bl_options = {"REGISTER", "UNDO", "GRAB_CURSOR", "BLOCKING"}
	
	@classmethod
	def poll(cls, context):
		return (context.mode == "EDIT_MESH")
	
	def modal(self, context, event):
		if event.type == 'LEFTMOUSE':
			self.v3d.Finish(context,self.m_obj.m_Obj,self.d_obj,self.m_obj.show_wire, self.m_obj.show_all_edges,self.axis_mode, self.m_obj.u_modifier)
			return {'FINISHED'}
		if event.type == 'RIGHTMOUSE':
			self.v3d.Cancel(context,self.m_obj.m_Obj,self.d_obj,self.m_obj.show_wire, self.m_obj.show_all_edges,self.axis_mode, self.m_obj.u_modifier)
			return {'CANCELLED'}
		# if event.type == 'MOUSEMOVE':
		# 	if self.axis_mode:
		# 		point, value = self.v3d.EventMouseNormal(context, event)
		# 		self.d_obj.Move(context, value, self.axis.normalized())
		# 		pass
		# 	else:
		# 		point, value = self.v3d.EventMouseNormal(context, event)
		# 		self.d_obj.SwapCoordinate(context, self.m_obj.w_offset)
		# 		self.d_obj.d_obj.modifiers[0].thickness = (value - self.v3d.starMousePos)#.length
		# print ('value', value)
		
		####___________________FIX___AFTER___________________________###
		# print('test', context.active_object.matrix_world.to_3x3() * value.normalized())
		# if self.value[1] > (context.active_object.matrix_world.to_3x3() * value.normalized())[1]:
		# 	if self.m_obj.m_Obj.modifiers[-1].operation != 'DIFFERENCE':
		# 		self.m_obj.m_Obj.modifiers[-1].operation = 'DIFFERENCE'
		# 		self.m_obj.SwapCoordinate(self, context, self.m_obj.w_offset)
		# 	self.d_obj.d_obj.modifiers[0].thickness = (value - self.v3d.starMousePos).length
		# else:
		# 	if self.m_obj.m_Obj.modifiers[-1].operation == 'DIFFERENCE':
		# 		self.m_obj.m_Obj.modifiers[-1].operation = 'UNION'
		# 		self.m_obj.SwapCoordinate(self, context, self.m_obj.n_offset)
		# 	self.d_obj.d_obj.modifiers[0].thickness = -1 * (value - self.v3d.starMousePos).length
		####_________________________________________________________###
		
		if event.type == 'X' and self.axis != Vector((1.0, 0.0, 0.0)):
			self.axis_mode = True
			self.axis = Vector((1.0, 0.0, 0.0))
			if len(self.d_obj.d_obj.modifiers): self.d_obj.SetObjForAxisMove(context)
		if event.type == 'Y' and self.axis != Vector((0.0, 1.0, 0.0)):
			self.axis = Vector((0.0, 1.0, 0.0))
			self.axis_mode = True
			if len(self.d_obj.d_obj.modifiers): self.d_obj.SetObjForAxisMove(context)
		if event.type == 'Z' and self.axis != Vector((0.0, 0.0, 1.0)):
			self.axis = Vector((0.0, 0.0, 1.0))
			self.axis_mode = True
			if len(self.d_obj.d_obj.modifiers): self.d_obj.SetObjForAxisMove(context)
		
		if event.ctrl:
			if self.axis_mode:
				RayCast(self, context, event, ray_max=1000.0)
				self.d_obj.Move(context, context.scene.cursor_location, self.axis.normalized())
			else:
				RayCast(self, context, event, ray_max=1000.0)
				distnace = float("inf")
				index = 0
				for i in self.d_obj.d_obj.data.vertices:
					tamp = (self.d_obj.d_obj.matrix_world * i.co.copy() - context.scene.cursor_location).length
					if tamp < distnace:
						distnace = tamp
						index = i.index
				v2 = self.d_obj.d_obj.matrix_local.inverted() * context.scene.cursor_location.copy()
				v1 = self.d_obj.d_obj.data.vertices[index].co.copy()
				dvec = v2 - v1
				dnormal = np.dot(dvec, self.v3d.direction)
				v2 = v1 + Vector(dnormal * self.v3d.direction)
				value = (v1 - v2).length
				self.d_obj.d_obj.modifiers[0].thickness = value
				
		elif event.type == 'MOUSEMOVE':
			if self.axis_mode:
				point, value = self.v3d.EventMouseNormal(context, event)
				self.d_obj.Move(context, point, self.axis.normalized())
			
			else:
				point, value = self.v3d.EventMouseNormal(context, event)
				value = (value - self.v3d.starMousePosValuem)
				if self.m_obj.m_Obj.modifiers[-1].operation == 'DIFFERENCE' and value < 0:
					self.m_obj.m_Obj.modifiers[-1].operation = 'UNION'
					self.d_obj.SwapCoordinate(context, self.d_obj.n_offset)
				elif self.m_obj.m_Obj.modifiers[-1].operation == 'UNION' and value > 0:
					self.m_obj.m_Obj.modifiers[-1].operation = 'DIFFERENCE'
					self.d_obj.SwapCoordinate(context, self.d_obj.w_offset)
				self.d_obj.d_obj.modifiers[0].thickness = value

		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}
	
	def invoke(self, context, event):
		if context.space_data.type == 'VIEW_3D':
			auto_merge = bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = False
			self.m_obj = M_Object(context)
			self.d_obj = D_Object(context, self.m_obj.n_offset, self.m_obj.w_offset)
			self.v3d = Util(context, event, self.d_obj.d_obj, self.m_obj.u_modifier, auto_merge,
			                self.m_obj.center_selection, self.m_obj.direction)
			self.enter = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '+', '*', '/', '.', ',']
			#self.value = context.active_object.matrix_world.to_3x3() * self.v3d.EventMouseNormal(context,event).normalized()
			self.axis_mode = False
			self.axis = Vector((0.0, 0.0, 0.0))
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "is't 3dview")
			return {'CANCELLED'}


def operator_draw(self, context):
	layout = self.layout
	col = layout.column(align=True)
	self.layout.operator_context = 'INVOKE_REGION_WIN'
	col.operator("mesh.destructive_extrude", text="Destructive Extrude")


def register():
	bpy.utils.register_class(DestructiveExtrude)
	bpy.types.VIEW3D_MT_edit_mesh_extrude.append(operator_draw)


def unregister():
	bpy.utils.unregister_class(DestructiveExtrude)
	bpy.types.VIEW3D_MT_edit_mesh_extrude.remove(operator_draw)


if __name__ == "__main__":
	register()

