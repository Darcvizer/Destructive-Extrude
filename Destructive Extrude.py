#Импорт библиотек
import bpy
import bmesh 
from bpy.props import IntProperty, FloatProperty
bl_info = {
"name": "Destructive Extrude :)",
"location": "View3D > Add > Mesh > Destructive Extrude,",
"description": "Extrude how sketch up. (Warning! Not work snaping)",
"author": "Vladislav Kindushov",
"version": (0,4),
"blender": (2, 7, 8),
"category": "Mesh",
}
class DestructiveExtrude(bpy.types.Operator):
	bl_idname = "destructive.extrude"
	bl_label = "Destructive Extrude"
	bl_options = {'REGISTER', 'UNDO'}


	first_mouse_x = IntProperty()
	first_value = FloatProperty()
	
	
	
	@classmethod
	def poll(cls, context):
		return (context.active_object is not None) and (context.mode == "EDIT_MESH")
###################################################################
####bpy.context.area.width 
###################################################################
#-----------------------------------------------------------------------------------------------
	def modal(self, context, event):
			object_B_Name = bpy.context.active_object.name
			object_A_Name = None
			sel_obj = bpy.context.selected_objects
			for i in sel_obj:
				object_A_Name = i.name
				#print(object_A_Name)
			#print(object_A_Name)
			#print(object_B_Name)
			if event.type == 'MOUSEMOVE':
				delta = self.first_mouse_x - event.mouse_x
				tic = bpy.context.active_object.modifiers["Solidify"]
				#self.first_value = delta * 0.001
				tic.thickness = delta * 0.001 + self.first_value if event.shift else  delta * 0.01
				if bpy.context.active_object.modifiers["Solidify"].thickness < 0:
					bpy.data.objects[object_A_Name].modifiers["Boolean"].operation = 'UNION'
				elif bpy.context.active_object.modifiers["Solidify"].thickness > 0:
					bpy.data.objects[object_A_Name].modifiers["Boolean"].operation = 'DIFFERENCE'
					
					
					
			elif event.type in {'RIGHTMOUSE', 'ESC'}:
				bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Solidify")
				#B_name = bpy.context.active_object.name
				bpy.context.scene.objects.active = bpy.data.objects[object_A_Name]
				bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Boolean")
				bpy.data.objects[object_A_Name].select = False
				bpy.data.objects[object_B_Name].select = True
				bpy.ops.object.delete()
				bpy.data.objects[object_A_Name].select = True
				bpy.data.objects[object_A_Name].show_wire = False
				bpy.data.objects[object_A_Name].show_all_edges = False
				object_A_D = bpy.context.object
				for i in object_A_D.modifiers:
					if i.type == ('BEVEL'):
						i.show_viewport = True
				bpy.ops.object.mode_set(mode='EDIT')
				bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = True

				bpy.ops.ed.undo()
				return {'FINISHED'}
			
			
			
			
			elif event.type in {'LEFTMOUSE'}:
				bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Solidify")
				#B_name = bpy.context.active_object.name
				bpy.context.scene.objects.active = bpy.data.objects[object_A_Name]
				bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Boolean")
				bpy.data.objects[object_A_Name].select = False
				bpy.data.objects[object_B_Name].select = True
				bpy.ops.object.delete()
				bpy.data.objects[object_A_Name].select = True
				bpy.data.objects[object_A_Name].show_wire = False
				bpy.data.objects[object_A_Name].show_all_edges = False
				object_A_D = bpy.context.object
				for i in object_A_D.modifiers:
					if i.type == ('BEVEL'):
						i.show_viewport = True
				bpy.ops.object.mode_set(mode='EDIT')
				bpy.ops.mesh.select_all(action='SELECT')
				bpy.ops.mesh.remove_doubles()
				bpy.ops.transform.edge_bevelweight(value=-1)
				bpy.ops.mesh.select_all(action='DESELECT')
				bpy.ops.mesh.edges_select_sharp()
				bpy.ops.transform.edge_bevelweight(value=1)
				bpy.ops.mesh.select_all(action='DESELECT')
				bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = True
				
				
				#bpy.context.active_object.modifiers["Solidify"].thickness = self.first_value
				return {'FINISHED'}
	
			return {'RUNNING_MODAL'}
	
	def invoke(self, context, event):
		bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = False
		object_A = bpy.context.active_object
		object_A_Name = bpy.context.active_object.name
		object_A_D = bpy.context.object
		bpy.data.objects[object_A_Name].show_wire = True
		bpy.data.objects[object_A_Name].show_all_edges = True
		
		C_name = object_A_Name
		
		#bpy.data.objects[object_A_Name].modifiers["Bavel"].show_viewport = False

		for i in object_A_D.modifiers:
			if i.type == ('BEVEL'):
				i.show_viewport = False

		#Separate
		bm = bmesh.from_edit_mesh(object_A.data) 
		face = bm.select_history
		bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode":1})
		bpy.ops.mesh.separate(type='SELECTED')
		bpy.ops.object.mode_set(mode='OBJECT')
		sel_obj = bpy.context.selected_objects

		object_B= None
		object_B_Name = None

		for i in sel_obj:
			object_B_Name = i.name
			object_B = i
			break

		#create bool for A object
		Bool = bpy.ops.object.modifier_add(type='BOOLEAN')
		object_A = Bool
		bpy.context.object.modifiers["Boolean"].operation = 'DIFFERENCE'
		bpy.context.object.modifiers["Boolean"].object = object_B
		bpy.context.object.modifiers["Boolean"].solver = 'CARVE'

		# set active B object
		bpy.context.scene.objects.active = bpy.data.objects[object_B_Name]

		# Create shel for B object
		bpy.data.objects[object_B_Name].draw_type = 'WIRE'
		#bpy.context.active_object.hide
		shel = bpy.ops.object.modifier_add(type='SOLIDIFY')
		bpy.data.objects[object_B_Name].modifiers["Solidify"].use_even_offset = True
		object_B = shel
		#bpy.ops.object.origin_set(type = 'ORIGIN_GEOMETRY')

		if context.object:
			self.first_mouse_x = event.mouse_x
			#self.first_value = bpy.context.active_object.modifiers["Solidify"].thickness

			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "No active object, could not finish")
			return {'CANCELLED'}


#-----------------------------------------------------------------------------------------------
		return {'FINISHED'}

def operator_draw(self,context):
    layout = self.layout
    col = layout.column(align=True)
    col.operator('destructive.extrude', text="Destructive Extrude")

def register():
	bpy.utils.register_class(DestructiveExtrude)
	bpy.types.VIEW3D_MT_edit_mesh_extrude.append(operator_draw)
	kc = bpy.context.window_manager.keyconfigs.addon
	if kc:
		km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
		kmi = km.keymap_items.new('destructive.extrude', 'L', 'PRESS',)

def unregister():
	bpy.utils.unregister_class(DestructiveExtrude)
	bpy.types.VIEW3D_MT_edit_mesh_extrude.remove(operator_draw)
	kc = bpy.context.window_manager.keyconfigs.addon
	if kc:
		km = kc.keymaps["3D View"]
		for kmi in km.keymap_items:
			if kmi.idname == 'destructive.extrude':
				km.keymap_items.remove(kmi)
				break

if __name__ == "__main__":
	register()
