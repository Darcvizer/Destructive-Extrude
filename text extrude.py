import bpy
import bmesh
from math import degrees
from mathutils import Vector, kdtree
from bpy_extras import view3d_utils
from bpy.props import StringProperty
import blf
import bgl
import os
from bpy.props import IntProperty, FloatProperty

from bpy.types import Operator
from math import sin, cos, pi, radians
from time import perf_counter
from bpy.props import (
    BoolProperty,
    FloatProperty,
    EnumProperty,)


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

class M_Object:
    def __init__(self, context):
        self.m_Obj = context.active_object # main object

        self.n_offset, self.w_offset = self.__offset(context)
        bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')
        self.u_modifier = []# save a on user modifier
        self.__Off_All_Modifier(context)
        self.index_bool_modifier = self.__Create_Boolean_Modifier(context) # Index boolean modifiers

    def __Off_All_Modifier(self,context):
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

    def __GetCoord(self, context, face, ind=[], coor=[]):
        '''Get coordinate vertex for offset'''
        bm = bmesh.from_edit_mesh(self.m_Obj.data)
        bm.faces.ensure_lookup_table()
        if not ind:
            coord = []
            index = []
            for f in face:
                for e in bm.faces[f].edges:
                    if abs(degrees(e.calc_face_angle_signed())) > 29.999:
                        e.select = True
                        for v in e.verts:
                            if v.index not in index:
                                coord.append(v.co.copy())
                                index.append(v.index)
            return coord, index
        else:
            bm.verts.ensure_lookup_table()
            for j, i in enumerate(ind):
                bm.verts[i].co = coor[j]
            for i in face:
                bm.faces[i].select = True

    def __offset(self, context):
        bm = bmesh.from_edit_mesh(self.m_Obj.data)
        face_selection = [f.index for f in bm.faces if f.select]
        for f in face_selection:
                bm.faces[f].select = False
        coordNof, index = self.__GetCoord(context, face=face_selection)
        bpy.ops.mesh.offset_edges(geometry_mode='move', width=0.00002)
        coordWof, index = self.__GetCoord(context, face=face_selection)
        bm = bmesh.from_edit_mesh(self.m_Obj.data)
        self.__GetCoord(context, face=face_selection, ind=index, coor=coordNof)

        return coordNof, coordWof

    def GetBool(self):
        return self.m_Obj[self.index_bool_modifier]

class D_Object:
    def __init__(self, context, n_offset, w_offset):
        # Use in normal move
        self.d_obj = context.selected_objects[0] # extrude object
        self.i_offset = [] # index offset vertex
        self.n_offset = n_offset # coordinate with not offset
        self.w_offset = w_offset # coordinate with offset
        # self.d_obj.hide = True
        self.d_obj.draw_type = 'WIRE'
        # Use in axis mode
        self.Normal = Vector((0.0, 0.0, 1.0)) # normal the first face for detect side offset
        self.vertx_for_move = [] # these vertices will move
        self.i_For_Comp_Axis = [] # these vertices responsible for maintaining proportions
        self.save_Coord = [] # save coordinate for change axis
        self.constrain_axis = False # Looks after whether or not to prepare the object for movement along the axis
        self.i_offset2 = []
        self.axis = False


        self.__GetIndexForOffset(context)
        self.__CreateSolidifityModifier(context)

    def __GetIndexForOffset(self, context):
        '''get vertex index for offset'''
        tempN = []
        tempW = []
        for i in self.d_obj.data.vertices:
                for j, o in enumerate(self.n_offset):
                    if o == i.co:
                        tempN.append(self.n_offset[j])
                        tempW.append(self.w_offset[j])
                        self.i_offset.append(i.index)
        self.n_offset = tempN
        self.w_offset = tempW

    def __GetIndexForOffsetSolidify(self, context):
        for i in self.i_offset:
            for j in self.d_obj.data.vertices:
                if self.d_obj.data.vertices[i].co == j.co and j.index != i:
                    self.i_offset2.append(j.index)

    def __CreateSolidifityModifier(self, context):
        for i in self.d_obj.modifiers:
            self.d_obj.modifiers.remove(i)

        self.d_obj.modifiers.new('Solidify', 'SOLIDIFY')
        self.d_obj.modifiers[0].use_even_offset = True
        #self.d_obj.modifiers[0].use_flip_normals = True

    def SetValSolidifity(self, context, loc, bool):
        self.d_obj.modifiers[0].thickness = loc
        self.__Swap(context, bool)

    def __SwapCoordinate(self, context, coord):
        for j, i in enumerate(self.i_offset):
            self.d_obj.data.vertices[i].co = coord[j]
        if len(self.i_offset2) > 0:
            for j, i in enumerate(self.i_offset2):
                self.d_obj.data.vertices[i].co = coord[j]

    def __Swap(self, context, bool):
        if self.d_obj.modifiers[0].thickness > 0 and 'UNION':
            context.active_object.modifiers[bool].operation = 'DIFFERENCE'
            self.__SwapCoordinate(context, self.w_offset)
        elif self.d_obj.modifiers[0].thickness < 0 and 'DIFFERENCE':
            context.active_object.modifiers[bool].operation = 'UNION'
            self.__SwapCoordinate(context, self.n_offset)

    def __SetMeshForAxisConstrain(self, context):
        a = context.active_object
        self.Normal = self.d_obj.data.polygons[0].normal.copy()
        vert = [i.index for i in self.d_obj.data.vertices] # save vertex for invert
        bpy.context.scene.objects.active = self.d_obj
        if self.d_obj.modifiers[0].thickness < 0:
            for i in self.d_obj.data.polygons:
                i.flip()

        self.d_obj.modifiers[0].thickness = 0
        bpy.ops.object.modifier_apply(modifier=self.d_obj.modifiers[0].name)
        for i in self.d_obj.data.vertices:
            if i.index not in vert:
                self.vertx_for_move.append(i.index)
        for x in self.vertx_for_move[1:]:
            self.i_For_Comp_Axis.append((self.d_obj.data.vertices[x].co.copy() - self.d_obj.data.vertices[self.vertx_for_move[0]].co.copy()))
        for i in self.vertx_for_move:
            self.save_Coord.append(self.d_obj.data.vertices[i].co.copy())
        self.__GetIndexForOffsetSolidify(context)
        bpy.context.scene.objects.active = a

    def Move(self, context, loc, bool, axis):
        if not self.constrain_axis and type(axis) != int:
            self.SetValSolidifity(context, loc, bool)
        elif type(axis) == int and not self.constrain_axis:
            self.__SetMeshForAxisConstrain(context)
            self.constrain_axis = True
            self.__setMove(context, loc, axis, bool)
        else:
            self.__setMove(context, loc, axis, bool)

    def __ReturnCoord(self, context):
        for j, i in enumerate(self.vertx_for_move):
            self.d_obj.data.vertices[i].co =  self.save_Coord[j]

    def __setMove(self, context, loc, axis, bool):
        if self.axis != axis:
            self.axis = axis
            self.__ReturnCoord(context)
        for j, i in enumerate(self.vertx_for_move):
            if i == self.vertx_for_move[0]:
                self.d_obj.data.vertices[i].co[axis] = loc[axis]
            else:
                x = self.i_For_Comp_Axis[j-1]
                self.d_obj.data.vertices[i].co[axis] = loc[axis] + x[axis]
        self.__SwapBool(context, bool, axis, loc)

    def __SwapBool(self, context, bool, axis, loc):
        if self.Normal[2] > 0.0:
            if loc[axis] < self.save_Coord[0][axis]:
                if context.active_object.modifiers[bool].operation == 'UNION':
                    context.active_object.modifiers[bool].operation = 'DIFFERENCE'
                    self.__SwapCoordinate(context, self.w_offset)
                    for i in self.d_obj.data.polygons:
                        i.flip()


            else:
                if context.active_object.modifiers[bool].operation == 'DIFFERENCE':
                    context.active_object.modifiers[bool].operation = 'UNION'
                    self.__SwapCoordinate(context, self.n_offset)
                    for i in self.d_obj.data.polygons:
                        i.flip()
        else:
            if loc[axis] > self.save_Coord[0][axis]:
                if context.active_object.modifiers[bool].operation == 'UNION':
                    context.active_object.modifiers[bool].operation = 'DIFFERENCE'
                    self.__SwapCoordinate(context, self.w_offset)
                    for i in self.d_obj.data.polygons:
                        i.flip()


            else:
                if context.active_object.modifiers[bool].operation == 'DIFFERENCE':
                    context.active_object.modifiers[bool].operation = 'UNION'
                    self.__SwapCoordinate(context, self.n_offset)
                    for i in self.d_obj.data.polygons:
                        i.flip()




class Util:
    def __init__(self, context, event, obj, modifer):
        self.starMousePos = self.__StarPosMouse(context, event, obj)
        self.starMousePosForAxis = False
        self.modifier = modifer
        self.auto_snap = bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge
        bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = False
        self.show_wire = obj.show_wire
        self.show_all_edges = obj.show_all_edges
        obj.show_wire = True
        obj.show_all_edges = True


    def __StarPosMouse(self, context, event, obj, mode=False):
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)
        if type(mode) != int:
            normal = obj.data.polygons[0].normal
            loc = ((normal * -1) * loc)
            return loc
        return loc

    def EventMouseNormal(self, context, event, obj, mode=False):
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)
        if type(mode) != int:
            normal = obj.data.polygons[0].normal.copy()
            loc = ((((normal * -1) * loc) - self.starMousePos) / self.Zoom(context))
            loc *= 4
        elif not self.starMousePosForAxis:
            self.starMousePos = self.__StarPosMouse(context, event, obj, mode=True)
        return loc

    def Zoom(self, context):
        ar = None
        for i in bpy.context.window.screen.areas:
            if i.type == 'VIEW_3D': ar = i
        ar = ar.spaces[0].region_3d.view_distance
        return ar

    def __RetModifier(self, obj):
        for i in obj.modifiers:
            if i in self.modifier:
                i.show_viewport = True


    def Finish(self, context, bool_index, m_obj, d_obj):
        bpy.ops.object.modifier_apply(modifier=m_obj.modifiers[bool_index].name)
        bpy.context.scene.objects.unlink(d_obj)
        bpy.data.objects.remove(d_obj)
        bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = self.auto_snap
        m_obj.show_wire = self.show_wire
        m_obj.show_all_edges = self.show_all_edges
        self.__RetModifier(m_obj)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.select_all(action='DESELECT')

    def Cancel(self, context, bool_index, m_obj, d_obj):
        m_obj.modifiers.remove(m_obj.modifiers[bool_index])
        bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = self.auto_snap
        m_obj.show_wire = self.show_wire
        m_obj.show_all_edges = self.show_all_edges
        bpy.context.scene.objects.unlink(d_obj)
        bpy.data.objects.remove(d_obj)
        self.__RetModifier(m_obj)
        bpy.ops.object.mode_set(mode='EDIT')





class DestructiveExtrude(bpy.types.Operator):
    bl_idname = "mesh.destructive_extrude"
    bl_label = "Destructive Extrude"
    bl_options = {"REGISTER", "UNDO", "GRAB_CURSOR", "BLOCKING"}
    axis = False
    @classmethod
    def poll(cls, context):
        return (context.mode == "EDIT_MESH")

    def modal(self, context, event):
        if event.type == 'Q':
            return {'FINISHED'}
        if event.type == 'MOUSEMOVE':
            #self.d_obj.SetValSolidifity(context, self.v3d.EventMouseNormal(context, event, self.d_obj.d_obj), self.m_obj.index_bool_modifier)
            self.d_obj.Move(context, self.v3d.EventMouseNormal(context, event, self.d_obj.d_obj, mode=self.axis), self.m_obj.index_bool_modifier, axis=self.axis)

        if event.type == 'LEFTMOUSE':
            self.v3d.Finish(context, self.m_obj.index_bool_modifier, self.m_obj.m_Obj, self.d_obj.d_obj)
            return {'FINISHED'}

        if event.type == 'RIFGTMOUSE':
            self.v3d.Cancel(context, self.m_obj.index_bool_modifier, self.m_obj.m_Obj, self.d_obj.d_obj)
            return {'FINISHED'}
        if event.type == 'X':
            self.axis = 0
        if event.type == 'Y':
            self.axis = 1
        if event.type == 'Z':
            self.axis = 2
            print(self.axis)

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            self.m_obj = M_Object(context)
            self.d_obj = D_Object(context, self.m_obj.n_offset, self.m_obj.w_offset)
            self.v3d = Util(context, event, self.d_obj.d_obj, self.m_obj.u_modifier)

            #args = (self, context)
            #self._handle = bpy.types.SpaceView          3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "is't 3dview")
            return {'CANCELLED'}



def register():
    bpy.utils.register_class(DestructiveExtrude)
def unregister():
    bpy.utils.unregister_class(DestructiveExtrude)
if __name__ == "__main__":
    register()
