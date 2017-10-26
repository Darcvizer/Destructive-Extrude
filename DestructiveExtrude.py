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

    run(best_obj, best_matrix, best_face, best_hit)

class M_Object:
    def __init__(self, context):
        self.m_Obj = context.active_object # main object

        self.n_offset, self.w_offset = self.__offset(context)
        #bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.m_Obj.show_wire = True
        self.m_Obj.show_all_edges = True
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

    # def __GetCoord(self, context, face, ind=[], coor=[]):
    #     '''Get coordinate vertex for offset'''
    #     bm = bmesh.from_edit_mesh(self.m_Obj.data)
    #     bm.faces.ensure_lookup_table()
    #     if not ind:
    #         coord = []
    #         index = []
    #         for f in face:
    #             for e in bm.faces[f].edges:
    #                 if abs(degrees(e.calc_face_angle_signed())) > 29.999:
    #                     e.select = True
    #                     for v in e.verts:
    #                         if v.index not in index:
    #                             coord.append(v.co.copy())
    #                             index.append(v.index)
    #         return coord, index
    #     else:
    #         bm.verts.ensure_lookup_table()
    #         for j, i in enumerate(ind):
    #             bm.verts[i].co = coor[j]
    #         for i in face:
    #             bm.faces[i].select = True

    def __offset(self, context):
        bm = bmesh.from_edit_mesh(self.m_Obj.data)
        face_selection = [f for f in bm.faces if f.select]
        save_pos = {}
        #distance = 0.000002
        distance = 0.2
        coordNof = []
        coordWof = []
        for i in bm.verts:
            if i.select:
                save_pos[i.index] = i.co.copy()
                coordNof.append(i.co.copy())
        move = {}

        for i in face_selection:
            for edge in i.edges:
                if edge.link_faces[0].select and edge.link_faces[1].select:
                    continue
                try:  # ____Fix opening edges
                    angle = degrees(edge.calc_face_angle_signed())
                    nor = [i.normal for i in edge.link_faces if not i.select]
                except:
                    continue

                if angle > 29.999:  # ____Fix
                    if angle > 90.001:
                        print(angle)
                        n = nor[0] + i.normal
                        for v in edge.verts:
                            if isinstance(move.get(v.index), type(None)):
                                v.co += n * distance
                                move[v.index] = nor

                            if move.get(v.index) != nor:
                                v.co += n * distance
                                move[v.index] = nor

                    for v in edge.verts:
                        print(angle)
                        if isinstance(move.get(v.index), type(None)):
                            v.co += nor[0] * distance
                            move[v.index] = nor


                        if move.get(v.index) != nor:
                            v.co += nor[0] * distance
                            move[v.index] = nor


        bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})
        bpy.ops.mesh.separate(type='SELECTED')
        bm.verts.ensure_lookup_table()
        for i in save_pos:
            coordWof.append(bm.verts[i].co.copy())
            bm.verts[i].co = save_pos.get(i)


        return coordNof, coordWof

    def GetBool(self):
        return self.m_Obj[self.index_bool_modifier]

class D_Object:
    def __init__(self, context, n_offset, w_offset):
        # Use in normal move
        self.d_obj = context.selected_objects[0] # extrude object
        self.__Apply_transform(context)
        self.i_offset = [] # index offset vertex
        self.n_offset = n_offset # coordinate with not offset
        self.w_offset = w_offset # coordinate with offset
        # self.d_obj.hide = True
        self.d_obj.draw_type = 'WIRE'
        # Use in axis mode
        self.Normal = self.__GetNormal(context) # normal the first face for detect side offset
        self.vertx_for_move = [] # these vertices will move


        self.i_For_Comp_Axis = [] # these vertices responsible for maintaining proportions
        self.save_Coord = [] # save coordinate for change axis
        self.constrain_axis = False # Looks after whether or not to prepare the object for movement along the axis
        self.i_offset2 = []
        self.axis = False
        self.KDT = self.__KDTree(self.d_obj.data)
        self.KDA = None
        self.temp_verts = None
        self.S_val = 0

        self.i_For_Comp_Axis2 = []



        self.__GetIndexForOffset(context)

        self.__CreateSolidifityModifier(context)

    def __Apply_transform(self, context):
        s = context.selected_objects
        bpy.ops.object.select_all(action='DESELECT')
        self.d_obj.select = True
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        for i in s:
            bpy.data.objects[i.name].select = True


    def __GetIndexForOffset(self, context):
        '''get vertex index for offset'''
        tempN = []
        tempW = []

        for n, p in enumerate(self.w_offset):
            for v in self.d_obj.data.vertices:
                wp = context.active_object.matrix_world * p
                vw = self.d_obj.matrix_world * v.co
                if wp == vw:
                    self.i_offset.append(v.index)
                    tempN.append(context.active_object.matrix_world * self.n_offset[n])
                    tempW.append(context.active_object.matrix_world * self.w_offset[n])
        self.n_offset = [self.d_obj.matrix_world.inverted() * i for i in tempN]
        self.w_offset = [self.d_obj.matrix_world.inverted() * i for i in tempW]

    def __GetIndexForOffsetSolidify(self, context):
        for i in self.i_offset:
            for j in self.d_obj.data.vertices:
                if context.active_object.matrix_world * self.d_obj.data.vertices[i].co == context.active_object.matrix_world *j.co.copy() and j.index != i:
                    self.i_offset2.append(j.index)

    def __CreateSolidifityModifier(self, context):
        for i in self.d_obj.modifiers:
            self.d_obj.modifiers.remove(i)

        self.d_obj.modifiers.new('Solidify', 'SOLIDIFY')
        self.d_obj.modifiers[0].use_even_offset = True
        self.d_obj.modifiers[0].use_quality_normals = True
        self.d_obj.modifiers[0].thickness = 0.1
        #self.d_obj.modifiers[0].use_flip_normals = True

    def __GetNormal(self, context):
        joint_normal = Vector((0.0,0.0,0.0))
        for i in self.d_obj.data.polygons:
            joint_normal += i.normal.copy()
        #print(joint_normal)
        return joint_normal

    def SetValSolidifity(self, context, value, bool, snap=False, SV=None):
        if snap:
            self.d_obj.modifiers[0].thickness = value
            self.__Swap(context, bool)
            if value > 0:
                self.d_obj.modifiers[0].thickness = SV
            else:
                self.d_obj.modifiers[0].thickness = SV * -1

        else:
            self.d_obj.modifiers[0].thickness = value
            self.__Swap(context, bool)
        self.S_val = self.d_obj.modifiers[0].thickness

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
            self.i_For_Comp_Axis.append((self.d_obj.data.vertices[x].co.copy()) - self.d_obj.data.vertices[self.vertx_for_move[0]].co.copy())
        for x in self.vertx_for_move:
            self.i_For_Comp_Axis2.append((self.d_obj.data.vertices[x].co.copy() - self.d_obj.data.vertices[self.vertx_for_move[0]].co.copy()))

        for i in self.vertx_for_move:
            self.save_Coord.append(self.d_obj.data.vertices[i].co.copy())
        self.__GetIndexForOffsetSolidify(context)
        bpy.context.scene.objects.active = a

        kd = kdtree.KDTree(len(self.vertx_for_move))

        for i in self.vertx_for_move:
            kd.insert(self.d_obj.data.vertices[i].co.copy(), i)
        kd.balance()
        self.KDA = kd

    def Move(self, context, event, loc, bool, axis, snap=False):
        if not self.constrain_axis and type(axis) != int:
            if snap:
                self.__Snap_S_Mode(context, event, bool, loc)
            else:
                self.SetValSolidifity(context, loc, bool)
        elif type(axis) == int and not self.constrain_axis:
            self.__SetMeshForAxisConstrain(context)
            self.constrain_axis = True
            self.__setMove(context, loc, axis, bool)

        else:
            if snap:# and self.constrain_axis:
                self.__SnapAxis(context, event, axis, loc, bool)
                return 0
            else:
                print('asdfsadfasfasdfsad')
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
        def P_Norm(self, context, bool, axis, loc):
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
        def N_Norm(self, context, bool, axis, loc):
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
        if self.Normal[axis] > 0.0:
           P_Norm(self, context, bool, axis, loc)
        else:
            N_Norm(self, context, bool, axis, loc)

    def __KDTree(self, mesh):
        size = len(mesh.vertices)
        kd = kdtree.KDTree(size)
        for i, vtx in enumerate(mesh.vertices):
            kd.insert(vtx.co, i)
        kd.balance()
        return kd

    def __SnapAxis(self, context, event, axis, Mloc, bool):
        c = context.scene.cursor_location
        try:
            RayCast(self, context, event, ray_max=1000.0)
        except:
            context.scene.cursor_location = c
            return 0
        kuda = self.d_obj.matrix_local.inverted() * context.scene.cursor_location
        #axis = 0

        # kd = kdtree.KDTree(len(self.vertx_for_move))
        #
        #
        # for i in self.vertx_for_move:
        #     kd.insert(self.d_obj.data.vertices[i].co.copy(), i)
        # kd.balance()
        sh, I, P = self.KDA.find(kuda)

        newV = self.vertx_for_move[I:] + self.vertx_for_move[:I]

        newP = self.i_For_Comp_Axis[I-1:] + self.i_For_Comp_Axis[:I-1]

        t = 0
        for j, i in enumerate(newV):
            if t == 0:
                self.d_obj.data.vertices[i].co[axis] = kuda[axis]
                t+=1
            else:
                x = newP[j-1]
                self.d_obj.data.vertices[i].co[axis] = kuda[axis] + x[axis]
        self.__SwapBool(context, bool, axis, kuda)


    def __Snap_S_Mode(self,context, event, bool, loc):
        try:
            RayCast(self, context, event, ray_max=1000.0)
        except:
            self.SetValSolidifity(context, loc, bool)# snap=True, SV=dist)
            return 0

        v2 = self.d_obj.matrix_local.inverted() * context.scene.cursor_location
        v1, I, P = self.KDT.find(v2)

        dvec = v2 - v1

        dnormal = np.dot(dvec, self.Normal)

        v2 = v1 + Vector(dnormal * self.Normal)

        locx = v1[0] - v2[0]
        locy = v1[1] - v2[1]
        locz = v1[2] - v2[2]

        dist = sqrt((locx) ** 2 + (locy) ** 2 + (locz) ** 2)

        print(dist)

        self.SetValSolidifity(context, loc, bool, snap=True, SV=dist)

class Util:
    def __init__(self, context, event, obj, modifer, auto_merge):
        self.starMousePos = self.__StarPosMouse(context, event, obj) + 0.001
        self.starMousePosForAxis = False
        self.modifier = modifer
        self.auto_snap = bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge
        bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = auto_merge
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
            normal =obj.matrix_local * obj.data.polygons[0].normal
            loc = ((normal * -1) * (obj.matrix_local * loc))
            return loc
        elif mode:
            return loc

    def EventMouseNormal(self, context, event, obj, mode=False, axis = False):
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)
        if type(mode) != int:
            normal =obj.matrix_local * obj.data.polygons[0].normal
            loc = ((((normal * -1) * (obj.matrix_local * loc)) - self.starMousePos) / self.Zoom(context))
            loc *= 4
            return loc
        elif not self.starMousePosForAxis:
            self.starMousePos = self.__StarPosMouse(context, event, obj, mode=True)
        elif isinstance(axis, int):
            return loc - self.starMousePos[axis]
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


    def Finish(self, context, bool_index, m_obj, d_obj, coord, index, bevel=False):
        face = self.__Intersect(context, coord, index, m_obj, d_obj, bool_index)
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
        if bevel:
            bpy.ops.mesh.edges_select_sharp()
            bpy.ops.transform.edge_bevelweight(value=1)
            bpy.ops.mesh.select_all(action='DESELECT')
        for i in face:
            m_obj.data.polygons[i].select = True

    def Cancel(self, context, bool_index, m_obj, d_obj):
        m_obj.modifiers.remove(m_obj.modifiers[bool_index])
        bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = self.auto_snap
        m_obj.show_wire = self.show_wire
        m_obj.show_all_edges = self.show_all_edges
        bpy.context.scene.objects.unlink(d_obj)
        bpy.data.objects.remove(d_obj)
        self.__RetModifier(m_obj)
        bpy.ops.object.mode_set(mode='EDIT')

    def __Intersect(self, context, coord, index, objA, objB, bool_index):
        bpy.ops.object.modifier_apply(modifier=objA.modifiers[bool_index].name)
        for j, i in enumerate(index):
                objB.data.vertices[i].co = coord[j]

        vertTemp = []
        for i in objA.data.vertices:
            countSel = 0
            for j in objB.data.vertices:
                if objA.matrix_world * i.co == objB.matrix_world * j.co:
                    #i.select = True
                    countSel +=1
                    vertTemp.append(i.index)
            if countSel == len(objB.data.vertices):
                break
        face_index = []
        for i in objA.data.polygons:
            t = list(set(i.vertices) & set(vertTemp))
            if len(t) >= 3:
                #i.select
                face_index.append(i.index)
        print('vert - ',vertTemp)
        print('face - ',face_index)
        return face_index






class DestructiveExtrude(bpy.types.Operator):
    bl_idname = "mesh.destructive_extrude"
    bl_label = "Destructive Extrude"
    bl_options = {"REGISTER", "UNDO", "GRAB_CURSOR", "BLOCKING"}
    axis = False
    key = False
    @classmethod
    def poll(cls, context):
        return (context.mode == "EDIT_MESH")





    def modal(self, context, event):
        #print(str(self.d_obj.S_val))
        def draw(self, context, event, val):
            if isinstance(val, float):
                n = 4
                template = '{:.' + str(n) + 'f}'
                val =(template.format(val))

            d = "Distance = (" + str(val) + "), "

            if isinstance(self.axis, bool):
                mv = "Normal "
            else:
                if self.axis == 0:
                    mv = "Axis along X,"
                elif self.axis == 1:
                    mv = "Axis along Y,"
                elif self.axis == 2:
                    mv = "Axis along Z,"

            m = "Mode: " + mv + ", "

            if isinstance(self.axis, bool):
                hm = "Press axis: (X, Y, Z), "
            else:
                hm = "Press axis: (X, Y, Z), "

            if event.ctrl:
                s = "Snapping (CTRL) = ON"
            else:
                s = "Snapping (CTRL) = OFF"

            return d + m + hm + s

        if event.unicode in self.enter or not isinstance(self.key, bool):
            s = ['+', '-', '*', '/', '.', ',']
            if isinstance(self.key, bool):
                self.key = ''

            if event.unicode == '.' or event.unicode == ',':
                ok = False
                if self.key != '':
                    if '.' in self.key:
                        for j, i in enumerate(reversed(self.key)):
                            print(i)
                            if i == s[0] or i == s[1] or i == s[2] or i == s[3]:
                                if not self.key[j+1].isdigit():
                                    self.key += '0.'
                                else:
                                    self.key += '.'
                                break
                            elif i == '.':
                                break

                else:
                    self.key += '0.'
            else:
                self.key += event.unicode


            if event.type == 'BACK_SPACE':
                temp = self.key[:-1]
                self.key = temp

                if self.key[:-1] in s:
                    temp = self.key[:-1]
                    self.key = temp

            if self.key != '':
                if self.key[-1].isdigit():
                    self.d_obj.Move(context, event, float(eval(self.key)), self.m_obj.index_bool_modifier, axis=self.axis, snap=False)

            if event.type in {'RET', 'NUMPAD_ENTER'}:
                self.v3d.Finish(context, self.m_obj.index_bool_modifier, self.m_obj.m_Obj, self.d_obj.d_obj)

            context.area.header_text_set(draw(self, context, event, self.key))
            return {'RUNNING_MODAL'}

        context.area.header_text_set(draw(self, context, event, self.d_obj.S_val))

        if event.type == 'LEFTMOUSE':
            self.v3d.Finish(context, self.m_obj.index_bool_modifier, self.m_obj.m_Obj, self.d_obj.d_obj, self.d_obj.n_offset, self.d_obj.i_offset)
            print('super sisi')
            context.area.header_text_set()
            return {'FINISHED'}

        if event.type == 'Q':
            context.area.header_text_set()
            return {'FINISHED'}

        if event.ctrl:
            self.d_obj.Move(context, event, self.v3d.EventMouseNormal(context, event, self.d_obj.d_obj, mode=self.axis),self.m_obj.index_bool_modifier, axis=self.axis, snap=True)
            return {'RUNNING_MODAL'}



        if event.type == 'RIFGTMOUSE':
            self.v3d.Cancel(context, self.m_obj.index_bool_modifier, self.m_obj.m_Obj, self.d_obj.d_obj)
            context.area.header_text_set()
            return {'FINISHED'}


        if event.type == 'SPACE':
            self.v3d.Finish(context, self.m_obj.index_bool_modifier, self.m_obj.m_Obj, self.d_obj.d_obj, bevel=True)
            context.area.header_text_set()


        if event.type == 'X':
            self.axis = 0
        if event.type == 'Y':
            self.axis = 1
        if event.type == 'Z':
            self.axis = 2
            print(self.axis)

        if event.type == 'MOUSEMOVE':
            # self.d_obj.SetValSolidifity(context, self.v3d.EventMouseNormal(context, event, self.d_obj.d_obj), self.m_obj.index_bool_modifier)
            self.d_obj.Move(context, event,
            self.v3d.EventMouseNormal(context, event, self.d_obj.d_obj, mode=self.axis, axis=self.axis),
            self.m_obj.index_bool_modifier, axis=self.axis, snap=False)

        



        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            auto_merge = bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = False
            self.m_obj = M_Object(context)
            self.d_obj = D_Object(context, self.m_obj.n_offset, self.m_obj.w_offset)
            self.v3d = Util(context, event, self.d_obj.d_obj, self.m_obj.u_modifier, auto_merge)
            self.enter = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '+', '*', '/', '.', ',']

            #args = (self, context)
            #self._handle = bpy.types.SpaceView          3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')

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
