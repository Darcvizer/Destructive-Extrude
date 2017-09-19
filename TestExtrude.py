# ?????? ?????????
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
import bpy
import bmesh
from bpy.types import Operator
from math import sin, cos, pi, radians
from mathutils import Vector
from time import perf_counter

from bpy.props import (
    BoolProperty,
    FloatProperty,
    EnumProperty,
)

# import win32api
bl_info = {
    "name": "Destructive Extrude :)",
    "location": "View3D > Add > Mesh > Destructive Extrude,",
    "description": "Extrude how SketchUp.",
    "author": "Vladislav Kindushov",
    "version": (0, 8, 9),
    "blender": (2, 7, 8),
    "category": "Mesh",
}


def MeshForAxisConstrain(self, context):
    a = bpy.context.active_object
    bpy.context.scene.objects.active = self.var[1]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.scene.objects.active = a

    for i in self.var[1].modifiers:
        if i.type == "SOLIDIFY":
            self.var[1].modifiers.remove(i)
    self.Normal = Vector((0.0, 0.0, 0.0))
    z = True
    for i in bpy.context.active_object.data.polygons:
        if z:
            self.Normal = i.normal
            z = False
            break
        else:
            self.Normal *= i.normal

    bm = bmesh.new()
    bm.from_mesh(self.var[1].data)
    vert = [i.index for i in bm.verts]
    bmesh.ops.solidify(bm, geom=[f for f in bm.faces], thickness=0)
    # bmesh.ops.reverse_faces(bm, faces = [f for f in bm.faces])
    vert = [i.index for i in bm.verts if i.index not in vert]
    compCoor = []
    bm.verts.ensure_lookup_table()
    saveCoord = []

    for x in vert:
        saveCoord.append((bm.verts[x].co))

    for x in vert[1:]:
        compCoor.append((bm.verts[x].co - bm.verts[vert[0]].co))

    for i in vert:
        self.firstCoordForAxisConstrain.append(bm.verts[i].co.copy())
    bm.verts.ensure_lookup_table()
    bm.to_mesh(self.var[1].data)
    self.var[1].data.update()
    bm.free()
    bpy.data.objects.remove(bpy.data.objects['Negative'])
    bpy.context.scene.objects.active = self.var[0]
    return vert, compCoor, saveCoord

def RayCast(self, context, event, ray_max=1000.0):
    """Run this function on left mouse, execute the ray cast"""
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

    # get the ray from the viewport and mouse
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    ray_target = ray_origin + view_vector

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

    run(best_obj, best_matrix, best_face, best_hit)

def firstD(self, context):
    bm = bmesh.from_edit_mesh(bpy.context.active_object.data)
    ferst_verts = [v.index for v in bm.verts if v.select]
    sf = [v.index for v in bm.faces if v.select]

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})
    bpy.ops.mesh.separate(type='SELECTED')

    bpy.ops.object.mode_set(mode='OBJECT')

    sel_obj = bpy.context.selected_objects
    object_B = None
    object_B_Name = None
    for i in sel_obj:
        object_B_Name = i.name
        object_B = i
        break

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(bpy.context.active_object.data)
    bm.faces.ensure_lookup_table()
    for i in sf:
        bm.faces[i].select = True
    return object_B

# bmesh.update_edit_mesh(obj.data)
# bm.free()

def Duplicate(self, context, act_obj):
    """Create 2 copy selection. the firs orygnal posisiton. the second modifi"""
    main_select_obj = bpy.context.selected_objects
    object_B = firstD(self, context)
    save_pos = {}
    angle = 0.0
    distance = -0.00002

    bm = bmesh.from_edit_mesh(act_obj.data)
    ferst_verts = [v.index for v in bm.verts if v.select]
    face = [v.index for v in bm.faces if v.select]
    f = [f for f in bm.faces if f.select]
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for i in ferst_verts:
        if bm.verts[i].select:
            save_pos[i] = bm.verts[i].co.copy()

    bpy.ops.mesh.select_all(action='DESELECT')
    edgeForLen = []

    for i in f:  # ?? ??????
        for edge in i.edges:  # ?? ?????? ??? ?????
            angle = edge.calc_face_angle_signed()  # ??????? ????
            angle = degrees(angle)  # ????????? ???? ? ???????????? ???
            if abs(angle) > 29.999:  # ???? 90 ????????1
                edge.select = True
                edgeForLen.append(edge.index)
    bmesh.update_edit_mesh(act_obj.data)
    #bm.free()
    if len(edgeForLen) <= 1:
        for i in f:  # ?? ??????
            for edge in i.edges:  # ?? ?????? ??? ?????
                angle = edge.calc_face_angle_signed()  # ??????? ????
                angle = degrees(angle)  # ????????? ???? ? ???????????? ???
                if abs(angle) > 29.999:  # ???? 90 ????????1
                    vert1, vert2 = edge.verts  # ??????? ??????? ???? ????????
                    if not vert1.index in save_pos:
                        save_pos[vert1.index] = vert1.co.copy()
                    if not vert2.index in save_pos:
                        save_pos[vert2.index] = vert2.co.copy()
                    # ??????? ???? ?????, ??? ?????? ??????? ?????, ???? ????? ???? ??? ??? ??????
                    link1 = vert1.link_edges
                    link2 = vert2.link_edges
                    for l in link1:
                        for e in i.edges:
                            if l == e and not l == edge:
                                V1, V2 = l.verts
                                vec = V2.co
                                # if V1 == vert1:
                                vec = V2.co - V1.co
                                if V2 == vert1:
                                    vec = V1.co - V2.co
                                vec.normalize()
                                vert1.co += vec * distance
                    for l in link2:
                        for e in i.edges:
                            if l == e and not l == edge:
                                V1, V2 = l.verts
                                vec = V2.co
                                if V1 == vert2:
                                    vec = V2.co - V1.co
                                if V2 == vert2:
                                    vec = V1.co - V2.co
                                vec.normalize()
                                vert2.co += vec * distance

    else:
        #bm.verts.ensure_lookup_table()
        #bm.faces.ensure_lookup_table()
        bm = bmesh.from_edit_mesh(act_obj.data)
        bm.edges.ensure_lookup_table()
        L = bm.edges[edgeForLen[0]].calc_length()
        bpy.ops.mesh.offset_edges(geometry_mode='move', width=0.00002)

        bm = bmesh.from_edit_mesh(act_obj.data)
        bm.edges.ensure_lookup_table()
        if L > bm.edges[edgeForLen[0]].calc_length():
            bpy.ops.mesh.offset_edges(geometry_mode='move', width=-0.00004)




    bm = bmesh.from_edit_mesh(act_obj.data)

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for i in face:
        bm.faces[i].select = True

    bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode": 1})

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for key, value in save_pos.items():
        bm.verts[key].co = value

    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')

    sel_obj = bpy.context.selected_objects
    object_C = None
    object_C_Name = None
    for i in sel_obj:
        object_C_Name = i.name
        object_C = i
        break

    for i in main_select_obj:
        i.select = True

    return object_B, object_C

def HideModeifiers(context, obj):
    """Hide modifiers"""
    desable_modifier = []
    for i in obj.modifiers:
        if i.show_viewport == True:
            desable_modifier.append(i)
        i.show_viewport = False
    return desable_modifier

def GetB_obj(self, context):
    '''Get sepatare object'''
    sel_obj = bpy.context.selected_objects
    object_B = None
    object_B_Name = None
    for i in sel_obj:
        object_B_Name = i.name
        object_B = i
        break

    return object_B

def SetupObjB(self, context, object_B_Name, f=False):
    """Setup objectB """
    context.scene.objects.active = bpy.data.objects[object_B_Name]

    bpy.ops.object.modifier_add(type='SOLIDIFY')

    for i in bpy.data.objects[object_B_Name].modifiers:
        if i.type == 'SOLIDIFY':
            i.use_even_offset = True
    bpy.data.objects[object_B_Name].draw_type = 'WIRE'

def SetupBoolean(self, context, obj_A, obj_B):
    """Add modifier boolean for the first object"""
    context.scene.objects.active = bpy.data.objects[obj_A.name]
    bpy.ops.object.modifier_add(type='BOOLEAN')
    for i in obj_A.modifiers:
        if i.type == 'BOOLEAN' and i.show_viewport:
            i.operation = 'DIFFERENCE'
            i.object = obj_B
            i.solver = 'CARVE'

def RemoveModifier(self, context, obj):
    """Remove modifier"""
    context.scene.objects.active = bpy.data.objects[obj.name]
    for i in obj.modifiers:
        bpy.ops.object.modifier_remove(modifier=i.name)

def CrateKD(context, obj):
    """Create KDTree"""
    size = len(obj.data.vertices)
    kd = kdtree.KDTree(size)
    for i, vtx in enumerate(obj.data.vertices):
        kd.insert(vtx.co, i)
    kd.balance()
    return kd

def Setup(self, context):
    """Main function for invoke"""

    modifiers_list = None
    select_objects = None
    object_A = context.active_object
    object_Negative = None
    object_Positive = None
    kd = None

    object_Negative, object_Positive = Duplicate(self, context, object_A)
    modifiers_list = HideModeifiers(context, object_A)
    SetupBoolean(self, context, object_A, object_Negative)
    RemoveModifier(self, context, object_Negative)
    RemoveModifier(self, context, object_Positive)
    SetupObjB(self, context, object_Negative.name)
    SetupObjB(self, context, object_Positive.name)
    kd = CrateKD(context, object_Negative)
    object_Negative.name = 'Negative'
    object_Positive.name = 'Positive'
    ver = []
    ver.append(object_A)
    ver.append(object_Positive)
    ver.append(object_Negative)
    ver.append(modifiers_list)
    ver.append(select_objects)
    ver.append(kd)
    return ver

def StarPosMouse(self, context, event):
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)

    normal = self.var[2].data.polygons[0].normal
    loc = ((normal * -1) * loc)
    return loc

def SwitchMesh(context, var):
    """Switch in a object boolean source a mesh"""
    try:
        if var[1].modifiers['Solidify'].thickness < 0.0:
            for i in var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    if i.object != var[2]:
                        i.object = var[2]
                        i.operation = 'UNION'

                        # bpy.data.objects[var[2].name].hide = True
                        return True

        elif var[1].modifiers['Solidify'].thickness > 0.0:
            for i in var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    if i.object != var[1]:
                        i.object = var[1]
                        i.operation = 'DIFFERENCE'
                        # bpy.data.objects[var[1].name].hide = True
                        # bpy.data.objects[var[2].name].hide = True

                        return False


        elif var[1].modifiers['Solidify'].thickness == 0.0:
            for i in var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    i.object = None

    except:
        return False

def AxsisConstrainEventMouse(self, context, event):
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)

    # area = Zoom(self, context)
    # loc = loc - self.start_mouse / area
    return loc

def EventMouse(self, context, event, obj):
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)

    normal = obj[2].data.polygons[0].normal
    area = Zoom(self, context)
    loc = (((normal * -1) * loc) - self.start_mouse) / area

    loc *= 4

    self.var[1].modifiers['Solidify'].thickness = loc
    self.var[2].modifiers['Solidify'].thickness = loc

    SwitchMesh(context, obj)

    if self.temp_key:
        self.key = str(loc)

def EventCtrl(self, context, event, var):
    try:
        RayCast(self, context, event)
    except:
        pass

    temp = SwitchMesh(context, var)
    mat = None
    if temp:
        mat = var[1].matrix_world
        mat = mat.inverted()
    else:
        mat = var[2].matrix_world
        mat = mat.inverted()

    dist = var[-1].find(mat * context.scene.cursor_location)

    pos1 = None
    pos2 = None

    if temp:
        pos1 = var[1].data.vertices[dist[1]].co
        pos2 = mat * context.scene.cursor_location
    else:
        pos1 = var[2].data.vertices[dist[1]].co
        pos2 = mat * context.scene.cursor_location

    bm = bmesh.new()
    me = bpy.data.meshes.new("Mesh")
    bm.to_mesh(me)
    A = bm.verts.new(pos1)
    bm.to_mesh(me)
    B = bm.verts.new(pos2)
    bm.to_mesh(me)
    V = bm.edges.new((A, B))
    bm.to_mesh(me)

    scene = bpy.context.scene
    obj = bpy.data.objects.new("Length", me)
    scene.objects.link(obj)

    bm.to_mesh(me)

    bm.edges.ensure_lookup_table()
    lenn = bm.edges[0].calc_length()

    bm.to_mesh(me)
    bm.free()

    bpy.context.scene.objects.unlink(obj)
    bpy.data.objects.remove(obj)

    for i in var[0].modifiers:
        if i.show_viewport == True and i.type == 'BOOLEAN':
            if i.object == var[2]:

                var[2].modifiers['Solidify'].thickness = lenn * -1
            else:
                var[1].modifiers['Solidify'].thickness = lenn

    SwitchMesh(context, var)

    if self.temp_key:
        self.key = str(lenn)

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

def Finish(self, context, f=False):
    context.scene.objects.active = bpy.data.objects[self.var[0].name]
    for i in self.var[0].modifiers:
        if i.show_viewport == True and i.type == 'BOOLEAN':
            a = i.name
            bpy.ops.object.modifier_apply(modifier=i.name)

        # bpy.data.objects[self.var[0].name].modifiers.apply(i)

    bpy.data.objects[self.var[0].name].show_wire = self.show_wire
    bpy.data.objects[self.var[0].name].show_all_edges = self.show_all_edges

    bpy.context.scene.objects.unlink(self.var[1])
    bpy.data.objects.remove(self.var[1])
    try:
        bpy.context.scene.objects.unlink(self.var[2])
        bpy.data.objects.remove(self.var[2])
    except:
        pass
    for i in self.var[0].modifiers:
        if i in self.var[3]:
            i.show_viewport = True

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = self.auto_snap
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.select_all(action='DESELECT')

    if f:
        bpy.ops.mesh.edges_select_sharp()
        bpy.ops.transform.edge_bevelweight(value=1)
        bpy.ops.mesh.select_all(action='DESELECT')

def Cansl(self, context):
    context.scene.objects.active = bpy.data.objects[self.var[0].name]
    for i in self.var[0].modifiers:
        if i.show_viewport == True and i.type == 'BOOLEAN':
            bpy.data.objects[self.var[0].name].modifiers.remove(i)

    bpy.data.objects[self.var[0].name].show_wire = self.show_wire
    bpy.data.objects[self.var[0].name].show_all_edges = self.show_all_edges

    bpy.context.scene.objects.unlink(self.var[1])
    bpy.data.objects.remove(self.var[1])
    try:
        bpy.context.scene.objects.unlink(self.var[2])
        bpy.data.objects.remove(self.var[2])
    except:
        pass

    for i in self.var[0].modifiers:
        if i in self.var[3]:
            i.show_viewport = True

    bpy.ops.object.mode_set(mode='EDIT')

def Zoom(self, context):
    ar = None
    for i in bpy.context.window.screen.areas:
        if i.type == 'VIEW_3D': ar = i

    ar = ar.spaces[0].region_3d.view_distance
    # print(ar)
    return ar

def SwitchModeInAxisConstrain(self, context, event, loc, axis):
    if axis == 'x':
        axis = 0
    elif axis == 'y':
        axis = 1
    elif axis == 'z':
        axis = 2

    if self.Normal[2] > 0.0:
        if loc[axis] < self.firstCoordForAxisConstrain[0][axis]:
            for i in self.var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    if i.operation != 'DIFFERENCE':
                        i.operation = 'DIFFERENCE'
                        bm = bmesh.new()
                        bm.from_mesh(self.var[1].data)
                        for i in bm.faces:
                            i.normal_flip()
                        bm.to_mesh(self.var[1].data)
                        self.var[1].data.update()
                        bm.free()
        else:
            for i in self.var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    if i.operation != 'UNION':
                        i.operation = 'UNION'
                        bm = bmesh.new()
                        bm.from_mesh(self.var[1].data)
                        for i in bm.faces:
                            i.normal_flip()
                        bm.to_mesh(self.var[1].data)
                        self.var[1].data.update()
                        bm.free()
    else:
        if loc[axis] > self.firstCoordForAxisConstrain[0][axis]:
            for i in self.var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    if i.operation != 'DIFFERENCE':
                        i.operation = 'DIFFERENCE'
                        bm = bmesh.new()
                        bm.from_mesh(self.var[1].data)
                        for i in bm.faces:
                            i.normal_flip()
                        bm.to_mesh(self.var[1].data)
                        self.var[1].data.update()
                        bm.free()
        else:
            for i in self.var[0].modifiers:
                if i.show_viewport == True and i.type == 'BOOLEAN':
                    if i.operation != 'UNION':
                        i.operation = 'UNION'
                        bm = bmesh.new()
                        bm.from_mesh(self.var[1].data)
                        for i in bm.faces:
                            i.normal_flip()
                        bm.to_mesh(self.var[1].data)
                        self.var[1].data.update()
                        bm.free()



def Offset(self, context, event, axis):
    if axis == 'x':
        loc = AxsisConstrainEventMouse(self, context, event)
        bm = bmesh.new()
        bm.from_mesh(self.var[1].data)
        bm.verts.ensure_lookup_table()
        bestMatrix = self.var[0].matrix_world
        for j, i in enumerate(self.vert):
            if i == self.vert[0]:
                bm.verts[i].co[0] = loc[0]# * bestMatrix
            else:
                a = self.compCoor[j - 1]
                bm.verts[i].co[0] = loc[0] + a[0]

        bm.to_mesh(self.var[1].data)
        self.var[1].data.update()
        bm.free()

    elif axis == 'y':
        loc = AxsisConstrainEventMouse(self, context, event)
        bm = bmesh.new()
        bm.from_mesh(self.var[1].data)
        bm.verts.ensure_lookup_table()

        for j, i in enumerate(self.vert):
            if i == self.vert[0]:
                bm.verts[i].co[1] = loc[1]
            else:
                a = self.compCoor[j - 1]
                bm.verts[i].co[1] = loc[1] + a[1]

        bm.to_mesh(self.var[1].data)
        self.var[1].data.update()
        bm.free()

    elif axis == 'z':
        loc = AxsisConstrainEventMouse(self, context, event)
        bm = bmesh.new()
        bm.from_mesh(self.var[1].data)
        bm.verts.ensure_lookup_table()

        for j, i in enumerate(self.vert):
            if i == self.vert[0]:
                bm.verts[i].co[2] = loc[2]
            else:
                a = self.compCoor[j - 1]
                bm.verts[i].co[2] = loc[2] + a[2]

        bm.to_mesh(self.var[1].data)
        self.var[1].data.update()
        bm.free()
    try:
        SwitchModeInAxisConstrain(self, context, event, loc, axis)
    except:
        pass

# --------------------------------------------------------------------------------#

class DestructiveExtrude(bpy.types.Operator):
    bl_idname = "mesh.destructive_extrude"
    bl_label = "Destructive Extrude"
    bl_options = {"REGISTER", "UNDO", "GRAB_CURSOR", "BLOCKING"}

    first_mouse_x = IntProperty()
    first_value = FloatProperty()
    X = False
    Y = False
    Z = False
    vert = None
    compCoor = None
    axis_constrain = True
    firstCoordForAxisConstrain = []
    modOffset = False
    Normal = None
    saveCoord = []
    matrix = bpy.context.active_object.matrix_world.copy()
    @classmethod
    def poll(cls, context):
        return (context.mode == "EDIT_MESH")  # and (context.tool_settings.mesh_select_mode == (False, False, True))

    # return (context.active_object is not None) and (context.mode == "EDIT_MESH")

    # -----------------------------------------------------------------------------------------------
    def modal(self, context, event):
        try:
            if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'LEFTMOUSE', 'RIGHTMOUSE'} and (
                        event.alt or event.shift):
                return {'PASS_THROUGH'}
            if self.X == False and self.Y == False and self.Z == False:
                SwitchMesh(context, self.var)
                if event.type == 'MOUSEMOVE':
                    EventMouse(self, context, event, self.var)
                    context.area.tag_redraw()

                if event.ctrl:
                    EventCtrl(self, context, event, self.var)
                    context.area.tag_redraw()

                if event.unicode in self.enter:
                    if self.draw == False:
                        self.temp_key = False
                        self.draw = True
                        temp = ''
                        self.key = temp
                    if event.unicode == ',':
                        self.key += '.'
                    else:
                        self.key += event.unicode
                        context.area.tag_redraw()

                if event.type == 'BACK_SPACE':
                    temp = self.key[:-1]
                    self.key = temp
                    s = ['+', '-', '*', '/', '.']
                    if self.key[:-1] in s:
                        temp = self.key[:-1]
                        self.key = temp
                    context.area.tag_redraw()

                if event.type in {'RET', 'NUMPAD_ENTER'}:
                    if not self.temp_key:
                        self.var[1].modifiers['Solidify'].thickness = float(eval(self.key))
                        self.var[2].modifiers['Solidify'].thickness = float(eval(self.key))
                        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                        context.area.tag_redraw()
                        context.area.header_text_set()
                        Finish(self, context)
                        return {'FINISHED'}

                if event.type == 'LEFTMOUSE':
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                    context.area.tag_redraw()
                    context.area.header_text_set()
                    Finish(self, context)
                    return {'FINISHED'}

                if event.type == 'SPACE':
                    self.var[1].modifiers['Solidify'].thickness = float(eval(self.key))
                    self.var[2].modifiers['Solidify'].thickness = float(eval(self.key))
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                    context.area.tag_redraw()
                    context.area.header_text_set()
                    Finish(self, context, True)
                    return {'FINISHED'}

                if event.type == 'RIGHTMOUSE':
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                    context.area.tag_redraw()
                    context.area.header_text_set()
                    Cansl(self, context)
                    return {'CANCELLED'}

                if event.type in {'ESC'}:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                    context.area.tag_redraw()
                    context.area.header_text_set()
                    Cansl(self, context)
                    return {'CANCELLED'}

            # if (self.X == 'x' or self.Y == 'y' or self.Z == 'z'):# and (self.axis_constrain):
            #     self.modOffset = True
            #     if event.type == 'X':
            #         self.X = 'x'
            #         self.Y = False
            #         self.Z = False
            #
            #         bm = bmesh.new()
            #         bm.from_mesh(self.var[1].data)
            #         bm.verts.ensure_lookup_table()
            #         for j, i in enumerate(self.vert):
            #             bm.verts[i].co = self.firstCoordForAxisConstrain[j]
            #         bm.to_mesh(self.var[1].data)
            #         self.var[1].data.update()
            #         bm.free()
            #
            #     elif event.type == 'Y':
            #         self.X = False
            #         self.Y = 'y'
            #         self.Z = False
            #         bm = bmesh.new()
            #         bm.from_mesh(self.var[1].data)
            #         bm.verts.ensure_lookup_table()
            #         for j, i in enumerate(self.vert):
            #             bm.verts[i].co = self.firstCoordForAxisConstrain[j]
            #         bm.to_mesh(self.var[1].data)
            #         self.var[1].data.update()
            #         bm.free()
            #
            #     elif event.type == 'Z':
            #         self.X = False
            #         self.Y = False
            #         self.Z = 'z'
            #         bm = bmesh.new()
            #         bm.from_mesh(self.var[1].data)
            #         bm.verts.ensure_lookup_table()
            #         for j, i in enumerate(self.vert):
            #             bm.verts[i].co = self.firstCoordForAxisConstrain[j]
            #         bm.to_mesh(self.var[1].data)
            #         self.var[1].data.update()
            #         bm.free()


            if event.type == 'X':
                if event.type == 'LEFTMOUSE':
                    Finish(self, context)
                    return {'FINISHED'}

                #if self.X == False:
                self.X = 'x'
                self.Y = False
                self.Z = False

                if self.axis_constrain:
                    self.vert, self.compCoor, self.saveCoord  = MeshForAxisConstrain(self, context)
                    self.axis_constrain = False
                if self.axis_constrain == False:
                    bm = bmesh.new()
                    bm.from_mesh(self.var[1].data)
                    bm.verts.ensure_lookup_table()
                    for j, i in enumerate(self.vert):
                        bm.verts[i].co = self.firstCoordForAxisConstrain[j]

                    bm.to_mesh(self.var[1].data)
                    self.var[1].data.update()
                    bm.free()

            if event.type == 'Y':# or self.Y == 'y':
                if event.type == 'LEFTMOUSE':
                    Finish(self, context)
                    return {'FINISHED'}

                #if self.Y == False:
                self.X = False
                self.Y = 'y'
                self.Z = False
                if self.axis_constrain:
                    self.vert, self.compCoor, self.saveCoord = MeshForAxisConstrain(self, context)
                    self.axis_constrain = False
                if self.axis_constrain == False:
                    bm = bmesh.new()
                    bm.from_mesh(self.var[1].data)
                    bm.verts.ensure_lookup_table()
                    for j, i in enumerate(self.vert):
                        bm.verts[i].co = self.firstCoordForAxisConstrain[j]

                    bm.to_mesh(self.var[1].data)
                    self.var[1].data.update()
                    bm.free()


            if event.type == 'Z':# or self.Z == 'z':
                if event.type == 'LEFTMOUSE':
                    Finish(self, context)
                    return {'FINISHED'}


                self.X = False
                self.Y = False
                self.Z = 'z'

                if self.axis_constrain:
                    self.vert, self.compCoor, self.saveCoord = MeshForAxisConstrain(self, context)
                    self.axis_constrain = False
                if self.axis_constrain == False:
                    bm = bmesh.new()
                    bm.from_mesh(self.var[1].data)
                    bm.verts.ensure_lookup_table()
                    for j, i in enumerate(self.vert):
                        bm.verts[i].co = self.firstCoordForAxisConstrain[j]

                    bm.to_mesh(self.var[1].data)
                    self.var[1].data.update()
                    bm.free()

            if (self.X == 'x' or self.Y == 'y' or self.Z == 'z'):
                if event.type == 'LEFTMOUSE':
                    Finish(self, context)
                    return {'FINISHED'}

                if self.Z == 'z':
                    Offset(self, context, event, self.Z)
                elif self.Y == 'y':
                    Offset(self, context, event, self.Y)
                elif self.X == 'x':
                    Offset(self, context, event, self.X)
        except:
            pass
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            self.auto_snap = bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge
            bpy.data.scenes['Scene'].tool_settings.use_mesh_automerge = False
            self.var = Setup(self, context)
            #print(self.var[:])
            self.show_wire = bpy.data.objects[self.var[0].name].show_wire
            self.show_all_edges = bpy.data.objects[self.var[0].name].show_all_edges

            self.enter = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '+', '*', '/', '.', ',']

            self.start_mouse = StarPosMouse(self, context, event)

            args = (self, context)
            self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
            self.key = ""
            self.temp_key = True
            self.draw = False

            bpy.data.objects[self.var[0].name].show_wire = True
            bpy.data.objects[self.var[0].name].show_all_edges = True

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "is't 3dview")
            return {'CANCELLED'}


# __________________________________________________________________________________________________________________________________________________________________
# Globals
X_UP = Vector((1.0, .0, .0))
Y_UP = Vector((.0, 1.0, .0))
Z_UP = Vector((.0, .0, 1.0))
ZERO_VEC = Vector((.0, .0, .0))
ANGLE_90 = pi / 2
ANGLE_180 = pi
ANGLE_360 = 2 * pi

# switch performance logging
ENABLE_DEBUG = False


def calc_loop_normal(verts, fallback=Z_UP):
    # Calculate normal from verts using Newell's method
    normal = ZERO_VEC.copy()

    if verts[0] is verts[-1]:
        # Perfect loop
        range_verts = range(1, len(verts))
    else:
        # Half loop
        range_verts = range(0, len(verts))

    for i in range_verts:
        v1co, v2co = verts[i - 1].co, verts[i].co
        normal.x += (v1co.y - v2co.y) * (v1co.z + v2co.z)
        normal.y += (v1co.z - v2co.z) * (v1co.x + v2co.x)
        normal.z += (v1co.x - v2co.x) * (v1co.y + v2co.y)

    if normal != ZERO_VEC:
        normal.normalize()
    else:
        normal = fallback

    return normal


def collect_edges(bm):
    set_edges_orig = set()
    for e in bm.edges:
        if e.select:
            co_faces_selected = 0
            for f in e.link_faces:
                if f.select:
                    co_faces_selected += 1
                    if co_faces_selected == 2:
                        break
            else:
                set_edges_orig.add(e)

    if not set_edges_orig:
        return None

    return set_edges_orig


def collect_loops(set_edges_orig):
    set_edges_copy = set_edges_orig.copy()

    loops = []  # [v, e, v, e, ... , e, v]
    while set_edges_copy:
        edge_start = set_edges_copy.pop()
        v_left, v_right = edge_start.verts
        lp = [v_left, edge_start, v_right]
        reverse = False
        while True:
            edge = None
            for e in v_right.link_edges:
                if e in set_edges_copy:
                    if edge:
                        # Overlap detected.
                        return None
                    edge = e
                    set_edges_copy.remove(e)
            if edge:
                v_right = edge.other_vert(v_right)
                lp.extend((edge, v_right))
                continue
            else:
                if v_right is v_left:
                    # Real loop.
                    loops.append(lp)
                    break
                elif reverse is False:
                    # Right side of half loop
                    # Reversing the loop to operate same procedure on the left side
                    lp.reverse()
                    v_right, v_left = v_left, v_right
                    reverse = True
                    continue
                else:
                    # Half loop, completed
                    loops.append(lp)
                    break
    return loops


def get_adj_ix(ix_start, vec_edges, half_loop):
    # Get adjacent edge index, skipping zero length edges
    len_edges = len(vec_edges)
    if half_loop:
        range_right = range(ix_start, len_edges)
        range_left = range(ix_start - 1, -1, -1)
    else:
        range_right = range(ix_start, ix_start + len_edges)
        range_left = range(ix_start - 1, ix_start - 1 - len_edges, -1)

    ix_right = ix_left = None
    for i in range_right:
        # Right
        i %= len_edges
        if vec_edges[i] != ZERO_VEC:
            ix_right = i
            break
    for i in range_left:
        # Left
        i %= len_edges
        if vec_edges[i] != ZERO_VEC:
            ix_left = i
            break
    if half_loop:
        # If index of one side is None, assign another index
        if ix_right is None:
            ix_right = ix_left
        if ix_left is None:
            ix_left = ix_right

    return ix_right, ix_left


def get_adj_faces(edges):
    adj_faces = []
    for e in edges:
        adj_f = None
        co_adj = 0
        for f in e.link_faces:
            # Search an adjacent face
            # Selected face has precedence
            if not f.hide and f.normal != ZERO_VEC:
                adj_f = f
                co_adj += 1
                if f.select:
                    adj_faces.append(adj_f)
                    break
        else:
            if co_adj == 1:
                adj_faces.append(adj_f)
            else:
                adj_faces.append(None)
    return adj_faces


def get_edge_rail(vert, set_edges_orig):
    co_edges = co_edges_selected = 0
    vec_inner = None
    for e in vert.link_edges:
        if (e not in set_edges_orig and
                (e.select or (co_edges_selected == 0 and not e.hide))):
            v_other = e.other_vert(vert)
            vec = v_other.co - vert.co
            if vec != ZERO_VEC:
                vec_inner = vec
                if e.select:
                    co_edges_selected += 1
                    if co_edges_selected == 2:
                        return None
                else:
                    co_edges += 1
    if co_edges_selected == 1:
        vec_inner.normalize()
        return vec_inner
    elif co_edges == 1:
        # No selected edges, one unselected edge
        vec_inner.normalize()
        return vec_inner
    else:
        return None


def get_cross_rail(vec_tan, vec_edge_r, vec_edge_l, normal_r, normal_l):
    # Cross rail is a cross vector between normal_r and normal_l
    vec_cross = normal_r.cross(normal_l)
    if vec_cross.dot(vec_tan) < .0:
        vec_cross *= -1
    cos_min = min(vec_tan.dot(vec_edge_r), vec_tan.dot(-vec_edge_l))
    cos = vec_tan.dot(vec_cross)
    if cos >= cos_min:
        vec_cross.normalize()
        return vec_cross
    else:
        return None


def move_verts(width, depth, verts, directions, geom_ex):
    if geom_ex:
        geom_s = geom_ex['side']
        verts_ex = []
        for v in verts:
            for e in v.link_edges:
                if e in geom_s:
                    verts_ex.append(e.other_vert(v))
                    break
        verts = verts_ex

    for v, (vec_width, vec_depth) in zip(verts, directions):
        v.co += width * vec_width + depth * vec_depth


def extrude_edges(bm, edges_orig):
    extruded = bmesh.ops.extrude_edge_only(bm, edges=edges_orig)['geom']
    n_edges = n_faces = len(edges_orig)
    n_verts = len(extruded) - n_edges - n_faces

    geom = dict()
    geom['verts'] = verts = set(extruded[:n_verts])
    geom['edges'] = edges = set(extruded[n_verts:n_verts + n_edges])
    geom['faces'] = set(extruded[n_verts + n_edges:])
    geom['side'] = set(e for v in verts for e in v.link_edges if e not in edges)

    return geom


def clean(bm, mode, edges_orig, geom_ex=None):
    for f in bm.faces:
        f.select = False
    if geom_ex:
        for e in geom_ex['edges']:
            e.select = True
        if mode == 'offset':
            lis_geom = list(geom_ex['side']) + list(geom_ex['faces'])
            bmesh.ops.delete(bm, geom=lis_geom, context=2)
    else:
        for e in edges_orig:
            e.select = True


def collect_mirror_planes(edit_object):
    mirror_planes = []
    eob_mat_inv = edit_object.matrix_world.inverted()
    for m in edit_object.modifiers:
        if (m.type == 'MIRROR' and m.use_mirror_merge):
            merge_limit = m.merge_threshold
            if not m.mirror_object:
                loc = ZERO_VEC
                norm_x, norm_y, norm_z = X_UP, Y_UP, Z_UP
            else:
                mirror_mat_local = eob_mat_inv * m.mirror_object.matrix_world
                loc = mirror_mat_local.to_translation()
                norm_x, norm_y, norm_z, _ = mirror_mat_local.adjugated()
                norm_x = norm_x.to_3d().normalized()
                norm_y = norm_y.to_3d().normalized()
                norm_z = norm_z.to_3d().normalized()
            if m.use_x:
                mirror_planes.append((loc, norm_x, merge_limit))
            if m.use_y:
                mirror_planes.append((loc, norm_y, merge_limit))
            if m.use_z:
                mirror_planes.append((loc, norm_z, merge_limit))
    return mirror_planes


def get_vert_mirror_pairs(set_edges_orig, mirror_planes):
    if mirror_planes:
        set_edges_copy = set_edges_orig.copy()
        vert_mirror_pairs = dict()
        for e in set_edges_orig:
            v1, v2 = e.verts
            for mp in mirror_planes:
                p_co, p_norm, mlimit = mp
                v1_dist = abs(p_norm.dot(v1.co - p_co))
                v2_dist = abs(p_norm.dot(v2.co - p_co))
                if v1_dist <= mlimit:
                    # v1 is on a mirror plane
                    vert_mirror_pairs[v1] = mp
                if v2_dist <= mlimit:
                    # v2 is on a mirror plane
                    vert_mirror_pairs[v2] = mp
                if v1_dist <= mlimit and v2_dist <= mlimit:
                    # This edge is on a mirror_plane, so should not be offsetted
                    set_edges_copy.remove(e)
        return vert_mirror_pairs, set_edges_copy
    else:
        return None, set_edges_orig


def get_mirror_rail(mirror_plane, vec_up):
    p_norm = mirror_plane[1]
    mirror_rail = vec_up.cross(p_norm)
    if mirror_rail != ZERO_VEC:
        mirror_rail.normalize()
        # Project vec_up to mirror_plane
        vec_up = vec_up - vec_up.project(p_norm)
        vec_up.normalize()
        return mirror_rail, vec_up
    else:
        return None, vec_up


def reorder_loop(verts, edges, lp_normal, adj_faces):
    for i, adj_f in enumerate(adj_faces):
        if adj_f is None:
            continue

        v1, v2 = verts[i], verts[i + 1]
        fv = tuple(adj_f.verts)
        if fv[fv.index(v1) - 1] is v2:
            # Align loop direction
            verts.reverse()
            edges.reverse()
            adj_faces.reverse()

        if lp_normal.dot(adj_f.normal) < .0:
            lp_normal *= -1
        break
    else:
        # All elements in adj_faces are None
        for v in verts:
            if v.normal != ZERO_VEC:
                if lp_normal.dot(v.normal) < .0:
                    verts.reverse()
                    edges.reverse()
                    lp_normal *= -1
                break

    return verts, edges, lp_normal, adj_faces


def get_directions(lp, vec_upward, normal_fallback, vert_mirror_pairs, **options):
    opt_follow_face = options['follow_face']
    opt_edge_rail = options['edge_rail']
    opt_er_only_end = options['edge_rail_only_end']
    opt_threshold = options['threshold']

    verts, edges = lp[::2], lp[1::2]
    set_edges = set(edges)
    lp_normal = calc_loop_normal(verts, fallback=normal_fallback)

    # Loop order might be changed below
    if lp_normal.dot(vec_upward) < .0:
        # Make this loop's normal towards vec_upward
        verts.reverse()
        edges.reverse()
        lp_normal *= -1

    if opt_follow_face:
        adj_faces = get_adj_faces(edges)
        verts, edges, lp_normal, adj_faces = \
            reorder_loop(verts, edges, lp_normal, adj_faces)
    else:
        adj_faces = (None,) * len(edges)
    # Loop order might be changed above

    vec_edges = tuple((e.other_vert(v).co - v.co).normalized()
                      for v, e in zip(verts, edges))

    if verts[0] is verts[-1]:
        # Real loop. Popping last vertex
        verts.pop()
        HALF_LOOP = False
    else:
        # Half loop
        HALF_LOOP = True

    len_verts = len(verts)
    directions = []
    for i in range(len_verts):
        vert = verts[i]
        ix_right, ix_left = i, i - 1

        VERT_END = False
        if HALF_LOOP:
            if i == 0:
                # First vert
                ix_left = ix_right
                VERT_END = True
            elif i == len_verts - 1:
                # Last vert
                ix_right = ix_left
                VERT_END = True

        edge_right, edge_left = vec_edges[ix_right], vec_edges[ix_left]
        face_right, face_left = adj_faces[ix_right], adj_faces[ix_left]

        norm_right = face_right.normal if face_right else lp_normal
        norm_left = face_left.normal if face_left else lp_normal
        if norm_right.angle(norm_left) > opt_threshold:
            # Two faces are not flat
            two_normals = True
        else:
            two_normals = False

        tan_right = edge_right.cross(norm_right).normalized()
        tan_left = edge_left.cross(norm_left).normalized()
        tan_avr = (tan_right + tan_left).normalized()
        norm_avr = (norm_right + norm_left).normalized()

        rail = None
        if two_normals or opt_edge_rail:
            # Get edge rail
            # edge rail is a vector of an inner edge
            if two_normals or (not opt_er_only_end) or VERT_END:
                rail = get_edge_rail(vert, set_edges)
        if vert_mirror_pairs and VERT_END:
            if vert in vert_mirror_pairs:
                rail, norm_avr = get_mirror_rail(vert_mirror_pairs[vert], norm_avr)
        if (not rail) and two_normals:
            # Get cross rail
            # Cross rail is a cross vector between norm_right and norm_left
            rail = get_cross_rail(
                tan_avr, edge_right, edge_left, norm_right, norm_left)
        if rail:
            dot = tan_avr.dot(rail)
            if dot > .0:
                tan_avr = rail
            elif dot < .0:
                tan_avr = -rail

        vec_plane = norm_avr.cross(tan_avr)
        e_dot_p_r = edge_right.dot(vec_plane)
        e_dot_p_l = edge_left.dot(vec_plane)
        if e_dot_p_r or e_dot_p_l:
            if e_dot_p_r > e_dot_p_l:
                vec_edge, e_dot_p = edge_right, e_dot_p_r
            else:
                vec_edge, e_dot_p = edge_left, e_dot_p_l

            vec_tan = (tan_avr - tan_avr.project(vec_edge)).normalized()
            # Make vec_tan perpendicular to vec_edge
            vec_up = vec_tan.cross(vec_edge)

            vec_width = vec_tan - (vec_tan.dot(vec_plane) / e_dot_p) * vec_edge
            vec_depth = vec_up - (vec_up.dot(vec_plane) / e_dot_p) * vec_edge
        else:
            vec_width = tan_avr
            vec_depth = norm_avr

        directions.append((vec_width, vec_depth))

    return verts, directions


angle_presets = {'0°': 0,
                 '15°': radians(15),
                 '30°': radians(30),
                 '45°': radians(45),
                 '60°': radians(60),
                 '75°': radians(75),
                 '90°': radians(90),
                 }


def use_cashes(self, context):
    self.caches_valid = True


def assign_angle_presets(self, context):
    use_cashes(self, context)
    self.angle = angle_presets[self.angle_presets]


class OffsetEdges(Operator):
    bl_idname = "mesh.offset_edges"
    bl_label = "Offset Edges"
    bl_description = ("Extrude, Move or Offset the selected Edges\n"
                      "Operates only on separate Edge loops selections")
    bl_options = {'REGISTER', 'UNDO'}

    geometry_mode = EnumProperty(
        items=[('offset', "Offset", "Offset edges"),
               ('extrude', "Extrude", "Extrude edges"),
               ('move', "Move", "Move selected edges")],
        name="Geometry mode",
        default='offset',
        update=use_cashes
    )
    width = FloatProperty(
        name="Width",
        default=.2,
        precision=4, step=1,
        update=use_cashes
    )
    flip_width = BoolProperty(
        name="Flip Width",
        default=False,
        description="Flip width direction",
        update=use_cashes
    )
    depth = FloatProperty(
        name="Depth",
        default=.0,
        precision=4, step=1,
        update=use_cashes
    )
    flip_depth = BoolProperty(
        name="Flip Depth",
        default=False,
        description="Flip depth direction",
        update=use_cashes
    )
    depth_mode = EnumProperty(
        items=[('angle', "Angle", "Angle"),
               ('depth', "Depth", "Depth")],
        name="Depth mode",
        default='angle',
        update=use_cashes
    )
    angle = FloatProperty(
        name="Angle", default=0,
        precision=3, step=.1,
        min=-2 * pi, max=2 * pi,
        subtype='ANGLE',
        description="Angle",
        update=use_cashes
    )
    flip_angle = BoolProperty(
        name="Flip Angle",
        default=False,
        description="Flip Angle",
        update=use_cashes
    )
    follow_face = BoolProperty(
        name="Follow Face",
        default=False,
        description="Offset along faces around"
    )
    mirror_modifier = BoolProperty(
        name="Mirror Modifier",
        default=False,
        description="Take into account of Mirror modifier"
    )
    edge_rail = BoolProperty(
        name="Edge Rail",
        default=False,
        description="Align vertices along inner edges"
    )
    edge_rail_only_end = BoolProperty(
        name="Edge Rail Only End",
        default=False,
        description="Apply edge rail to end verts only"
    )
    threshold = FloatProperty(
        name="Flat Face Threshold",
        default=radians(0.05), precision=5,
        step=1.0e-4, subtype='ANGLE',
        description="If difference of angle between two adjacent faces is "
                    "below this value, those faces are regarded as flat",
        options={'HIDDEN'}
    )
    caches_valid = BoolProperty(
        name="Caches Valid",
        default=False,
        options={'HIDDEN'}
    )
    angle_presets = EnumProperty(
        items=[('0°', "0°", "0°"),
               ('15°', "15°", "15°"),
               ('30°', "30°", "30°"),
               ('45°', "45°", "45°"),
               ('60°', "60°", "60°"),
               ('75°', "75°", "75°"),
               ('90°', "90°", "90°"), ],
        name="Angle Presets",
        default='0°',
        update=assign_angle_presets
    )

    _cache_offset_infos = None
    _cache_edges_orig_ixs = None

    @classmethod
    def poll(self, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'geometry_mode', text="")

        row = layout.row(align=True)
        row.prop(self, 'width')
        row.prop(self, 'flip_width', icon='ARROW_LEFTRIGHT', icon_only=True)
        layout.prop(self, 'depth_mode', expand=True)

        if self.depth_mode == 'angle':
            d_mode = 'angle'
            flip = 'flip_angle'
        else:
            d_mode = 'depth'
            flip = 'flip_depth'
        row = layout.row(align=True)
        row.prop(self, d_mode)
        row.prop(self, flip, icon='ARROW_LEFTRIGHT', icon_only=True)
        if self.depth_mode == 'angle':
            layout.prop(self, 'angle_presets', text="Presets", expand=True)

        layout.separator()

        layout.prop(self, 'follow_face')

        row = layout.row()
        row.prop(self, 'edge_rail')
        if self.edge_rail:
            row.prop(self, 'edge_rail_only_end', text="OnlyEnd", toggle=True)

        layout.prop(self, 'mirror_modifier')
        layout.operator('mesh.offset_edges', text="Repeat")

        if self.follow_face:
            layout.separator()
            layout.prop(self, 'threshold', text="Threshold")

    def get_offset_infos(self, bm, edit_object):
        if self.caches_valid and self._cache_offset_infos is not None:
            # Return None, indicating to use cache
            return None, None

        if ENABLE_DEBUG:
            time = perf_counter()

        set_edges_orig = collect_edges(bm)
        if set_edges_orig is None:
            self.report({'WARNING'},
                        "No edges selected or edge loops could not be determined")
            return False, False

        if self.mirror_modifier:
            mirror_planes = collect_mirror_planes(edit_object)
            vert_mirror_pairs, set_edges = \
                get_vert_mirror_pairs(set_edges_orig, mirror_planes)

            if set_edges:
                set_edges_orig = set_edges
            else:
                vert_mirror_pairs = None
        else:
            vert_mirror_pairs = None

        loops = collect_loops(set_edges_orig)
        if loops is None:
            self.report({'WARNING'},
                        "Overlap detected. Select non-overlapping edge loops")
            return False, False

        vec_upward = (X_UP + Y_UP + Z_UP).normalized()
        # vec_upward is used to unify loop normals when follow_face is off
        normal_fallback = Z_UP
        # normal_fallback = Vector(context.region_data.view_matrix[2][:3])
        # normal_fallback is used when loop normal cannot be calculated

        follow_face = self.follow_face
        edge_rail = self.edge_rail
        er_only_end = self.edge_rail_only_end
        threshold = self.threshold

        offset_infos = []
        for lp in loops:
            verts, directions = get_directions(
                lp, vec_upward, normal_fallback, vert_mirror_pairs,
                follow_face=follow_face, edge_rail=edge_rail,
                edge_rail_only_end=er_only_end,
                threshold=threshold)
            if verts:
                offset_infos.append((verts, directions))

        # Saving caches
        self._cache_offset_infos = _cache_offset_infos = []
        for verts, directions in offset_infos:
            v_ixs = tuple(v.index for v in verts)
            _cache_offset_infos.append((v_ixs, directions))
        self._cache_edges_orig_ixs = tuple(e.index for e in set_edges_orig)

        if ENABLE_DEBUG:
            print("Preparing OffsetEdges: ", perf_counter() - time)

        return offset_infos, set_edges_orig

    def do_offset_and_free(self, bm, me, offset_infos=None, set_edges_orig=None):
        # If offset_infos is None, use caches
        # Makes caches invalid after offset

        if ENABLE_DEBUG:
            time = perf_counter()

        if offset_infos is None:
            # using cache
            bmverts = tuple(bm.verts)
            bmedges = tuple(bm.edges)
            edges_orig = [bmedges[ix] for ix in self._cache_edges_orig_ixs]
            verts_directions = []
            for ix_vs, directions in self._cache_offset_infos:
                verts = tuple(bmverts[ix] for ix in ix_vs)
                verts_directions.append((verts, directions))
        else:
            verts_directions = offset_infos
            edges_orig = list(set_edges_orig)

        if self.depth_mode == 'angle':
            w = self.width if not self.flip_width else -self.width
            angle = self.angle if not self.flip_angle else -self.angle
            width = w * cos(angle)
            depth = w * sin(angle)
        else:
            width = self.width if not self.flip_width else -self.width
            depth = self.depth if not self.flip_depth else -self.depth

        # Extrude
        if self.geometry_mode == 'move':
            geom_ex = None
        else:
            geom_ex = extrude_edges(bm, edges_orig)

        for verts, directions in verts_directions:
            move_verts(width, depth, verts, directions, geom_ex)

        clean(bm, self.geometry_mode, edges_orig, geom_ex)

        bpy.ops.object.mode_set(mode="OBJECT")
        bm.to_mesh(me)
        bpy.ops.object.mode_set(mode="EDIT")
        bm.free()
        self.caches_valid = False  # Make caches invalidauto
        if ENABLE_DEBUG:
            print("OffsetEdges offset: ", perf_counter() - time)

    def execute(self, context):
        # In edit mode
        edit_object = context.edit_object
        bpy.ops.object.mode_set(mode="OBJECT")

        me = edit_object.data
        bm = bmesh.new()
        bm.from_mesh(me)

        offset_infos, edges_orig = self.get_offset_infos(bm, edit_object)
        if offset_infos is False:
            bpy.ops.object.mode_set(mode="EDIT")
            return {'CANCELLED'}

        self.do_offset_and_free(bm, me, offset_infos, edges_orig)

        return {'FINISHED'}

    def restore_original_and_free(self, context):
        self.caches_valid = False  # Make caches invalid
        context.area.header_text_set()

        me = context.edit_object.data
        bpy.ops.object.mode_set(mode="OBJECT")
        self._bm_orig.to_mesh(me)
        bpy.ops.object.mode_set(mode="EDIT")

        self._bm_orig.free()
        context.area.header_text_set()

    def invoke(self, context, event):
        # In edit mode
        edit_object = context.edit_object
        me = edit_object.data
        bpy.ops.object.mode_set(mode="OBJECT")
        for p in me.polygons:
            if p.select:
                self.follow_face = True
                break

        self.caches_valid = False
        bpy.ops.object.mode_set(mode="EDIT")
        return self.execute(context)


def operator_draw(self, context):
    layout = self.layout
    col = layout.column(align=True)
    self.layout.operator_context = 'INVOKE_REGION_WIN'
    col.operator("mesh.destructive_extrude", text="Destructive Extrude")


def register():
    bpy.utils.register_class(DestructiveExtrude)
    bpy.types.VIEW3D_MT_edit_mesh_extrude.append(operator_draw)
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_class(DestructiveExtrude)
    bpy.types.VIEW3D_MT_edit_mesh_extrude.remove(operator_draw)
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
