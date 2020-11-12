import bpy
import bmesh
from mathutils import Matrix, Vector, kdtree
from mathutils.geometry import intersect_line_plane
from bpy_extras import view3d_utils

bl_info = {
    "name": "Destructive Extrude :)",
    "location": "View3D > Add > Mesh > Destructive Extrude,",
    "description": "Extrude how SketchUp.",
    "author": "Vladislav Kindushov",
    "version": (0, 1, 0),
    "blender": (2, 91, 0),
    "category": "Mesh",
}
EN_Mode = None
def SetENMode(mode):
    global EN_Mode
    EN_Mode = mode

class State():
    def __init__(self, LocalMatrix, Normal):
        self.GX = [Vector((-1,0,0)), Vector((0,1,0)), 'GX']
        self.GY = [Vector((0,-1,0)), Vector((-1,0,0)), 'GY']
        self.GZ = [Vector((0,0,-1)), self.Z_Plane(), 'GZ'] 

        self.LX = [LocalMatrix.inverted() @ Vector((1,0,0)), LocalMatrix.inverted() @ (Vector((0,1,0))), 'LX']
        self.LY = [LocalMatrix.inverted() @ Vector((0,1,0)), LocalMatrix.inverted() @ (Vector((1,0,0))), 'LY']
        self.LZ = [LocalMatrix.inverted() @ Vector((0,0,1)), self.Z_Local_Plane(), 'LZ']

        cam = bpy.context.region_data.view_rotation @ Vector((0.0,0.0,-1.0))
        self.Normal = [Normal, cam, 'N']

        self.Current = self.Normal
    
    def Z_Plane(self):
        cam = bpy.context.region_data.view_rotation @ Vector((0.0,0.0,-1.0))
        val = Vector((0,1,0)).dot(cam)
        
        if val > 0.5:
            return Vector((0,-1,cam.z))
        elif val < -0.5:
            return Vector((0,1,cam.z))
        
    def Z_Local_Plane(self):
        cam = bpy.context.active_object.matrix_world.inverted() @ (bpy.context.region_data.view_rotation @ Vector((0.0,0.0,-1.0)))
        val = Vector((0,1,0)).dot(cam)
        
        if val > 0.5:
            return Vector((0,1,0))
        elif val < 0.5:
            return Vector((0,-1,0))
    
    def SwapLocalGloval(self):
        C = self.Current[2]

        if C == "LX" or C == "GX":
            if C == "LX":
                self.Current = self.GX
            else:
                self.Current = self.LX
    #--------------------------------------------#
        if C == "LY" or C == "GY":
            if C == "LY":
                self.Current = self.GY
            else:
                self.Current = self.LY
    #--------------------------------------------#
        if C == "LZ" or C == "GZ":
            if C == "LZ":
                self.Current = self.GZ
            else:
                self.Current = self.LZ





class MainObjectDriver():
    def __init__(self, obj, ext_obj):
        self.obj = obj
        self.Modifier = self.GetModifiers()
        self.DisableModifiers()
        self.BooleanModifier = self.CreateBooleanModifier(self.obj, ext_obj)


    def GetModifiers(self):
        """Получаем рабочие модификаторы"""
        EnableModifiers = [] # str
        for i in self.obj.modifiers:
            if i.show_viewport:
                EnableModifiers.append(i.name)
        return EnableModifiers

    def DisableModifiers(self):
        for i in self.Modifier:
            self.obj.modifiers[i].show_viewport = False
    
    def EnableModifiers(self):
        for i in self.Modifier:
            self.obj.modifiers[i].show_viewport = True
    
    def CreateBooleanModifier(self, obj, ext_obj):
            #________Set Boolean________#
        bpy.context.view_layer.objects.active = obj
        BooleanModifier = bpy.context.object.modifiers.new('DestructiveBoolean', 'BOOLEAN')
        bpy.context.object.modifiers["DestructiveBoolean"].operation = 'DIFFERENCE'
        bpy.context.object.modifiers["DestructiveBoolean"].object = ext_obj
        bpy.context.object.modifiers["DestructiveBoolean"].show_viewport = True
        return BooleanModifier

class SceneDriver():
    def __init__(self):
        self.ShowAllEdges = bpy.context.active_object.show_all_edges
        self.ShowWire = bpy.context.active_object.show_wire
        self.Origyn = bpy.context.tool_settings.transform_pivot_point
        self.CursorPosition = self.GetCursorPosition()
        self.KD = self.CreateKD_Tree()
        



    def SetVisualMeshSetings(self):
        bpy.context.active_object.show_all_edges = self.ShowAllEdges
        bpy.context.active_object.show_wire = self.ShowWire

    def GetCursorPosition(self):
        return bpy.context.scene.cursor.location

    def SetCursorPosition(self,context, is_Set = False):
        bpy.context.scene.cursor.location = self.CursorLocation

    def Cansel(self, ext_obj, main_obj):
        bpy.data.objects.remove(ext_obj.obj)
        bpy.context.view_layer.objects.active = main_obj.obj
        bpy.ops.object.modifier_remove(modifier='DestructiveBoolean')
        self.SetVisualMeshSetings()
        main_obj.EnableModifiers()
        bpy.ops.object.mode_set(mode='EDIT')
    
    

    def Finish(self, ext_obj, main_obj, BevelUpdate=False):

        bpy.context.view_layer.objects.active = main_obj.obj
        bpy.ops.object.modifier_apply(modifier="DestructiveBoolean")

        bpy.data.objects.remove(ext_obj.obj)
        self.SetVisualMeshSetings()
        main_obj.EnableModifiers()
        bpy.ops.object.mode_set(mode='EDIT')

    def CreateKD_Tree(self):
        obj = bpy.context.active_object
        point = []
        for v in obj.data.vertices:
            point.append(v.co)

        for e in obj.data.edges:
            point.append((obj.data.vertices[e.vertices[0]].co + obj.data.vertices[e.vertices[1]].co) / 2)

        for p in obj.data.polygons:
            point.append(p.center)

        kd = kdtree.KDTree(len(point))

        for i in range(len(point)):
            kd.insert(point[i], i)

        kd.balance()
        return kd

    def KD_Find_Point(self, point, center, normal):
        co, index, dist = self.KD.find(point)
        return (((co - center) @ normal) * -1)

class MouseDriver():
    def __init__(self, center, face_normal):
        self.Center = center.copy()
        self.CameraVector = bpy.context.region_data.view_rotation @ Vector(( 0.0,0.0,-1.0))
        self.face_normal = face_normal
        self.LastPoint = center

    def GetValue(self, MousePosition, normal, clip_normal):
        region = bpy.context.region
        rv3d = bpy.context.region_data

        view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, MousePosition)
        ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, MousePosition)
        MouseVector =  view_vector_mouse + ray_origin_mouse
        pointLoc = intersect_line_plane(ray_origin_mouse, MouseVector, self.Center, clip_normal)

        # dvec = self.Center-pointLoc
        # dnormal = dvec.dot(normal)
        # val = self.Center + Vector(dnormal*normal)

        return ((((pointLoc - self.Center)) @ normal) * -1)

    def GetSnapSurface(self, MousePosition):
        region = bpy.context.region
        rv3d = bpy.context.region_data
        view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, MousePosition)
        ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, MousePosition)
        direction = ray_origin_mouse + (view_vector_mouse * 1000)
        #direction.normalized()
        result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, ray_origin_mouse, direction)

        return location

class ExtrudeObject():
    def __init__(self, MainObject):
        self.obj = self.CreateNewObject()
        self.TransformObject(self.obj)
        self.ClearModifiers(self.obj)
        self.Solidify, self.Displace = self.SetupModifier(self.obj)
        self.obj.display_type = 'WIRE'
        self.GeneralNormal = self.CalculateNormal(self.obj)
        self.Center = self.GetCenter()
        self.IsNormal = True

    

    def SetOffsetValue(self, value):
        if self.IsNormal:
            self.Solidify.thickness = value# * -1
        else:
            self.Displace.strength = value


    def CreateNewObject(self):
        # ________Duplicate Object________#
        bpy.ops.mesh.duplicate()
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')
        return bpy.context.selected_objects[-1] 
    
    def ClearModifiers(self, obj):
    # ________Clear Modifiers________#
        while len(obj.modifiers) != 0: 
            obj.modifiers.remove(obj.modifiers[0])

    def SetupModifier(self, obj):
        # ________Set Solidify________#
        #bpy.context.view_layer.objects.active = self.obj
        self.obj.modifiers.new('DestructiveSolidify', 'SOLIDIFY')
        self.obj.modifiers.new('DestructiveDisplace', 'DISPLACE')
        sol = self.obj.modifiers['DestructiveSolidify']
        dis = self.obj.modifiers['DestructiveDisplace']

        self.obj.vertex_groups.new(name='Destructive')

        sol.shell_vertex_group = "Destructive"
        sol.use_even_offset = True
        #sol.offset = 0

        dis.vertex_group = "Destructive"
        dis.direction = 'X'
        dis.space = 'GLOBAL'
        dis.strength = 0

        return sol, dis

    def GetCenter(self):
        local_bbox_center = 0.125 * sum((Vector(b) for b in self.obj.bound_box), Vector())
        return self.obj.matrix_world @ local_bbox_center

    def TransformObject(self, ext_obj):
        pass

    def CalculateNormal(self, obj):
        normal = Vector((0.0, 0.0, 0.0))
        for i in obj.data.polygons:
            normal += i.normal.copy()
        #print('notmal ', normal)
        return normal

    def SetAxis(self, axis):
        self.Displace.direction = axis

    def SwitchTohAxis(self, axis, mode):
        # print('ax', axis)
        # print('mode', mode.Current[2])
        if axis in mode.Current[2]:
            # print('local')
            mode.SwapLocalGloval()
        else:
            if axis == "X":
                mode.Current = mode.GX
            elif axis == "Z":
                mode.Current = mode.GY
            elif axis == "Z":
                mode.Current = mode.GZ
        self.Solidify.thickness = 0.0
        self.Displace.direction = axis
        self.IsNormal = False
        if "L" in mode.Current[2]:
            self.Displace.space = 'LOCAL'
        else:
            self.Displace.space = 'GLOBAL'
            

    def SwitchToNormal(self):
        self.IsNormal = True
        self.Displace.strength = 0

def Convert(a):
    if a =='ZERO' or a == 'NUMPAD_0':
       return '0'
    if a =='ONE' or a == 'NUMPAD_1':
       return '1'
    if a =='TWO' or a == 'NUMPAD_2':
       return '2'
    if a =='THREE' or a == 'NUMPAD_3':
       return '3'
    if a =='FOUR' or a == 'NUMPAD_4':
       return '4'
    if a =='FIVE' or a == 'NUMPAD_5':
       return '5'
    if a =='SIX' or a == 'NUMPAD_6':
       return '6'
    if a =='SEVEN' or a == 'NUMPAD_7':
       return '7'
    if a =='EIGHT' or a == 'NUMPAD_8':
       return '8'
    if a =='NINE' or a == 'NUMPAD_9':
        print('9')
        return '9'
    if a =='MINUS' or a == 'NUMPAD_MINUS':
        return '-'
    if a =='PLUS' or a == 'NUMPAD_PLUS':
        return '+'
    if a =='SLASH' or a == 'NUMPAD_SLASH':
        return '/'
    if a =='NUMPAD_ASTERIX':
        return '*'

def ManualInput(self, context, event):
    if (event.type == 'BACK_SPACE' and len(self.expression) != 0) and event.value == 'PRESS':
        if len(self.expression) != 0:
            self.expression = self.expression[:-1]
            if len(self.expression) != 0:
                try:
                    self.offset = eval(self.expression)
                    self.ExtObject.SetOffsetValue(self.offset*-1)
                except:
                    pass
                context.area.header_text_set('Offset-' + str(self.expression) + '=' + str(self.offset)  + ' ' + 'Press X, Y, Z To Axis Constrain. Double Press For Contrain To local Axis')
                return {'RUNNING_MODAL'}

    if event.type in self.event and event.value == 'PRESS':
        if event.type in ['MINUS','PLUS','SLASH','NUMPAD_ASTERIX', 'NUMPAD_SLASH', 'NUMPAD_MINUS', 'NUMPAD_PLUS']:
            if len(self.expression) != 0:
                if not self.expression[-1] in ['+','-','*','/']:
                    self.expression = self.expression + Convert(event.type)
        else:
            self.expression = self.expression + Convert(event.type)
            try:
                self.offset = eval(self.expression)
                self.ExtObject.SetOffsetValue(self.offset*-1)
            except:
                pass
        context.area.header_text_set('Offset-' + str(self.expression) + '=' + str(self.offset)  + ' ' + 'Press X, Y, Z To Axis Constrain. Double Press For Contrain To local Axis')
        return {'RUNNING_MODAL'}
    else:
        return None

    


class DestuctiveExtrude(bpy.types.Operator):
    bl_idname = "mesh.destuctive_extrude"
    bl_label = "Destructive Extrude"
    bl_options = {"REGISTER", "UNDO", "GRAB_CURSOR", "BLOCKING"}

    @classmethod
    def poll(cls, context):
        return (context.mode == "EDIT_MESH")

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE':
            self.Scene.Finish(self.ExtObject, self.MainObject)
            context.area.header_text_set(None)
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.Scene.Cansel(self.ExtObject, self.MainObject)
            context.area.header_text_set(None)
            return {'CANCELLED'}

        if (event.type in self.event or event.type == 'BACK_SPACE') and event.value == 'PRESS':
            value = ManualInput(self, context, event)
            if not value is None:
                return value

        if len(self.expression) != 0:
            if self.expression.find('+') != -1 or self.expression.find('-') != -1 or self.expression.find('*') != -1 or self.expression.find('/') != -1:
                context.area.header_text_set('Offset-' + str(self.expression) + '=' + str(self.offset)  + ' ' + 'Press X, Y, Z To Axis Constrain. Double Press For Contrain To local Axis')
            else:
                context.area.header_text_set('Offset-' + str(self.expression) + ' ' + 'Press X, Y, Z To Axis Constrain. Double Press For Contrain To local Axis')
            return {'RUNNING_MODAL'}

        context.area.header_text_set('Offset-' + str(self.offset) + ' ' + 'Press X, Y, Z To Axis Constrain. Double Press For Contrain To local Axis')
        if event.ctrl:
            MousePosition = Vector((event.mouse_x - context.area.regions.data.x, event.mouse_y - context.area.regions.data.y))
            point = self.Mouse.GetSnapSurface(MousePosition)
            self.offset = self.Scene.KD_Find_Point(point, self.ExtObject.Center, self.Mode.Current[0])
            self.ExtObject.SetOffsetValue(self.offset)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            MousePosition = Vector((event.mouse_x - context.area.regions.data.x, event.mouse_y - context.area.regions.data.y))
            self.offset = self.Mouse.GetValue(MousePosition, self.Mode.Current[0], self.Mode.Current[1])
            self.ExtObject.SetOffsetValue(self.offset)

        if event.type in ['X', 'Y', 'Z'] and event.value == 'PRESS':
            self.ExtObject.SwitchTohAxis(event.type, self.Mode)

        if event.type == 'N' and event.value == 'PRESS':
            self.ExtObject.SwitchToNormal()
            self.Mode.Current = self.Mode.Normal

        return {'RUNNING_MODAL'}


    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            self.offset = 0.0
            obj = context.active_object
            self.ExtObject = ExtrudeObject(obj)
            self.MainObject = MainObjectDriver(obj, self.ExtObject.obj)
            self.Scene = SceneDriver()
            self.Mouse = MouseDriver(self.ExtObject.Center, self.ExtObject.GeneralNormal)
            self.Mode = State(self.MainObject.obj.matrix_world, self.ExtObject.GeneralNormal)

            self.event = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE',
            'MINUS','PLUS','SLASH','NUMPAD_ASTERIX', 'NUMPAD_SLASH', 'NUMPAD_MINUS', 'NUMPAD_PLUS', 
            'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3','NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_0' ]
            self.expression = ''

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "is not 3dview")
            return {'CANCELLED'}

    

classes = (DestuctiveExtrude)

def operator_draw(self, context):
	layout = self.layout
	col = layout.column(align=True)
	self.layout.operator_context = 'INVOKE_REGION_WIN'
	col.operator("mesh.destuctive_extrude", text="Destructive Extrude")

def register():
    bpy.utils.register_class(classes)
    bpy.types.VIEW3D_MT_edit_mesh_extrude.append(operator_draw)

def unregister():
    bpy.utils.unregister_class(classes)
    bpy.types.VIEW3D_MT_edit_mesh_extrude.remove(operator_draw)

if __name__ == "__main__":
    register()
