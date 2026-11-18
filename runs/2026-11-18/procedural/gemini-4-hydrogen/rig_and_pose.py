"""Armature construction, smooth analytical skinning, and selfie portrait posing."""

import math
import numpy as np
import bpy
from mathutils import Vector, Matrix, Quaternion


def _capsule_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest distance from each point in pts (N,3) to line segment [a, b]."""
    ab = b - a
    len2 = float(np.dot(ab, ab)) + 1e-12
    t = np.clip(np.dot(pts - a, ab) / len2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def create_armature(landmarks: dict, eye_centers: dict, collection: bpy.types.Collection) -> bpy.types.Object:
    """Create hierarchical upper-body armature matching environment.json landmarks."""
    arm_data = bpy.data.armatures.new("Character_Armature")
    arm_obj = bpy.data.objects.new("Character_Rig", arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def add_bone(name, head, tail, parent=None, connected=False):
        b = eb.new(name)
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent is not None:
            b.parent = parent
            b.use_connect = connected
        return b

    spine_base = add_bone("spine_base", (0.0, -0.025, 1.020), (0.0, -0.015, 1.160))
    chest = add_bone("chest", (0.0, -0.015, 1.160), (0.0, -0.006, 1.365), parent=spine_base, connected=True)
    neck = add_bone("neck", (0.0, -0.006, 1.365), (0.0, -0.012, 1.455), parent=chest, connected=True)
    head = add_bone("head", (0.0, -0.012, 1.455), (0.0, -0.010, 1.660), parent=neck, connected=True)

    # Eye bones parented to head (aiming forward -Y by default)
    for side in ("L", "R"):
        c = Vector(eye_centers[side.lower()])
        add_bone(f"eye.{side}", c, c + Vector((0.0, -0.030, 0.0)), parent=head, connected=False)

    # Clavicle, upper arm, forearm, hand
    for side, s_key in (("L", "l"), ("R", "r")):
        sx = 1.0 if side == "L" else -1.0
        clav_h = Vector((sx * 0.024, -0.048, 1.335))
        sh = Vector(landmarks[f"joint-{s_key}-shoulder"])
        el = Vector(landmarks[f"joint-{s_key}-elbow"])
        ha = Vector(landmarks[f"joint-{s_key}-hand"])
        axis_fore = (ha - el).normalized()
        tip = ha + axis_fore * 0.145

        b_clav = add_bone(f"clavicle.{side}", clav_h, sh, parent=chest, connected=False)
        b_uarm = add_bone(f"upper_arm.{side}", sh, el, parent=b_clav, connected=True)
        b_farm = add_bone(f"forearm.{side}", el, ha, parent=b_uarm, connected=True)
        add_bone(f"hand.{side}", ha, tip, parent=b_farm, connected=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def assign_analytical_weights(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object):
    """Assign smooth, artifact-free analytical capsule bone weights to a mesh."""
    co = np.array([v.co[:] for v in mesh_obj.data.vertices], dtype=np.float64)
    n_verts = len(co)

    # Collect rest-pose bone endpoints
    bone_segs = {}
    for b in arm_obj.data.bones:
        bone_segs[b.name] = (np.array(b.head_local[:], dtype=np.float64), np.array(b.tail_local[:], dtype=np.float64))

    weights = {b_name: np.zeros(n_verts, dtype=np.float64) for b_name in bone_segs if not b_name.startswith("eye.")}

    # Head: strictly rigid 1.0 above z=1.445 and forward face, smoothly blending down neck
    z = co[:, 2]
    x = co[:, 0]
    x_abs = np.abs(x)

    def sstep(e0, e1, v):
        t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # Neck & head vertical blend (exclude arms x_abs > 0.12)
    is_torso_neck = sstep(0.16, 0.10, x_abs)
    w_head = sstep(1.415, 1.460, z) * is_torso_neck
    w_neck = (sstep(1.345, 1.395, z) - sstep(1.415, 1.460, z)) * is_torso_neck
    w_neck = np.clip(w_neck, 0.0, 1.0)

    # Spine base vs chest
    w_chest_base = sstep(1.11, 1.24, z) * (1.0 - sstep(1.345, 1.395, z))
    w_spine = (1.0 - sstep(1.11, 1.24, z))

    # Arm bone weights via capsule distance (active outside core torso)
    arm_mask = sstep(0.095, 0.155, x_abs) * (1.0 - sstep(1.40, 1.45, z))
    for side, sx in (("L", 1.0), ("R", -1.0)):
        side_m = (np.sign(x) == sx).astype(np.float64) * arm_mask
        if not np.any(side_m > 0):
            continue

        a_c, b_c = bone_segs[f"clavicle.{side}"]
        a_u, b_u = bone_segs[f"upper_arm.{side}"]
        a_f, b_f = bone_segs[f"forearm.{side}"]
        a_h, b_h = bone_segs[f"hand.{side}"]

        # Axial progression along shoulder -> elbow -> wrist -> hand
        d_clav = _capsule_distance(co, a_c, b_c)
        d_uarm = _capsule_distance(co, a_u, b_u)
        d_farm = _capsule_distance(co, a_f, b_f)
        d_hand = _capsule_distance(co, a_h, b_h)

        # Sharpness sigma for joint blending (~3.5 cm)
        sig = 0.032
        raw_c = np.exp(-0.5 * (d_clav / 0.048) ** 2) * (1.0 - sstep(0.145, 0.185, x_abs))
        raw_u = np.exp(-0.5 * (d_uarm / sig) ** 2) * sstep(0.135, 0.175, x_abs)
        raw_f = np.exp(-0.5 * (d_farm / sig) ** 2) * sstep(0.285, 0.335, x_abs)
        raw_h = np.exp(-0.5 * (d_hand / 0.055) ** 2) * sstep(0.415, 0.445, x_abs)

        tot_arm = raw_c + raw_u + raw_f + raw_h + 1e-9
        # Blend torso chest into clavicle/upper arm smoothly around x = 0.10..0.18
        arm_blend = sstep(0.105, 0.175, x_abs) * (1.0 - sstep(1.38, 1.44, z)) * (np.sign(x) == sx)

        weights[f"clavicle.{side}"] = (raw_c / tot_arm) * arm_blend
        weights[f"upper_arm.{side}"] = (raw_u / tot_arm) * arm_blend
        weights[f"forearm.{side}"] = (raw_f / tot_arm) * arm_blend
        weights[f"hand.{side}"] = (raw_h / tot_arm) * arm_blend

    total_arm = sum(weights[k] for k in weights if "." in k)
    core_rem = np.clip(1.0 - total_arm, 0.0, 1.0)

    weights["head"] = w_head * core_rem
    weights["neck"] = w_neck * core_rem
    rem_torso = np.clip(core_rem - weights["head"] - weights["neck"], 0.0, 1.0)
    weights["chest"] = w_chest_base * rem_torso
    weights["spine_base"] = (1.0 - w_chest_base) * rem_torso

    # Normalize & assign vertex groups
    total = np.zeros(n_verts, dtype=np.float64)
    for arr in weights.values():
        total += arr
    total = np.maximum(total, 1e-8)

    for b_name, arr in weights.items():
        vg = mesh_obj.vertex_groups.get(b_name) or mesh_obj.vertex_groups.new(name=b_name)
        norm_arr = arr / total
        idxs = np.where(norm_arr > 1e-4)[0]
        for i in idxs:
            vg.add([int(i)], float(norm_arr[i]), "REPLACE")

    # Add Armature modifier (above/before Subdivision!)
    arm_mod = mesh_obj.modifiers.new("Armature", type="ARMATURE")
    arm_mod.object = arm_obj
    arm_mod.use_vertex_groups = True
    # Ensure Armature executes before Subdivision
    mesh_obj.modifiers.move(len(mesh_obj.modifiers) - 1, 0)


def parent_eyes_to_rig(eye_groups: list, arm_obj: bpy.types.Object):
    """Attach eye objects rigidly to eye.L/R and head bones."""
    for side, o_in, o_cor, o_tear, o_car in eye_groups:
        for o in (o_in, o_cor):
            b = arm_obj.data.bones[f"eye.{side}"]
            rest_parent_mat = arm_obj.matrix_world @ b.matrix_local @ Matrix.Translation(Vector((0.0, b.length, 0.0)))
            o.parent = arm_obj
            o.parent_type = "BONE"
            o.parent_bone = f"eye.{side}"
            o.matrix_parent_inverse = rest_parent_mat.inverted()
        for o in (o_tear, o_car):
            b = arm_obj.data.bones["head"]
            rest_parent_mat = arm_obj.matrix_world @ b.matrix_local @ Matrix.Translation(Vector((0.0, b.length, 0.0)))
            o.parent = arm_obj
            o.parent_type = "BONE"
            o.parent_bone = "head"
            o.matrix_parent_inverse = rest_parent_mat.inverted()


def _aim_bone_toward(pb: bpy.types.PoseBone, target_world: Vector, arm_obj: bpy.types.Object, roll_deg: float = 0.0):
    """Set pose bone rotation_quaternion so bone (head->tail) points directly toward target_world."""
    bpy.context.view_layer.update()
    mat_w = arm_obj.matrix_world @ pb.matrix
    head_w = mat_w.translation.copy()
    cur_dir = (mat_w @ Vector((0.0, pb.length, 0.0)) - head_w).normalized()
    want_dir = (target_world - head_w).normalized()
    dq_w = cur_dir.rotation_difference(want_dir)
    # Apply world-space delta rotation onto pose bone's local rotation
    pb.rotation_mode = "QUATERNION"
    bone_rot_w = mat_w.to_quaternion()
    new_rot_w = dq_w @ bone_rot_w
    # Convert world rotation back to pose-bone local rotation
    if pb.parent:
        parent_rot_w = (arm_obj.matrix_world @ pb.parent.matrix).to_quaternion()
        rest_rel = pb.parent.bone.matrix_local.to_quaternion().inverted() @ pb.bone.matrix_local.to_quaternion()
        pb.rotation_quaternion = (parent_rot_w @ rest_rel).inverted() @ new_rot_w
    else:
        rest_q = pb.bone.matrix_local.to_quaternion()
        pb.rotation_quaternion = rest_q.inverted() @ new_rot_w
    if roll_deg != 0.0:
        pb.rotation_quaternion = pb.rotation_quaternion @ Quaternion((0.0, 1.0, 0.0), math.radians(roll_deg))


def pose_for_selfie(arm_obj: bpy.types.Object, beauty_cam_pos: Vector):
    """Apply natural gentle head tilt, forward lean, selfie arm reach, and eye gaze toward camera."""
    pb = arm_obj.pose.bones

    def set_euler(name, rx_deg, ry_deg, rz_deg):
        b = pb[name]
        b.rotation_mode = "QUATERNION"
        q_world = (
            Quaternion((1, 0, 0), math.radians(rx_deg))
            @ Quaternion((0, 1, 0), math.radians(ry_deg))
            @ Quaternion((0, 0, 1), math.radians(rz_deg))
        )
        rest_q = b.bone.matrix_local.to_quaternion()
        b.rotation_quaternion = rest_q.inverted() @ q_world @ rest_q
        bpy.context.view_layer.update()

    # Torso gentle forward lean & subtle twist
    set_euler("spine_base", 2.0, -1.0, -0.5)
    set_euler("chest", 2.5, -2.0, -1.2)
    # Gentle head tilt toward her right shoulder (~19 deg total roll, gentle ~6.8 deg yaw, open neck)
    set_euler("neck", 0.5, -5.5, -1.8)
    set_euler("head", -1.5, -11.5, -3.5)

    # Clavicles softly forward/down-relaxed
    set_euler("clavicle.L", 5.0, 0.0, -7.0)
    set_euler("clavicle.R", 5.0, 0.0, 7.0)

    # Aim arms cleanly toward natural selfie reach targets
    _aim_bone_toward(pb["upper_arm.R"], Vector((-0.245, -0.165, 1.155)), arm_obj, roll_deg=15.0)
    _aim_bone_toward(pb["forearm.R"], Vector((-0.245, -0.465, 1.205)), arm_obj, roll_deg=-10.0)
    _aim_bone_toward(pb["hand.R"], Vector((-0.205, -0.575, 1.225)), arm_obj)

    _aim_bone_toward(pb["upper_arm.L"], Vector((0.255, -0.145, 1.145)), arm_obj, roll_deg=-15.0)
    _aim_bone_toward(pb["forearm.L"], Vector((0.255, -0.445, 1.175)), arm_obj, roll_deg=10.0)
    _aim_bone_toward(pb["hand.L"], Vector((0.215, -0.555, 1.195)), arm_obj)

    # Aim eyes directly at beauty camera optical axis
    bpy.context.view_layer.update()
    eye_target = beauty_cam_pos + Vector((0.0, 0.0, -0.012))
    for side in ("L", "R"):
        eye_pb = pb[f"eye.{side}"]
        eye_pb.rotation_mode = "QUATERNION"
        _aim_bone_toward(eye_pb, eye_target, arm_obj)
    bpy.context.view_layer.update()
