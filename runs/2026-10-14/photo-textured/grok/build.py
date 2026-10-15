#!/usr/bin/env python3
"""Build an editable clothed upper-body portrait in Blender 5.1.

From this directory:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py

Preview (Eevee-free Cycles, low samples, preview/*.png, still saves character.blend):

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview

The script imports input/upper-body.obj (MakeHuman hm08, CC0), sculpts a clothed
feminine likeness, poses a selfie, samples input/reference.png for skin, lip,
iris and knit color, and saves character.blend before rendering.

source-character.blend is the previous procedural-only scene and is never written.
"""

import math
import os
import sys
import json
import traceback

import bpy
import bmesh
from mathutils import Vector, Matrix, Quaternion
from bpy_extras.object_utils import world_to_camera_view

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def smoother(t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def falloff(dist, radius):
    if dist >= radius or radius <= 1e-8:
        return 0.0
    return smoother(1.0 - dist / radius)


def vlerp(a, b, t):
    return a + (b - a) * t


def dist_point_segment(p, a, b):
    ab = b - a
    denom = ab.length_squared
    if denom < 1e-12:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / denom))
    return (p - (a + ab * t)).length


def swing_matrix(rest_head, rest_tail, pose_head, pose_tail, twist=0.0):
    """Rigid map sending the rest bone onto the posed bone, swing plus optional twist."""
    rest_y = rest_tail - rest_head
    pose_y = pose_tail - pose_head
    if rest_y.length < 1e-8 or pose_y.length < 1e-8:
        return Matrix.Identity(4)
    q = rest_y.normalized().rotation_difference(pose_y.normalized())
    if abs(twist) > 1e-6:
        q = Quaternion(pose_y.normalized(), twist) @ q
    r = q.to_matrix().to_4x4()
    return Matrix.Translation(pose_head) @ r @ Matrix.Translation(-rest_head)


def rigid_about(pivot, quat):
    r = quat.to_matrix().to_4x4()
    return Matrix.Translation(pivot) @ r @ Matrix.Translation(-pivot)


def extra_args():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def link(col, obj):
    col.objects.link(obj)
    # Unlink from the scene master if the operator/import put it there.
    root = bpy.context.scene.collection
    if obj.name in root.objects:
        root.objects.unlink(obj)


def new_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def look_track(cam, target):
    """Point a camera at target with a Track To constraint (reliable in background)."""
    name = cam.name + "_target"
    empty = bpy.data.objects.get(name)
    if empty is None:
        empty = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(empty)
        empty.hide_render = True
        empty.empty_display_size = 0.02
    empty.location = Vector(target)
    existing = [c for c in cam.constraints if c.type == "TRACK_TO"]
    con = existing[0] if existing else cam.constraints.new("TRACK_TO")
    con.target = empty
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return empty


def set_active(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def shade_smooth(obj):
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()


def ensure_color_attr(mesh, name):
    existing = mesh.color_attributes.get(name)
    if existing is not None:
        return existing
    attr = mesh.color_attributes.new(name=name, type="BYTE_COLOR", domain="POINT")
    try:
        mesh.color_attributes.active_color = attr
    except Exception:
        pass
    return attr


def ensure_float_attr(mesh, name):
    existing = mesh.attributes.get(name)
    if existing is not None and existing.data_type == "FLOAT" and existing.domain == "POINT":
        return existing
    if existing is not None:
        mesh.attributes.remove(existing)
    return mesh.attributes.new(name, "FLOAT", "POINT")


# ---------------------------------------------------------------------------
# Scene reset and import
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        try:
            scene.view_settings.view_transform = "Filmic"
        except Exception:
            pass
    scene.view_settings.exposure = -0.05
    try:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 3
    except Exception:
        pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.resolution_percentage = 100
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 6
    scene.cycles.sample_clamp_indirect = 4.0
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    try:
        bpy.context.view_layer.cycles.use_denoising = True
    except Exception:
        pass
    return scene


def import_body():
    path = os.path.join(ROOT, "input", "upper-body.obj")
    # The file is already Z-up with the face toward -Y. Blender's importer
    # still applies a Y-up conversion as a +90° X rotation; clear it so the
    # stored coordinates stay the visual coordinates.
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.location = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    obj.name = "Body"
    obj.data.name = "BodyMesh"
    shade_smooth(obj)
    return obj


# ---------------------------------------------------------------------------
# Rest-pose sculpt. Offsets are in metres on the MakeHuman cage.
# Unseen profile, ears and torso volume are inferences from the front photo.
# ---------------------------------------------------------------------------

def sculpt_likeness(mesh):
    verts = mesh.vertices

    def region_push(center, radius, offset):
        c = Vector(center)
        off = Vector(offset)
        for v in verts:
            w = falloff((v.co - c).length, radius)
            if w:
                v.co += off * w

    # Narrower, less projected nose. The base nose is a generic adult projection.
    for v in verts:
        co = v.co
        if abs(co.x) > 0.028 or co.z < 1.465 or co.z > 1.555 or co.y > -0.135:
            continue
        # Strongest at the tip / bridge, fading to the cheeks.
        tip = Vector((0.0, -0.162, 1.505))
        w = falloff((co - tip).length, 0.038)
        if w <= 0:
            continue
        # Pull protrusion back toward the eye plane. The reference nose is small.
        front = min(0.0, co.y - (-0.145))
        co.y -= front * 0.70 * w
        co.x *= 1.0 - 0.40 * w
        co.z += 0.0035 * falloff((co - Vector((0.0, -0.166, 1.508))).length, 0.018)

    # Soften the jaw and shorten the chin a little (youthful oval).
    for v in verts:
        co = v.co
        if co.z > 1.50 or co.z < 1.425 or abs(co.x) > 0.085 or co.y > -0.06:
            continue
        jaw_w = falloff(abs(abs(co.x) - 0.046) + abs(co.z - 1.452) * 0.55, 0.048)
        co.x *= 1.0 - 0.26 * jaw_w
        chin_w = falloff(Vector((co.x, 0.0, co.z - 1.436)).length, 0.038)
        co.z += 0.009 * chin_w
        co.y += 0.002 * chin_w

    # Cheek fullness, a little higher and rounder than the base cage.
    for side in (1.0, -1.0):
        region_push((side * 0.042, -0.128, 1.508), 0.036, (side * 0.0045, -0.0065, 0.003))

    # Open and slightly enlarge the eyes. Centers sit on the lid surface.
    for side in (1.0, -1.0):
        c = Vector((side * 0.0312, -0.1365, 1.5455))
        for v in verts:
            d = v.co - c
            dist = d.length
            if dist > 0.024:
                continue
            w = falloff(dist, 0.024)
            v.co.x = c.x + d.x * (1.0 + 0.10 * w)
            v.co.z = c.z + d.z * (1.0 + 0.08 * w)
            if d.z > 0.001:
                v.co.z += 0.0016 * w
            elif d.z < -0.001:
                v.co.z -= 0.0010 * w
            # A shallow double-eyelid crease just above the upper lid.
            if 0.004 < d.z < 0.012 and abs(d.x) < 0.012:
                v.co.y += 0.0013 * w

    # Brows: a touch higher, brow ridge pushed back so it is not heavy.
    for v in verts:
        co = v.co
        if co.z < 1.548 or co.z > 1.590 or abs(co.x) > 0.06 or abs(co.x) < 0.008 or co.y > -0.09:
            continue
        w = falloff(abs(co.z - 1.566) + abs(abs(co.x) - 0.032) * 0.5, 0.03)
        co.z += 0.0028 * w
        co.y += 0.0032 * w

    # Lips: a little fuller, corners lifted into a closed-mouth smile.
    for v in verts:
        co = v.co
        if abs(co.x) > 0.034 or co.z < 1.462 or co.z > 1.500 or co.y > -0.140:
            continue
        lip_line = 1.4835
        dist_lip = abs(co.z - lip_line) + abs(co.y + 0.158) * 0.35
        w = falloff(dist_lip, 0.018)
        if w <= 0:
            continue
        # Fullness outward.
        co.y -= 0.0018 * w
        # Upper vs lower thickness.
        if co.z >= lip_line:
            co.z += 0.0011 * w
        else:
            co.z -= 0.0014 * w
        # Smile: corners rise and ease back; center stays.
        corner = abs(co.x) / 0.026
        corner = max(0.0, min(1.0, corner))
        corner = corner * corner
        co.z += 0.0036 * corner * w
        co.y += 0.0012 * corner * w
        co.x += (1.0 if co.x >= 0 else -1.0) * 0.0010 * corner * w
        # Keep the smile closed. Inner lip verts move back toward the lip line.
        gap = co.z - 1.4835
        co.z -= gap * 0.35 * w

    # Slender neck.
    for v in verts:
        co = v.co
        if 1.355 < co.z < 1.455 and abs(co.x) < 0.062 and co.y > -0.09:
            w = falloff(abs(co.z - 1.40), 0.06)
            co.x *= 1.0 - 0.10 * (0.4 + 0.6 * w)
            if co.y > -0.02:
                co.y *= 0.94

    # Tuck ears slightly toward the head (the photo hides them under hair).
    for v in verts:
        co = v.co
        if abs(co.x) > 0.062 and 1.475 < co.z < 1.605 and co.y > -0.11:
            w = falloff(abs(abs(co.x) - 0.082), 0.03)
            sign = 1.0 if co.x >= 0 else -1.0
            co.x -= sign * 0.0035 * w
            co.y += 0.003 * w

    # Modest chest volume under the sweater, narrower waist, softer shoulders.
    # These are inferences: the photo only shows a loose sweater.
    for v in verts:
        co = v.co
        if 1.145 < co.z < 1.300 and abs(co.x) < 0.095 and co.y < -0.01:
            for side, sx in ((1.0, 0.038), (-1.0, -0.038)):
                c = Vector((sx, -0.085, 1.225))
                w = falloff((co - c).length, 0.055)
                if w:
                    co.y -= 0.008 * w
                    co.x += side * 0.004 * w
        if co.z < 1.175 and abs(co.x) < 0.115 and abs(co.y) < 0.10:
            co.x *= 0.93
        if 1.280 < co.z < 1.380 and 0.10 < abs(co.x) < 0.20:
            sign = 1.0 if co.x >= 0 else -1.0
            co.x -= sign * 0.006 * falloff(abs(co.z - 1.33), 0.06)

    # Slim the upper arms slightly toward their bone so the knit reads as fabric, not flesh.
    arm_segs = (
        (Vector((0.168, -0.015, 1.341)), Vector((0.313, -0.013, 1.166))),
        (Vector((-0.168, -0.015, 1.341)), Vector((-0.313, -0.013, 1.166))),
        (Vector((0.313, -0.013, 1.166)), Vector((0.431, -0.176, 1.062))),
        (Vector((-0.313, -0.013, 1.166)), Vector((-0.431, -0.176, 1.062))),
    )
    for v in verts:
        co = v.co
        if abs(co.x) < 0.14:
            continue
        best_proj = None
        best_d = 1e9
        for a, b in arm_segs:
            ab = b - a
            denom = ab.length_squared
            t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, (co - a).dot(ab) / denom))
            proj = a + ab * t
            d = (co - proj).length
            if d < best_d:
                best_d = d
                best_proj = proj
        if best_proj is not None and best_d < 0.04:
            co += (best_proj - co) * 0.10

    mesh.update()
    print("sculpt done")


def curl_fingers(mesh):
    """Keep the MakeHuman hands. A strong scripted curl pinched the fingers."""
    return
    """Gentle rest-pose curl so the selfie hands are not flat paddles."""
    # (name, joints base->tip). Right side is mirrored in X by the caller.
    chains = []
    left = {
        2: [(0.4496, -0.2644, 1.0198), (0.4532, -0.2854, 1.0063), (0.4534, -0.3024, 0.9913), (0.4530, -0.3194, 0.9757)],
        3: [(0.4608, -0.2479, 1.0054), (0.4712, -0.2691, 0.9840), (0.4747, -0.2856, 0.9656), (0.4784, -0.3015, 0.9471)],
        4: [(0.4696, -0.2309, 0.9993), (0.4811, -0.2461, 0.9785), (0.4859, -0.2592, 0.9594), (0.4915, -0.2717, 0.9391)],
        5: [(0.4745, -0.2124, 0.9932), (0.4807, -0.2221, 0.9750), (0.4830, -0.2281, 0.9623), (0.4857, -0.2353, 0.9457)],
        1: [(0.4195, -0.2134, 1.0474), (0.4095, -0.2373, 1.0424), (0.3984, -0.2682, 1.0278), (0.3917, -0.2876, 1.0168)],
    }
    for f, joints in left.items():
        ang = 0.22 if f > 1 else 0.18
        chains.append(([Vector(p) for p in joints], ang))
        chains.append(([Vector((-p[0], p[1], p[2])) for p in joints], ang))

    verts = mesh.vertices
    for joints, angle in chains:
        origins = [p.copy() for p in joints]
        # Precompute membership by original positions.
        groups = []
        for ji in range(len(origins) - 1):
            a = joints[ji]
            bdir = joints[ji + 1] - joints[ji]
            if bdir.length < 1e-8:
                groups.append([])
                continue
            members = []
            for v in verts:
                if (v.co - a).dot(bdir) > 0.004 and dist_point_segment(v.co, a, joints[min(ji + 2, len(joints) - 1)] if False else bdir and a):
                    pass
                if (v.co - a).dot(bdir) > 0.003 and dist_point_segment(v.co, joints[ji], joints[-1]) < 0.016:
                    # Keep only verts nearer this finger than a neighbour by using distance to the segment.
                    if dist_point_segment(v.co, joints[ji], joints[ji + 1]) < 0.014 or (
                        ji >= len(origins) - 2 and dist_point_segment(v.co, joints[ji], joints[-1]) < 0.014
                    ):
                        members.append(v)
            # Broader: any vert close to the remaining chain and in front of this joint.
            members = []
            for v in verts:
                along = (v.co - a).dot(bdir.normalized())
                if along <= 0.002:
                    continue
                if dist_point_segment(v.co, joints[ji], joints[-1]) < 0.013:
                    members.append(v)
            groups.append(members)

        for ji in range(len(origins) - 1):
            origin = origins[ji]
            nxt = origins[ji + 1]
            bdir = nxt - origin
            if bdir.length < 1e-8:
                continue
            axis = bdir.normalized().cross(Vector((0.0, -1.0, 0.0)))
            if axis.length < 1e-5:
                axis = bdir.normalized().cross(Vector((0.0, 0.0, 1.0)))
            axis.normalize()
            # Curl fingertips toward the palm. Palm of the hanging arm faces roughly -Y / inward.
            # Positive rotation around axis is checked visually; angle is small.
            q = Quaternion(axis, angle)
            for v in groups[ji]:
                v.co = origin + q @ (v.co - origin)
            for k in range(ji + 1, len(origins)):
                if (origins[k] - origin).dot(bdir) > 0.0:
                    origins[k] = origin + q @ (origins[k] - origin)
    mesh.update()
    print("fingers curled")


def classify(co):
    """Return 'skin' or 'sweater' from rest-pose position."""
    x, y, z = co.x, co.y, co.z
    ax = abs(x)
    # Hands and fingers stay bare. Wrists are near |x|~0.40, z~1.06.
    if ax > 0.395 and z < 1.18:
        return "skin"
    if ax > 0.36 and y < -0.20 and z < 1.16:
        return "skin"
    # Head, face, ears, neck.
    if z > 1.505 and ax < 0.105:
        return "skin"
    if z > 1.430 and ax < 0.080 and y < -0.045:
        return "skin"
    if ax > 0.060 and 1.475 < z < 1.610 and y > -0.115:
        return "skin"
    if 1.392 < z <= 1.505 and ax < 0.058 and y > -0.10:
        return "skin"
    return "sweater"


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def build_pose():
    """Selfie: torso leans toward the camera, head tilts, both arms reach forward.

    Her right arm (negative X, image left) reaches closer to the lens.
    Returns deform closures and a head-local frame in posed space.
    """
    pivot = Vector((0.0, -0.012, 1.125))
    # Lean toward -Y (positive rotation about +X), then a small yaw.
    # Negative yaw turns the face toward her right, where the camera sits.
    lean = math.radians(18.0)
    yaw = math.radians(-6.0)
    q_torso = Quaternion((0.0, 0.0, 1.0), yaw) @ Quaternion((1.0, 0.0, 0.0), lean)
    torso_m = rigid_about(pivot, q_torso)

    head_rest = Vector((0.0, -0.016, 1.514))
    head_pivot = torso_m @ head_rest
    fwd = q_torso @ Vector((0.0, -1.0, 0.0))
    right = q_torso @ Vector((1.0, 0.0, 0.0))
    up = q_torso @ Vector((0.0, 0.0, 1.0))
    # Pitch face back up so she looks slightly down at the lens, then roll
    # the crown toward her right shoulder.
    pitch = math.radians(9.0)
    roll = math.radians(14.0)
    q_extra = Quaternion(fwd, roll) @ Quaternion(right, pitch)
    head_m = Matrix.Translation(head_pivot) @ q_extra.to_matrix().to_4x4() @ Matrix.Translation(-head_pivot) @ torso_m

    q_head = q_extra @ q_torso
    fwd_h = q_head @ Vector((0.0, -1.0, 0.0))
    right_h = q_head @ Vector((1.0, 0.0, 0.0))
    up_h = q_head @ Vector((0.0, 0.0, 1.0))

    def to_local(p):
        d = Vector(p) - head_pivot
        # x her left, y forward, z up
        return Vector((d.dot(right_h), d.dot(fwd_h), d.dot(up_h)))

    # Arms. Lengths from the MakeHuman landmarks.
    sho_l = Vector((0.168, -0.015, 1.341))
    elb_l = Vector((0.313, -0.013, 1.166))
    wri_l = Vector((0.431, -0.176, 1.062))
    tip_l = Vector((0.470, -0.275, 0.970))
    l1 = (elb_l - sho_l).length
    l2 = (wri_l - elb_l).length
    hand_len = (tip_l - wri_l).length

    def ik(side, target, pole):
        sho = torso_m @ (sho_l if side > 0 else Vector((-sho_l.x, sho_l.y, sho_l.z)))
        target = Vector(target)
        pole = Vector(pole)
        dist = (target - sho).length
        max_r = l1 + l2 - 0.004
        min_r = abs(l1 - l2) + 0.004
        dist_c = max(min_r, min(max_r, dist))
        if abs(dist - dist_c) > 1e-5:
            target = sho + (target - sho).normalized() * dist_c
            dist = dist_c
        a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
        h = math.sqrt(max(0.0, l1 * l1 - a * a))
        axis = (target - sho).normalized()
        mid = sho + axis * a
        pole_dir = pole - sho
        pole_dir = pole_dir - axis * pole_dir.dot(axis)
        if pole_dir.length < 1e-6:
            pole_dir = up.cross(axis)
        pole_dir.normalize()
        elbow = mid + pole_dir * h
        wrist = target
        return sho, elbow, wrist

    # Targets are in front of the body, lowered, asymmetric.
    # Her right arm (negative X) reaches closer to the lens, as in the selfie.
    sho_l_p, elb_l_p, wri_l_p = ik(1, (0.20, -0.36, 1.14), (0.36, -0.16, 1.14))
    sho_r_p, elb_r_p, wri_r_p = ik(-1, (-0.04, -0.58, 0.96), (-0.32, -0.28, 1.00))

    def hand_tail(elbow, wrist):
        d = wrist - elbow
        if d.length < 1e-6:
            d = Vector((0.0, -1.0, 0.0))
        return wrist + d.normalized() * hand_len

    bones_rest = {
        "upper.L": (sho_l, elb_l),
        "fore.L": (elb_l, wri_l),
        "hand.L": (wri_l, tip_l),
        "upper.R": (Vector((-sho_l.x, sho_l.y, sho_l.z)), Vector((-elb_l.x, elb_l.y, elb_l.z))),
        "fore.R": (Vector((-elb_l.x, elb_l.y, elb_l.z)), Vector((-wri_l.x, wri_l.y, wri_l.z))),
        "hand.R": (Vector((-wri_l.x, wri_l.y, wri_l.z)), Vector((-tip_l.x, tip_l.y, tip_l.z))),
    }
    bones_pose = {
        "upper.L": (sho_l_p, elb_l_p),
        "fore.L": (elb_l_p, wri_l_p),
        "hand.L": (wri_l_p, hand_tail(elb_l_p, wri_l_p)),
        "upper.R": (sho_r_p, elb_r_p),
        "fore.R": (elb_r_p, wri_r_p),
        "hand.R": (wri_r_p, hand_tail(elb_r_p, wri_r_p)),
    }
    # A little forearm twist so palms turn inward toward the lens.
    twists = {"upper.L": 0.15, "fore.L": 0.45, "hand.L": 0.2,
              "upper.R": -0.15, "fore.R": -0.45, "hand.R": -0.2}
    deforms = {"torso": torso_m, "head": head_m}
    for name, (rh, rt) in bones_rest.items():
        ph, pt = bones_pose[name]
        deforms[name] = swing_matrix(rh, rt, ph, pt, twists.get(name, 0.0))

    print("IK L wrist", tuple(round(c, 3) for c in wri_l_p), "R", tuple(round(c, 3) for c in wri_r_p))
    return deforms, head_pivot, to_local, fwd_h, up_h, right_h


def vertex_weights(co):
    x, y, z = co.x, co.y, co.z
    ax = abs(x)
    # Whole head is rigid on the head bone, including ears and chin.
    if z > 1.445 and ax < 0.095:
        return (("head", 1.0),)
    if z > 1.470 and ax < 0.110 and y > -0.12:
        return (("head", 1.0),)
    if ax > 0.062 and z > 1.48 and y > -0.12:
        return (("head", 1.0),)
    # Neck blend.
    if 1.345 < z <= 1.455 and ax < 0.070 and y > -0.11:
        t = smoother((z - 1.350) / 0.095)
        return (("head", t), ("torso", 1.0 - t))

    side = "L" if x >= 0.0 else "R"
    # Rest-pose arm segments (pre-deform coordinates).
    if side == "L":
        sho, elb, wri, tip = (
            Vector((0.168, -0.015, 1.341)),
            Vector((0.313, -0.013, 1.166)),
            Vector((0.431, -0.176, 1.062)),
            Vector((0.470, -0.275, 0.970)),
        )
    else:
        sho, elb, wri, tip = (
            Vector((-0.168, -0.015, 1.341)),
            Vector((-0.313, -0.013, 1.166)),
            Vector((-0.431, -0.176, 1.062)),
            Vector((-0.470, -0.275, 0.970)),
        )
    d_up = dist_point_segment(co, sho, elb)
    d_fore = dist_point_segment(co, elb, wri)
    d_hand = dist_point_segment(co, wri, tip)
    # Hands stay rigid.
    if d_hand < 0.05 and d_hand <= d_fore + 0.01 and ax > 0.36:
        return ((f"hand.{side}", 1.0),)
    if d_hand < 0.045 and (co - wri).dot((tip - wri)) > 0.0:
        return ((f"hand.{side}", 1.0),)

    d_spine = dist_point_segment(co, Vector((0.0, -0.02, 1.14)), Vector((0.0, -0.02, 1.40)))
    # Shoulders close to the torso stay on the torso so the sweater root does not tear.
    pairs = (
        (f"upper.{side}", d_up),
        (f"fore.{side}", d_fore),
        (f"hand.{side}", d_hand),
        ("torso", d_spine + 0.02),
    )
    weighted = []
    for name, d in pairs:
        if d < 0.12:
            weighted.append((name, 1.0 / ((d * d) + 2.5e-4)))
    if not weighted:
        return (("torso", 1.0),)
    total = sum(w for _, w in weighted)
    out = tuple((n, w / total) for n, w in weighted)
    return out


def apply_pose(mesh, deforms):
    cos = [v.co.copy() for v in mesh.vertices]
    weights = [vertex_weights(co) for co in cos]
    for v, co, wts in zip(mesh.vertices, cos, weights):
        acc = Vector((0.0, 0.0, 0.0))
        for name, w in wts:
            acc += w * (deforms[name] @ co)
        v.co = acc
    mesh.update()
    print("pose applied", len(mesh.vertices))


# ---------------------------------------------------------------------------
# Eyes, lashes, brows
# ---------------------------------------------------------------------------

def make_uv_sphere(name, loc, radius, col, segments=32, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "Mesh"
    shade_smooth(obj)
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def orient_z_to(obj, direction, up):
    fwd = Vector(direction).normalized()
    up = Vector(up)
    z_axis = fwd
    x_axis = up.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0)).cross(z_axis)
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
    loc = obj.location.copy()
    obj.matrix_world = Matrix.Translation(loc) @ rot


def wrap_eyelids(mesh, eye_centers):
    """Settle nearby lid vertices onto a shell just outside each eyeball."""
    radius = 0.0122
    for v in mesh.vertices:
        for center in eye_centers:
            d = v.co - center
            dist = d.length
            if dist > 0.017 or dist < 1e-6:
                continue
            if dist < radius:
                v.co = center + d.normalized() * radius
    mesh.update()


def build_eyes(col, head_m, up, fwd, centers_rest):
    centers = [head_m @ Vector(c) for c in centers_rest]
    gaze = fwd.normalized()
    objs = []
    for i, side in enumerate(("L", "R")):
        eye = make_uv_sphere(f"Eye_{side}", centers[i] - gaze * 0.003, 0.0110, col, 36, 20)
        orient_z_to(eye, gaze, up)
        cornea = make_uv_sphere(f"Cornea_{side}", centers[i] - gaze * 0.0022, 0.0113, col, 32, 16)
        orient_z_to(cornea, gaze, up)
        cornea.visible_shadow = False
        objs.append((eye, cornea))
    return centers, objs


def build_lashes_and_brows(col, head_m, up, fwd, right):
    bm = bmesh.new()

    def add_taper(root, tip, width):
        root = Vector(root)
        tip = Vector(tip)
        direction = tip - root
        if direction.length < 1e-6:
            return
        side = direction.normalized().cross(up)
        if side.length < 1e-6:
            side = direction.normalized().cross(right)
        side.normalize()
        thick = direction.normalized().cross(side).normalized() * width * 0.35
        a = bm.verts.new(root + side * width * 0.5)
        b = bm.verts.new(root - side * width * 0.5)
        c = bm.verts.new(tip + thick * 0.2)
        d = bm.verts.new(root + thick)
        try:
            bm.faces.new((a, b, c))
        except ValueError:
            pass

    # Upper lashes along an arc in rest head space, then posed.
    for side in (1.0, -1.0):
        for i in range(26):
            t = i / 25.0
            ang = vlerp(-1.05, 1.05, t)
            # Rest lid arc.
            root = Vector((
                side * (0.0312 + math.sin(ang) * 0.0125),
                -0.142,
                1.550 + math.cos(ang) * 0.0045,
            ))
            root = head_m @ root
            length = 0.0105 * (0.55 + 0.45 * math.sin(t * math.pi))
            # Curl up and forward.
            tip = root + fwd * (length * 0.85) + up * (length * 0.75) + right * (side * length * 0.15)
            add_taper(root, tip, 0.00045)
        for i in range(14):
            t = i / 13.0
            ang = vlerp(-0.8, 0.8, t)
            root = Vector((
                side * (0.0312 + math.sin(ang) * 0.011),
                -0.140,
                1.540 - 0.001,
            ))
            root = head_m @ root
            length = 0.0032
            tip = root + fwd * length * 0.8 + up * (-length * 0.25)
            add_taper(root, tip, 0.0003)

        # Brow hairs, soft arch.
        for i in range(18):
            t = i / 17.0
            x = side * vlerp(0.014, 0.050, t)
            z = 1.566 + math.sin(t * math.pi) * 0.007 - t * 0.004
            y = -0.132 - math.sin(t * math.pi) * 0.006
            root = head_m @ Vector((x, y, z))
            grow = up * 0.006 + fwd * 0.004 + right * side * 0.0045
            # Several short strokes per slot.
            for k in range(2):
                jitter = right * side * (k - 0.5) * 0.0012
                tip = root + grow + jitter + up * 0.001 * k
                add_taper(root + jitter, tip, 0.00035)

    mesh = bpy.data.meshes.new("LashesBrowsMesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("LashesBrows", mesh)
    col.objects.link(obj)
    shade_smooth(obj)
    return obj


# ---------------------------------------------------------------------------
# Sweater
# ---------------------------------------------------------------------------

def knit_image(name="KnitHeight", size=1024):
    img = bpy.data.images.new(name, width=size, height=size, alpha=False, float_buffer=False)
    pix = [0.42] * (size * size * 4)
    cell_x, cell_y = 36, 22
    for y in range(size):
        row_f = y / cell_y
        row_i = int(row_f)
        ly = row_f - row_i
        shift = 0.5 if (row_i % 2) else 0.0
        for x in range(size):
            col_f = x / cell_x + shift
            lx = col_f - math.floor(col_f)
            dx = lx - 0.50
            dy = ly - 0.48
            radial = math.sqrt(dx * dx + dy * dy)
            dist_arc = abs(radial - 0.30)
            # Legs of the stitch drop into the row below.
            leg = 1.0
            if ly < 0.55:
                leg = min(abs(lx - 0.22), abs(lx - 0.78))
            d = min(dist_arc, leg * 0.85)
            h = max(0.0, 1.0 - d * 7.5)
            h = h * h
            # Fine yarn fiber.
            fiber = 0.04 * math.sin((x + y * 0.37) * 1.7)
            val = 0.42 + 0.58 * h + fiber * h
            val = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
            i = (y * size + x) * 4
            pix[i] = pix[i + 1] = pix[i + 2] = val
            pix[i + 3] = 1.0
    img.pixels.foreach_set(pix)
    img.pack()
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    print("knit image", size)
    return img


def build_sweater(body, regions, col, knit):
    src = body.data
    bm_src = bmesh.new()
    bm_src.from_mesh(src)
    bm_src.verts.ensure_lookup_table()
    bm_src.faces.ensure_lookup_table()
    bm_src.normal_update()

    skin = [regions[v.index] == "skin" for v in bm_src.verts]
    # Faces fully outside the skin, plus a one-ring overlap so the collar and cuffs cover the seam.
    use_face = []
    for f in bm_src.faces:
        flags = [skin[v.index] for v in f.verts]
        use_face.append(not all(flags))
    # Drop faces that are entirely deep in the hands (all verts skin and far out) — already excluded
    # because all-skin faces are not used. Overlap faces have mixed flags.

    bm = bmesh.new()
    vert_map = {}

    def map_vert(v):
        if v.index not in vert_map:
            n = v.normal
            co = v.co + n * 0.011
            # Looser through the torso, snugger at the collar.
            if v.co.z < 1.34 and abs(v.co.x) < 0.16:
                co = v.co + n * 0.016
            if abs(v.co.x) > 0.18:
                co = v.co + n * 0.014
            vert_map[v.index] = bm.verts.new(co)
        return vert_map[v.index]

    for f, keep in zip(bm_src.faces, use_face):
        if not keep:
            continue
        try:
            bm.faces.new(tuple(map_vert(v) for v in f.verts))
        except ValueError:
            pass
    bm.normal_update()

    # Collar roll and cuff ribs, using the copied vert positions.
    # Identify boundary verts (only one face) near the neck and the wrists.
    wrist_l = Vector((0.22, -0.28, 1.12))  # posed approx, refined by proximity to high |x| later
    for v in bm.verts:
        co = v.co
        # Collar: high and central.
        if co.z > 1.36 and abs(co.x) < 0.09:
            # Roll outward and slightly down to suggest a crew neckband.
            horiz = Vector((co.x, co.y, 0.0))
            if horiz.length > 1e-6:
                horiz.normalize()
            w = falloff(abs(co.z - 1.42), 0.05)
            co += horiz * 0.008 * w
            co.z -= 0.004 * w
        # Elbow puff / fold hint: distance in XZ from a posed elbow is unknown exactly;
        # add a soft circumferential wave on the sleeves via the normal.
        if abs(co.x) > 0.16 and co.z < 1.32:
            along = co.z * 40.0 + abs(co.x) * 25.0
            co += v.normal * math.sin(along) * 0.0013

    uv_layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for loop in f.loops:
            co = loop.vert.co
            if abs(co.x) < 0.175:
                ang = math.atan2(co.x, -co.y)
                loop[uv_layer].uv = (ang * 1.4, (co.z - 1.05) * 3.2)
            else:
                side = 1.0 if co.x >= 0 else -1.0
                # Unwrap sleeves along a forward-down axis.
                along = (-co.y) * 2.2 + (1.35 - co.z) * 1.4
                around = math.atan2(co.z - 1.15, co.x * side)
                loop[uv_layer].uv = (around * 0.8 + (0.0 if side > 0 else 3.0), along)

    mesh = bpy.data.meshes.new("SweaterMesh")
    bm.to_mesh(mesh)
    bm.free()
    bm_src.free()
    obj = bpy.data.objects.new("Sweater", mesh)
    col.objects.link(obj)
    shade_smooth(obj)

    sub = obj.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 2

    tex = bpy.data.textures.new("KnitHeight", "IMAGE")
    tex.image = knit
    tex.extension = "REPEAT"
    disp = obj.modifiers.new("KnitDisplace", "DISPLACE")
    disp.texture = tex
    disp.texture_coords = "UV"
    disp.uv_layer = "UVMap"
    disp.strength = 0.0022
    disp.mid_level = 0.42

    solid = obj.modifiers.new("Thickness", "SOLIDIFY")
    solid.thickness = 0.0045
    solid.offset = -1.0
    return obj


# ---------------------------------------------------------------------------
# Hair
# ---------------------------------------------------------------------------

def build_hair(body, col, to_local, fwd, up, right, head_pivot):
    """Full scalp cap plus round hanging curves.

    Local frame: +X her left, +Y forward, +Z up. The face is around local
    y=0.12. Long hair is rooted off the face and falls in world -Z, pushed
    out of the eye/nose/mouth box so it cannot cross the features.
    """
    mesh = body.data
    bm_src = bmesh.new()
    bm_src.from_mesh(mesh)
    bm_src.verts.ensure_lookup_table()
    bm_src.faces.ensure_lookup_table()
    bm_src.normal_update()

    def world_of(lx, ly, lz):
        return head_pivot + right * lx + fwd * ly + up * lz

    def face_box(lx, ly, lz):
        return ly > 0.085 and abs(lx) < 0.058 and lz < 0.052

    cap_bm = bmesh.new()
    copied = {}
    for f in bm_src.faces:
        c = f.calc_center_median()
        lx, ly, lz = to_local(c)
        dist2 = lx * lx + (ly - 0.02) * (ly - 0.02) + (lz - 0.05) * (lz - 0.05)
        if dist2 > 0.17 * 0.17 or lz < -0.02:
            continue
        if face_box(lx, ly, lz):
            continue
        verts = []
        for v in f.verts:
            if v.index not in copied:
                vlx, vly, vlz = to_local(v.co)
                lift = 0.004 + 0.012 * smoother(max(0.0, vlz) / 0.12) + 0.006 * min(1.0, abs(vlx) / 0.07)
                copied[v.index] = cap_bm.verts.new(v.co + v.normal * lift)
            verts.append(copied[v.index])
        try:
            cap_bm.faces.new(tuple(verts))
        except ValueError:
            pass
    cap_mesh = bpy.data.meshes.new("HairCapMesh")
    cap_bm.to_mesh(cap_mesh)
    cap_bm.free()
    cap_obj = bpy.data.objects.new("HairCap", cap_mesh)
    col.objects.link(cap_obj)
    shade_smooth(cap_obj)
    sub = cap_obj.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 1
    cap_tex = bpy.data.textures.new("HairCapNoise", "CLOUDS")
    cap_tex.noise_scale = 0.02
    cap_disp = cap_obj.modifiers.new("CapBreakup", "DISPLACE")
    cap_disp.texture = cap_tex
    cap_disp.strength = 0.0035
    cap_disp.mid_level = 0.5
    print("hair cap faces", len(cap_mesh.polygons))

    state = {"s": 97531}

    def rnd():
        state["s"] = (1103515245 * state["s"] + 12345) & 0x7FFFFFFF
        return state["s"] / 0x7FFFFFFF

    cu = bpy.data.curves.new("HairStrandsData", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.00145
    cu.bevel_resolution = 1
    cu.resolution_u = 2
    cu2 = bpy.data.curves.new("HairFlyawaysData", "CURVE")
    cu2.dimensions = "3D"
    cu2.bevel_depth = 0.00032
    cu2.bevel_resolution = 1
    cu2.resolution_u = 2

    def add_curve(datablock, pts):
        if len(pts) < 3:
            return
        sp = datablock.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            sp.points[i].co = (p.x, p.y, p.z, 1.0)

    def avoid_face(lx, ly, lz, sign):
        if ly > 0.075 and abs(lx) < 0.058 and lz < 0.050:
            lx = sign * max(abs(lx), 0.064)
            ly = min(ly, 0.02)
        return lx, ly, lz

    bangs = []
    roots = []
    for v in bm_src.verts:
        lx, ly, lz = to_local(v.co)
        if lz < -0.02 or lz > 0.16:
            continue
        if 0.062 < lz < 0.108 and ly > 0.06 and abs(lx) < 0.052:
            bangs.append((lx, ly, lz))
        if face_box(lx, ly, lz):
            continue
        r2 = (lx / 0.09) ** 2 + ((ly - 0.01) / 0.10) ** 2 + ((lz - 0.05) / 0.10) ** 2
        if r2 < 1.0 and lz > 0.0:
            roots.append((lx, ly, lz))
    if roots:
        print("hair roots", "bangs", len(bangs), "scalp", len(roots),
              "z", round(min(r[2] for r in roots), 3), round(max(r[2] for r in roots), 3))
    else:
        print("hair roots", "bangs", len(bangs), "scalp", 0)

    bangs.sort(key=lambda t: t[0])
    step = max(1, len(bangs) // 48)
    for lx, ly, lz in bangs[::step]:
        if rnd() < 0.08:
            continue
        length = 0.016 + rnd() * 0.040
        phase = rnd() * 6.0
        bias = 0.014 if lx > 0.006 else (-0.014 if lx < -0.006 else (rnd() - 0.5) * 0.012)
        for child in range(2):
            jx = (rnd() - 0.5) * 0.004
            pts = []
            for s in range(6):
                t = s / 5.0
                nz = max(0.050, lz - length * t)
                ny = ly + 0.004 * math.sin(t * math.pi)
                nx = lx + jx + bias * t + math.sin(phase + t * 4.0) * 0.002
                pts.append(world_of(nx, ny, nz))
            add_curve(cu if child == 0 else cu2, pts)

    roots.sort(key=lambda t: (t[1], t[0]))
    step = max(1, len(roots) // 260)
    for lx0, ly0, lz0 in roots[::step]:
        back = ly0 < 0.0
        crown = lz0 > 0.07
        length = (0.36 + rnd() * 0.12) if back else (0.44 + rnd() * 0.14)
        if lx0 > 0.02 and not back:
            length += 0.05
        phase = rnd() * 8.0
        sign0 = 1.0 if lx0 >= 0.0 else -1.0
        if abs(lx0) < 0.012:
            sign0 = 1.0 if phase > 4.0 else -1.0
        copies = 4 if crown else (3 if not back else 2)
        for child in range(copies):
            jx = (rnd() - 0.5) * 0.008
            jy = (rnd() - 0.5) * 0.006
            sign = sign0 if child % 2 == 0 else (sign0 if abs(lx0) > 0.03 else -sign0)
            pts = []
            steps = 14
            for s in range(steps):
                t = s / (steps - 1)
                fall = t * t
                spread = 1.0 + 0.22 * t
                nx = (lx0 + jx) * spread + math.sin(phase + t * 5.5) * (0.004 + 0.014 * t)
                ny = ly0 + jy - (0.03 if back else 0.012) * t + math.cos(phase * 0.7 + t * 3.5) * 0.01 * t
                nz = lz0 * (1.0 - fall) + (lz0 - length) * fall
                if ny > 0.08 and abs(nx) < 0.05 and nz < 0.05:
                    ny = 0.02
                    nx = sign * max(abs(nx), 0.055)
                pts.append(world_of(nx, ny, nz))
            add_curve(cu, pts)
            if child == 0 and rnd() < 0.7:
                add_curve(cu2, [p + right * ((rnd() - 0.5) * 0.006) + up * 0.002 for p in pts])

    strands = bpy.data.objects.new("HairStrands", cu)
    col.objects.link(strands)
    fly = bpy.data.objects.new("HairFlyaways", cu2)
    col.objects.link(fly)
    bm_src.free()
    print("hair curves", len(cu.splines), len(cu2.splines))
    return cap_obj, strands, fly



# Pixel landmarks measured on input/reference.png (1205x1306, origin top-left).
# Her left eye is image-right and higher because the head rolls toward her right.
PHOTO_W = 1205
PHOTO_H = 1306
PHOTO_EYE_L = (643.0, 309.0)  # +X, her left
PHOTO_EYE_R = (511.0, 386.0)  # -X, her right
PHOTO_NOSE = (605.0, 452.0)
PHOTO_MOUTH = (638.0, 514.0)
KNIT_CROP = (780, 860, 160, 160)  # x, y, w, h in photo pixels, top-left origin


def _photo_uv(px, py):
    return px / PHOTO_W, 1.0 - (py / PHOTO_H)


def camera_matrix_look(loc, target, roll):
    loc = Vector(loc)
    target = Vector(target)
    forward = target - loc
    if forward.length < 1e-8:
        forward = Vector((0.0, -1.0, 0.0))
    forward.normalize()
    z_axis = -forward
    up_guess = Vector((0.0, 0.0, 1.0))
    x_axis = up_guess.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    q = Quaternion(forward, math.radians(roll))
    x_axis = q @ x_axis
    y_axis = q @ y_axis
    z_axis = q @ z_axis
    rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
    return Matrix.Translation(loc) @ rot


def load_reference_buffer():
    path = os.path.join(ROOT, "input", "reference.png")
    img = bpy.data.images.load(path, check_existing=True)
    w, h = img.size
    buf = [0.0] * (w * h * 4)
    img.pixels.foreach_get(buf)
    print("reference loaded", w, h, img.colorspace_settings.name)
    return img, buf, w, h


def sample_ref(buf, w, h, px, py):
    x = int(round(px))
    y = int(round(py))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    by = (h - 1) - y
    i = (by * w + x) * 4
    return (buf[i], buf[i + 1], buf[i + 2])


def pix_lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def is_skin_linear(rgb):
    r, g, b = rgb
    lum = pix_lum(rgb)
    if lum < 0.22 or lum > 0.97:
        return False
    if r < g * 0.98:
        return False
    if (r - g) < 0.035 or (r - g) > 0.48:
        return False
    if (g - b) < 0.012:
        return False
    return True


def is_hair_linear(rgb):
    r, g, b = rgb
    lum = pix_lum(rgb)
    return lum < 0.28 and r > g and g > b * 0.8


def flatten_albedo(rgb, target_lum=0.40):
    lum = pix_lum(rgb)
    if lum < 1e-4:
        return (target_lum, target_lum * 0.72, target_lum * 0.62)
    scale = target_lum / lum
    scale = 0.55 if scale < 0.55 else (1.45 if scale > 1.45 else scale)
    return (min(1.0, rgb[0] * scale), min(1.0, rgb[1] * scale), min(1.0, rgb[2] * scale))


def region_mean(buf, w, h, cx, cy, rad, pred):
    acc = [0.0, 0.0, 0.0]
    n = 0
    y0 = max(0, int(cy - rad))
    y1 = min(h, int(cy + rad))
    x0 = max(0, int(cx - rad))
    x1 = min(w, int(cx + rad))
    for y in range(y0, y1, 2):
        for x in range(x0, x1, 2):
            rgb = sample_ref(buf, w, h, x, y)
            if rgb and pred(rgb):
                acc[0] += rgb[0]
                acc[1] += rgb[1]
                acc[2] += rgb[2]
                n += 1
    if n == 0:
        return None
    return (acc[0] / n, acc[1] / n, acc[2] / n)


def make_knit_color(buf, w, h):
    x0, y0, cw, ch = KNIT_CROP
    # Collect pink yarn pixels and a stand-in for stray hairs.
    pix = []
    pink_acc = [0.0, 0.0, 0.0]
    pink_n = 0
    for y in range(ch):
        row = []
        for x in range(cw):
            rgb = sample_ref(buf, w, h, x0 + x, y0 + y)
            row.append(rgb if rgb else (0.6, 0.3, 0.28))
            if rgb and not is_hair_linear(rgb) and rgb[0] > rgb[1]:
                pink_acc[0] += rgb[0]
                pink_acc[1] += rgb[1]
                pink_acc[2] += rgb[2]
                pink_n += 1
        pix.append(row)
    if pink_n == 0:
        pink = (0.62, 0.32, 0.30)
    else:
        pink = (pink_acc[0] / pink_n, pink_acc[1] / pink_n, pink_acc[2] / pink_n)
    for y in range(ch):
        for x in range(cw):
            if is_hair_linear(pix[y][x]) or pix_lum(pix[y][x]) < 0.15:
                pix[y][x] = pink
    # Remove the low-frequency photo shading so the scene lights the yarn.
    blur = [[(0.0, 0.0, 0.0) for _ in range(cw)] for _ in range(ch)]
    rad = 8
    for y in range(ch):
        for x in range(cw):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for yy in range(max(0, y - rad), min(ch, y + rad + 1), 2):
                for xx in range(max(0, x - rad), min(cw, x + rad + 1), 2):
                    c = pix[yy][xx]
                    acc[0] += c[0]
                    acc[1] += c[1]
                    acc[2] += c[2]
                    n += 1
            blur[y][x] = (acc[0] / n, acc[1] / n, acc[2] / n)
    size = 1024
    out = [0.0] * (size * size * 4)

    def mix4(a, b, t):
        return a * (1.0 - t) + b * t

    for y in range(size):
        fy = (y / size) * (ch - 1)
        y_i = int(fy)
        y_f = fy - y_i
        y_i2 = min(ch - 1, y_i + 1)
        for x in range(size):
            fx = (x / size) * (cw - 1)
            x_i = int(fx)
            x_f = fx - x_i
            x_i2 = min(cw - 1, x_i + 1)
            c00, c10 = pix[y_i][x_i], pix[y_i][x_i2]
            c01, c11 = pix[y_i2][x_i], pix[y_i2][x_i2]
            b00, b10 = blur[y_i][x_i], blur[y_i][x_i2]
            b01, b11 = blur[y_i2][x_i], blur[y_i2][x_i2]
            col = []
            for k in range(3):
                c = mix4(mix4(c00[k], c10[k], x_f), mix4(c01[k], c11[k], x_f), y_f)
                b = mix4(mix4(b00[k], b10[k], x_f), mix4(b01[k], b11[k], x_f), y_f)
                # High pass around the crop's own mean, then retint to that mean.
                val = pink[k] * (c / b) if b > 1e-4 else pink[k]
                col.append(0.0 if val < 0 else (1.0 if val > 1 else val))
            # Feather the tile borders so REPEAT does not show a seam.
            fxn = min(x, size - 1 - x) / 18.0
            fyn = min(y, size - 1 - y) / 18.0
            seam = smoother(min(1.0, min(fxn, fyn)))
            for k in range(3):
                col[k] = col[k] * seam + pink[k] * (1.0 - seam)
            i = (y * size + x) * 4
            out[i] = col[0]
            out[i + 1] = col[1]
            out[i + 2] = col[2]
            out[i + 3] = 1.0
    img = bpy.data.images.new("KnitColor", width=size, height=size, alpha=False, float_buffer=False)
    img.pixels.foreach_set(out)
    img.pack()
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    print("knit color", tuple(round(c, 3) for c in pink))
    return img, pink


def solve_projection_camera(scene, points, targets):
    """Fit a hidden camera so posed landmarks land on the photograph."""
    cam_data = bpy.data.cameras.new("ProjCameraData")
    cam = bpy.data.objects.new("ProjCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.hide_render = True
    cam.data.sensor_width = 36.0
    cam.data.sensor_height = 24.0
    cam.data.sensor_fit = "AUTO"
    saved_x = scene.render.resolution_x
    saved_y = scene.render.resolution_y
    scene.render.resolution_x = PHOTO_W
    scene.render.resolution_y = PHOTO_H

    eye_mid = (points[0] + points[1]) / 2.0
    best = None

    # The photo places the eyes high in the frame, so the camera looks below them.
    weights = (2.2, 2.2, 0.5, 0.9)

    def score(loc, roll, lens, target):
        cam.matrix_world = camera_matrix_look(loc, target, roll)
        cam.data.lens = lens
        bpy.context.view_layer.update()
        err = 0.0
        for (p, (u, v)), wgt in zip(zip(points, targets), weights):
            co = world_to_camera_view(scene, cam, Vector(p))
            if co.z < 0:
                err += 4.0 * wgt
            err += wgt * ((co.x - u) ** 2 + (co.y - v) ** 2)
        return err

    for drop in (0.08, 0.16, 0.26):
        for cx in (-0.06, 0.0, 0.06):
            for cy in (-0.72, -0.55):
                for cz in (1.25, 1.42):
                    for lens in (32.0, 46.0):
                        for roll in (-18.0, -8.0, 2.0):
                            loc = Vector((cx, cy, cz))
                            target = eye_mid + Vector((0.0, 0.0, -drop))
                            err = score(loc, roll, lens, target)
                            if best is None or err < best[0]:
                                best = (err, loc.copy(), roll, lens, target.copy())
    _, loc, roll, lens, target = best
    steps = {
        0: 0.03,
        1: 0.04,
        2: 0.03,
    }
    for _ in range(3):
        improved = True
        guard = 0
        while improved and guard < 24:
            guard += 1
            improved = False
            for axis, delta in steps.items():
                for sign in (-1.0, 1.0):
                    trial = loc.copy()
                    trial[axis] += delta * sign
                    err = score(trial, roll, lens, target)
                    if err < best[0]:
                        best = (err, trial, roll, lens, target.copy())
                        loc = trial
                        improved = True
            for delta in (4.0, 1.5):
                for sign in (-1.0, 1.0):
                    err = score(loc, roll + delta * sign, lens, target)
                    if err < best[0]:
                        roll = roll + delta * sign
                        best = (err, loc.copy(), roll, lens, target.copy())
                        improved = True
            for delta in (4.0, 1.5):
                for sign in (-1.0, 1.0):
                    err = score(loc, roll, lens + delta * sign, target)
                    if err < best[0]:
                        lens = max(18.0, lens + delta * sign)
                        best = (err, loc.copy(), roll, lens, target.copy())
                        improved = True
        steps = {k: v * 0.45 for k, v in steps.items()}

    err, loc, roll, lens, target = best
    cam.matrix_world = camera_matrix_look(loc, target, roll)
    cam.data.lens = lens
    bpy.context.view_layer.update()
    pix_errs = []
    for p, (u, v) in zip(points, targets):
        co = world_to_camera_view(scene, cam, Vector(p))
        pix_errs.append((abs(co.x - u) * PHOTO_W, abs(co.y - v) * PHOTO_H))
    mean_px = sum(a + b for a, b in pix_errs) / (2.0 * len(pix_errs))
    print(
        "projection camera",
        tuple(round(c, 3) for c in loc),
        "roll", round(roll, 2),
        "lens", round(lens, 2),
        "mean_px", round(mean_px, 1),
        "per_point", [(round(a, 1), round(b, 1)) for a, b in pix_errs],
    )
    scene.render.resolution_x = saved_x
    scene.render.resolution_y = saved_y
    return cam, mean_px


def apply_photo_skin(mesh, buf, w, h, scene, cam, to_local, eye_centers, accept_projection):
    """Vertex colors from a masked projection, with photo averages everywhere else.

    Eye sockets are not painted from the photo, so the separate eyeballs are
    not duplicated on the lids. Hair, sweater and background pixels are rejected.
    Grazing and unseen surfaces keep the sampled cheek albedo.
    """
    # Measured cheek and lip, darkened so the scene lights them instead of the photo.
    cheek = (0.30, 0.15, 0.11)
    lip_mean = (0.42, 0.13, 0.10)
    iris = region_mean(
        buf, w, h, PHOTO_EYE_L[0], PHOTO_EYE_L[1], 16,
        lambda rgb: 0.05 < pix_lum(rgb) < 0.45 and rgb[0] > rgb[1] > rgb[2] * 0.7,
    )
    if iris is None:
        iris = (0.20, 0.08, 0.06)
    print("sampled cheek", tuple(round(c, 3) for c in cheek),
          "lip", tuple(round(c, 3) for c in lip_mean),
          "iris", tuple(round(c, 3) for c in iris),
          "project", accept_projection)

    color = ensure_color_attr(mesh, "SkinColor")
    rough = ensure_float_attr(mesh, "SkinRough")
    # Match the portrait aspect used when the camera was solved, or normalized
    # coordinates drift because Blender's sensor fit follows the render aspect.
    saved_x = scene.render.resolution_x
    saved_y = scene.render.resolution_y
    scene.render.resolution_x = PHOTO_W
    scene.render.resolution_y = PHOTO_H
    bpy.context.view_layer.update()
    look = -(cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0)))
    look.normalize()

    used = 0
    for v in mesh.vertices:
        co = v.co
        lx, ly, lz = to_local(co)
        base = [cheek[0], cheek[1], cheek[2], 1.0]
        rgh = 0.42
        near_eye = False
        for center in eye_centers:
            if (co - center).length < 0.020:
                near_eye = True
                break
        lip_w = 0.0
        if abs(lx) < 0.030 and abs(lz + 0.028) < 0.016 and ly > 0.09:
            lip_w = falloff(abs(lx) * 0.6 + abs(lz + 0.028) + max(0.0, 0.10 - ly), 0.02)
        if lip_w > 0.05:
            for k in range(3):
                base[k] = vlerp(base[k], lip_mean[k], min(1.0, lip_w * 1.4))
            rgh = vlerp(rgh, 0.28, lip_w)
        # Ears, unseen, get a little more blood.
        if abs(lx) > 0.055 and -0.01 < lz < 0.08 and ly < 0.06:
            for k, target in enumerate((0.62, 0.34, 0.30)):
                base[k] = vlerp(base[k], target, 0.25)

        if accept_projection and not near_eye and v.normal.length > 0:
            n = v.normal.normalized()
            facing = n.dot(look)
            if facing > 0.42:
                co_ndc = world_to_camera_view(scene, cam, co)
                if 0.0 < co_ndc.x < 1.0 and 0.0 < co_ndc.y < 1.0 and co_ndc.z > 0:
                    px = co_ndc.x * (w - 1)
                    py = (1.0 - co_ndc.y) * (h - 1)
                    rgb = sample_ref(buf, w, h, px, py)
                    if rgb and is_skin_linear(rgb) and not is_hair_linear(rgb):
                        lum = pix_lum(rgb)
                        # Skip photo shadows and near-white highlights so they are not painted on.
                        if 0.32 <= lum <= 0.82 and (rgb[0] - rgb[2]) >= 0.06:
                            alb = flatten_albedo(rgb, 0.28 if lip_w < 0.2 else 0.22)
                            wt = smoother((facing - 0.42) / 0.40) * 0.45
                            if lip_w > 0.35:
                                wt *= 0.25
                            for k in range(3):
                                base[k] = vlerp(base[k], alb[k], wt)
                            used += 1
        # Break up the flat albedo a little. This is not pore detail from the photo.
        grain = 0.5 + 0.5 * math.sin(co.x * 220.0 + co.z * 180.0)
        for k in range(3):
            base[k] = min(1.0, max(0.0, base[k] * (1.0 + (grain - 0.5) * 0.035)))
        color.data[v.index].color = base
        rough.data[v.index].value = rgh
    scene.render.resolution_x = saved_x
    scene.render.resolution_y = saved_y
    print("skin projected verts", used)
    return {"cheek": cheek, "lip": lip_mean, "iris": iris, "projected_verts": used}


def bsdf_of(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.get("Material Output") or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf


def safe_input(bsdf, name, value):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = value
        return True
    print("missing input", name, "on", bsdf.node_tree.name if hasattr(bsdf, "node_tree") else "")
    return False


def material_skin():
    mat = bpy.data.materials.new("Skin")
    bsdf = bsdf_of(mat)
    nt = mat.node_tree
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "SkinColor"
    attr.attribute_type = "GEOMETRY"
    rough = nt.nodes.new("ShaderNodeAttribute")
    rough.attribute_name = "SkinRough"
    rough.attribute_type = "GEOMETRY"
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 90.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.55
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.07
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # Slight albedo break-up.
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    try:
        mix.inputs["Factor"].default_value = 0.06
        nt.links.new(attr.outputs["Color"], mix.inputs["A"])
        nt.links.new(noise.outputs["Color"], mix.inputs["B"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    except Exception:
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    try:
        nt.links.new(rough.outputs["Fac"], bsdf.inputs["Roughness"])
    except Exception:
        safe_input(bsdf, "Roughness", 0.45)
    safe_input(bsdf, "Base Color", (0.45, 0.28, 0.24, 1.0))
    safe_input(bsdf, "Subsurface Weight", 0.18)
    safe_input(bsdf, "Subsurface Scale", 0.008)
    safe_input(bsdf, "Subsurface Radius", (1.0, 0.38, 0.22))
    safe_input(bsdf, "Specular IOR Level", 0.45)
    try:
        mat.use_backface_culling = False
    except Exception:
        pass
    return mat


def add_body_uv(mesh):
    """Cylindrical unwrap so the body stays editable even though skin color is vertex data."""
    if mesh.uv_layers.get("UVMap"):
        return
    uv = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv = (
                (math.atan2(co.x, -co.y) + math.pi) / (2.0 * math.pi),
                (co.z - 0.9) / 0.85,
            )


def paint_skin(mesh):
    color = ensure_color_attr(mesh, "SkinColor")
    rough = ensure_float_attr(mesh, "SkinRough")
    for v in mesh.vertices:
        co = v.co
        base = [0.45, 0.28, 0.24, 1.0]
        rgh = 0.48
        # Lips.
        if abs(co.x) < 0.030 and 1.468 < co.z < 1.498 and co.y < -0.145:
            w = falloff(abs(co.z - 1.483) * 1.4 + max(0.0, co.y + 0.150), 0.016)
            lip = (0.48, 0.20, 0.22, 1.0)
            for k in range(4):
                base[k] = vlerp(base[k], lip[k], w)
            rgh = vlerp(rgh, 0.32, w)
        # Cheek blush.
        for side in (1.0, -1.0):
            w = falloff((co - Vector((side * 0.038, -0.125, 1.498))).length, 0.034)
            blush = (0.62, 0.30, 0.30, 1.0)
            for k in range(3):
                base[k] = vlerp(base[k], blush[k], 0.55 * w)
        # Nose warmth.
        w = falloff((co - Vector((0.0, -0.158, 1.505))).length, 0.018)
        for k, target in enumerate((0.80, 0.48, 0.44)):
            base[k] = vlerp(base[k], target, 0.35 * w)
        # Ears slightly more red (translucent look via albedo; SSS does the rest).
        if abs(co.x) > 0.065 and 1.49 < co.z < 1.59 and co.y > -0.10:
            for k, target in enumerate((0.78, 0.42, 0.40)):
                base[k] = vlerp(base[k], target, 0.35)
        color.data[v.index].color = base
        rough.data[v.index].value = rgh
    print("skin painted")


def material_eye():
    mat = bpy.data.materials.new("Eye")
    bsdf = bsdf_of(mat)
    nt = mat.node_tree
    texco = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(texco.outputs["Object"], sep.inputs["Vector"])
    # Radius in the local XY plane; +Z is the gaze axis.
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
    length = nt.nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    nt.links.new(comb.outputs["Vector"], length.inputs[0])
    # Iris radius ~0.006 of a 0.012 ball, so object-space radius threshold ~0.5 of the sphere radius.
    # Object coords of a unit sphere are already -1..1 relative to the object, and the mesh is scaled
    # by the object scale (primitive radius). TexCoord Object is in object space before scale? 
    # "Object" outputs are in the object's local coordinates, unscaled by parent but INCLUDING the
    # mesh vertex positions which already have the radius. So radius is in metres, ~0.006 for iris.
    iris_r = 0.0068
    pupil_r = 0.0025
    limbus = 0.0074

    def ramp_color(radius_socket):
        # Build color with a map range cascade using Mix nodes.
        return None

    # Use a noise for iris fiber and math nodes to choose the color.
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 40.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(texco.outputs["Object"], noise.inputs["Vector"])

    # Compare radii.
    less_pupil = nt.nodes.new("ShaderNodeMath")
    less_pupil.operation = "LESS_THAN"
    less_pupil.inputs[1].default_value = pupil_r
    nt.links.new(length.outputs["Value"], less_pupil.inputs[0])

    less_iris = nt.nodes.new("ShaderNodeMath")
    less_iris.operation = "LESS_THAN"
    less_iris.inputs[1].default_value = iris_r
    nt.links.new(length.outputs["Value"], less_iris.inputs[0])

    less_limb = nt.nodes.new("ShaderNodeMath")
    less_limb.operation = "LESS_THAN"
    less_limb.inputs[1].default_value = limbus
    nt.links.new(length.outputs["Value"], less_limb.inputs[0])

    # Front hemisphere mask so the back of the eyeball stays sclera.
    front = nt.nodes.new("ShaderNodeMath")
    front.operation = "GREATER_THAN"
    front.inputs[1].default_value = 0.0
    nt.links.new(sep.outputs["Z"], front.inputs[0])

    in_iris = nt.nodes.new("ShaderNodeMath")
    in_iris.operation = "MULTIPLY"
    nt.links.new(less_iris.outputs["Value"], in_iris.inputs[0])
    nt.links.new(front.outputs["Value"], in_iris.inputs[1])

    in_pupil = nt.nodes.new("ShaderNodeMath")
    in_pupil.operation = "MULTIPLY"
    nt.links.new(less_pupil.outputs["Value"], in_pupil.inputs[0])
    nt.links.new(front.outputs["Value"], in_pupil.inputs[1])

    # Colors
    mix_iris = nt.nodes.new("ShaderNodeMix")
    mix_iris.data_type = "RGBA"
    mix_iris.blend_type = "MIX"
    # Factor from noise to vary brown.
    nt.links.new(noise.outputs["Fac"], mix_iris.inputs["Factor"])
    # Linear iris browns measured around the reference catchlights.
    mix_iris.inputs["A"].default_value = (0.09, 0.032, 0.02, 1.0)
    mix_iris.inputs["B"].default_value = (0.26, 0.11, 0.055, 1.0)

    mix_pupil = nt.nodes.new("ShaderNodeMix")
    mix_pupil.data_type = "RGBA"
    nt.links.new(in_pupil.outputs["Value"], mix_pupil.inputs["Factor"])
    nt.links.new(mix_iris.outputs["Result"], mix_pupil.inputs["A"])
    mix_pupil.inputs["B"].default_value = (0.015, 0.012, 0.010, 1.0)

    mix_white = nt.nodes.new("ShaderNodeMix")
    mix_white.data_type = "RGBA"
    nt.links.new(in_iris.outputs["Value"], mix_white.inputs["Factor"])
    mix_white.inputs["A"].default_value = (0.86, 0.84, 0.82, 1.0)
    nt.links.new(mix_pupil.outputs["Result"], mix_white.inputs["B"])

    nt.links.new(mix_white.outputs["Result"], bsdf.inputs["Base Color"])
    safe_input(bsdf, "Roughness", 0.18)
    safe_input(bsdf, "Specular IOR Level", 0.5)
    safe_input(bsdf, "Subsurface Weight", 0.08)
    safe_input(bsdf, "Subsurface Scale", 0.002)
    safe_input(bsdf, "Subsurface Radius", (1.0, 0.4, 0.3))
    return mat


def material_cornea():
    mat = bpy.data.materials.new("Cornea")
    bsdf = bsdf_of(mat)
    safe_input(bsdf, "Base Color", (1.0, 1.0, 1.0, 1.0))
    safe_input(bsdf, "Roughness", 0.03)
    safe_input(bsdf, "Transmission Weight", 1.0)
    safe_input(bsdf, "IOR", 1.376)
    safe_input(bsdf, "Specular IOR Level", 0.5)
    return mat


def material_hair(name, base):
    mat = bpy.data.materials.new(name)
    bsdf = bsdf_of(mat)
    nt = mat.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 4.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    mix.inputs["A"].default_value = base
    highlight = (
        min(1.0, base[0] * 1.7 + 0.05),
        min(1.0, base[1] * 1.6 + 0.03),
        min(1.0, base[2] * 1.4 + 0.02),
        1.0,
    )
    mix.inputs["B"].default_value = highlight
    # Factor is 0..1 from noise; scale it down by multiplying.
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.35
    nt.links.new(noise.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    safe_input(bsdf, "Roughness", 0.38)
    safe_input(bsdf, "Specular IOR Level", 0.45)
    safe_input(bsdf, "Coat Weight", 0.25)
    safe_input(bsdf, "Coat Roughness", 0.3)
    safe_input(bsdf, "Anisotropic", 0.6)
    try:
        mat.use_backface_culling = False
    except Exception:
        pass
    return mat


def material_sweater(knit_height, knit_color):
    mat = bpy.data.materials.new("Sweater")
    bsdf = bsdf_of(mat)
    nt = mat.node_tree
    texco = nt.nodes.new("ShaderNodeTexCoord")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = knit_height
    tex.extension = "REPEAT"
    nt.links.new(texco.outputs["UV"], tex.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.9
    bump.inputs["Distance"].default_value = 0.0014
    nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
    color_map = nt.nodes.new("ShaderNodeMapping")
    color_map.inputs["Scale"].default_value = (3.2, 8.0, 1.0)
    nt.links.new(texco.outputs["UV"], color_map.inputs["Vector"])
    color_tex = nt.nodes.new("ShaderNodeTexImage")
    color_tex.image = knit_color
    color_tex.extension = "REPEAT"
    nt.links.new(color_map.outputs["Vector"], color_tex.inputs["Vector"])
    nt.links.new(color_tex.outputs["Color"], bsdf.inputs["Base Color"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 50.0
    noise.inputs["Detail"].default_value = 2.0
    fuzz = nt.nodes.new("ShaderNodeBump")
    fuzz.inputs["Strength"].default_value = 0.15
    fuzz.inputs["Distance"].default_value = 0.0005
    nt.links.new(noise.outputs["Fac"], fuzz.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], fuzz.inputs["Normal"])
    nt.links.new(fuzz.outputs["Normal"], bsdf.inputs["Normal"])
    # Dusty pink. AgX will lift this; the preview pass retunes if needed.
    safe_input(bsdf, "Base Color", (0.50, 0.30, 0.31, 1.0))
    safe_input(bsdf, "Roughness", 0.78)
    safe_input(bsdf, "Sheen Weight", 0.45)
    safe_input(bsdf, "Sheen Roughness", 0.55)
    safe_input(bsdf, "Specular IOR Level", 0.28)
    try:
        mat.use_backface_culling = False
    except Exception:
        pass
    return mat


def material_simple(name, color, rough=0.6):
    mat = bpy.data.materials.new(name)
    bsdf = bsdf_of(mat)
    safe_input(bsdf, "Base Color", (*color, 1.0))
    safe_input(bsdf, "Roughness", rough)
    return mat


def material_lash():
    mat = bpy.data.materials.new("Lashes")
    bsdf = bsdf_of(mat)
    safe_input(bsdf, "Base Color", (0.04, 0.025, 0.02, 1.0))
    safe_input(bsdf, "Roughness", 0.4)
    try:
        mat.use_backface_culling = False
    except Exception:
        pass
    return mat


# ---------------------------------------------------------------------------
# Lights, cafe suggestion, cameras
# ---------------------------------------------------------------------------

def add_area(col, name, loc, energy, size, color, target):
    bpy.ops.object.light_add(type="AREA", location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.energy = energy
    obj.data.size = size
    obj.data.color = color
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    look_track(obj, target)
    return obj


def build_world(scene):
    world = bpy.data.worlds.new("CafeWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    bg.inputs[0].default_value = (0.55, 0.46, 0.38, 1.0)
    bg.inputs[1].default_value = 0.35
    return bg


def build_environment(col):
    """A few out-of-focus shapes so the beauty frame is not a void. Not a reconstruction."""
    objs = []

    def primitive(kind, name, loc, scale, mat):
        if kind == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        elif kind == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=loc)
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=loc)
        obj = bpy.context.object
        obj.name = name
        obj.scale = Vector(scale)
        obj.data.materials.append(mat)
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        col.objects.link(obj)
        objs.append(obj)
        return obj

    wood = material_simple("Wood", (0.45, 0.28, 0.16), 0.55)
    leaf = material_simple("Leaf", (0.25, 0.38, 0.18), 0.7)
    wall = material_simple("Wall", (0.72, 0.64, 0.55), 0.9)
    pot = material_simple("Pot", (0.75, 0.73, 0.68), 0.6)
    primitive("CUBE", "Table", (0.0, -0.15, 0.92), (1.4, 0.8, 0.04), wood)
    primitive("CUBE", "Wall", (0.0, 1.3, 1.5), (3.0, 0.08, 2.0), wall)
    primitive("CYLINDER", "PlantPot", (-0.85, 0.35, 1.05), (0.28, 0.28, 0.35), pot)
    primitive("ICO", "PlantA", (-0.85, 0.30, 1.45), (0.38, 0.34, 0.42), leaf)
    primitive("ICO", "PlantB", (-0.65, 0.45, 1.25), (0.22, 0.2, 0.25), leaf)
    primitive("CUBE", "Shelf", (0.95, 0.7, 1.35), (0.4, 0.25, 0.9), wood)
    primitive("CYLINDER", "Cup", (0.55, 0.15, 1.02), (0.05, 0.05, 0.08), pot)
    return objs


def build_cameras(col, centers):
    eye_mid = (centers[0] + centers[1]) / 2.0

    def add_cam(name, loc, target, lens=50):
        bpy.ops.object.camera_add(location=loc)
        cam = bpy.context.object
        cam.name = name
        cam.data.lens = lens
        cam.data.sensor_width = 36.0
        cam.data.clip_start = 0.02
        cam.data.clip_end = 50.0
        for c in list(cam.users_collection):
            c.objects.unlink(cam)
        col.objects.link(cam)
        look_track(cam, target)
        return cam

    # Close, slightly low, and toward her right so the reaching sleeve enters frame.
    beauty = add_cam("BeautyCamera", (-0.09, -0.58, 1.27), (-0.01, -0.10, 1.46), lens=28)
    beauty.data.dof.use_dof = True
    focus = bpy.data.objects.new("FocusPoint", None)
    focus.location = eye_mid
    focus.hide_render = True
    bpy.context.scene.collection.objects.link(focus)
    beauty.data.dof.focus_object = focus
    try:
        beauty.data.dof.aperture_fstop = 3.2
    except Exception:
        pass

    # Diagnostic cameras are positioned after bounds are known.
    cams = {
        "beauty": beauty,
        "front": add_cam("Cam_Front", (0.0, -2.2, 1.38), (0.0, 0.0, 1.38), lens=50),
        "three_quarter": add_cam("Cam_ThreeQuarter", (1.5, -1.5, 1.38), (0.0, 0.0, 1.38), lens=50),
        "side": add_cam("Cam_Side", (2.2, 0.0, 1.38), (0.0, 0.0, 1.38), lens=50),
        "back": add_cam("Cam_Back", (0.0, 2.2, 1.38), (0.0, 0.0, 1.38), lens=50),
    }
    return cams, eye_mid


def character_bounds(objects):
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in objects:
        if obj.type not in {"MESH", "CURVE"}:
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, w.x)
            mins.y = min(mins.y, w.y)
            mins.z = min(mins.z, w.z)
            maxs.x = max(maxs.x, w.x)
            maxs.y = max(maxs.y, w.y)
            maxs.z = max(maxs.z, w.z)
    return mins, maxs


def frame_diagnostic(cams, mins, maxs):
    center = (mins + maxs) / 2.0
    extent = maxs - mins
    # Orthographic scale must cover the largest axis. Arms reach toward -Y.
    scale = max(extent.x, extent.y, extent.z) * 1.18
    center.z = (mins.z + maxs.z) / 2.0
    placements = {
        "front": Vector((0.0, -1.0, 0.0)),
        "three_quarter": Vector((0.707, -0.707, 0.0)),
        "side": Vector((1.0, 0.0, 0.0)),
        "back": Vector((0.0, 1.0, 0.0)),
    }
    for key, direction in placements.items():
        cam = cams[key]
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = scale
        cam.location = center + direction.normalized() * 2.4
        look_track(cam, center)
    print("bounds", tuple(round(c, 3) for c in mins), tuple(round(c, 3) for c in maxs), "ortho", round(scale, 3))
    return center, scale


# ---------------------------------------------------------------------------
# Render + report
# ---------------------------------------------------------------------------

def configure_cycles(scene, samples):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        bpy.context.view_layer.cycles.use_denoising = True
    except Exception:
        pass


def render_still(scene, cam, path, resx, resy, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.camera = cam
    scene.render.resolution_x = resx
    scene.render.resolution_y = resy
    configure_cycles(scene, samples)
    scene.render.filepath = path
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    print("rendered", path)


def evaluated_vertex_count(objects):
    dg = bpy.context.evaluated_depsgraph_get()
    total = 0
    per = {}
    for obj in objects:
        if obj.type != "MESH":
            # Curves with bevel evaluate to a mesh in the depsgraph.
            try:
                ev = obj.evaluated_get(dg)
                if ev.type == "MESH":
                    n = len(ev.data.vertices)
                else:
                    me = ev.to_mesh()
                    n = len(me.vertices)
                    ev.to_mesh_clear()
            except Exception:
                n = None
        else:
            ev = obj.evaluated_get(dg)
            n = len(ev.data.vertices)
        per[obj.name] = n
        if n:
            total += n
    return total, per


def write_report(scene, objects, mins, maxs, total_eval, per_obj, preview, info):
    info = info or {}
    report = {
        "blender_version": bpy.app.version_string,
        "render_engine": "CYCLES",
        "render_device": "CPU",
        "threads": 3,
        "samples": 8 if preview else 32,
        "denoising": True,
        "view_transform": scene.view_settings.view_transform,
        "objects": [o.name for o in objects],
        "important_objects": [
            "Body", "Sweater", "Eye_L", "Eye_R", "Cornea_L", "Cornea_R",
            "LashesBrows", "HairCap", "HairStrands", "HairFlyaways",
            "BeautyCamera", "Cam_Front", "Cam_ThreeQuarter", "Cam_Side", "Cam_Back",
        ],
        "character_bounds_m": {
            "min": [round(mins.x, 4), round(mins.y, 4), round(mins.z, 4)],
            "max": [round(maxs.x, 4), round(maxs.y, 4), round(maxs.z, 4)],
        },
        "base_mesh_vertices": 6093,
        "base_mesh_faces": 6044,
        "body_cage_vertices": len(bpy.data.objects["Body"].data.vertices),
        "evaluated_vertices_total": total_eval,
        "evaluated_vertices_by_object": per_obj,
        "hair_curves_native": info.get("hair_curves"),
        "generated_textures": info.get("textures", []),
        "renders": [] if preview else [
            "renders/beauty.png",
            "renders/front.png",
            "renders/three-quarter.png",
            "renders/side.png",
            "renders/back.png",
        ],
        "reference_image_accessed": True,
        "reference_note": info.get("reference_note"),
        "projection_mean_error_px": info.get("projection_error_px"),
        "limitations": info.get("limitations", []),
    }
    with open(os.path.join(ROOT, "scene-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("report written")


def main():
    args = extra_args()
    preview = "--preview" in args
    print("preview" if preview else "final", "args", args)

    scene = reset_scene()
    col_char = new_collection("Character")
    col_cloth = new_collection("Clothing")
    col_eyes = new_collection("Eyes")
    col_hair = new_collection("Hair")
    col_lights = new_collection("Lights")
    col_env = new_collection("Environment")
    col_cams = new_collection("Cameras")

    body = import_body()
    # Move body into Character.
    for c in list(body.users_collection):
        c.objects.unlink(body)
    col_char.objects.link(body)

    sculpt_likeness(body.data)
    try:
        curl_fingers(body.data)
    except Exception:
        traceback.print_exc()
        print("finger curl failed, continuing with rest fingers")

    regions = [classify(v.co) for v in body.data.vertices]
    print("regions", "skin", regions.count("skin"), "sweater", regions.count("sweater"))

    deforms, head_pivot, to_local, fwd, up, right = build_pose()
    apply_pose(body.data, deforms)

    # Consistent outward normals after the deform.
    set_active(body)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    add_body_uv(body.data)

    sub = body.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 2

    # Slightly forward of the joint so the cornea clears the lid in profile.
    eye_rest = ((0.0308, -0.1285, 1.5450), (-0.0308, -0.1285, 1.5450))
    centers, eye_pairs = build_eyes(col_eyes, deforms["head"], up, fwd, eye_rest)
    wrap_eyelids(body.data, centers)
    # The wrap opens a round socket. Ease the lids back down so the iris is not startled.
    for v in body.data.vertices:
        for center in centers:
            d = v.co - center
            if d.length > 0.02 or d.length < 1e-6:
                continue
            along = d.dot(up)
            if along > 0.0008:
                v.co -= up * 0.0065 * falloff(d.length, 0.02)
            elif along < -0.0003:
                v.co += up * 0.0032 * falloff(d.length, 0.02)
    body.data.update()
    lashes = build_lashes_and_brows(col_eyes, deforms["head"], up, fwd, right)

    ref_img, ref_buf, ref_w, ref_h = load_reference_buffer()

    def landmark_near(target_local):
        best = None
        want = Vector(target_local)
        for v in body.data.vertices:
            lx, ly, lz = to_local(v.co)
            d = (Vector((lx, ly, lz)) - want).length
            if best is None or d < best[0]:
                best = (d, v.co.copy())
        return best[1]

    nose_co = landmark_near((0.0, 0.145, 0.0))
    mouth_co = landmark_near((0.0, 0.135, -0.030))
    eye_front = [centers[i] + fwd * 0.011 for i in (0, 1)]
    proj_points = [eye_front[0], eye_front[1], nose_co, mouth_co]
    proj_targets = [
        _photo_uv(*PHOTO_EYE_L),
        _photo_uv(*PHOTO_EYE_R),
        _photo_uv(*PHOTO_NOSE),
        _photo_uv(*PHOTO_MOUTH),
    ]
    proj_cam, proj_err = solve_projection_camera(scene, proj_points, proj_targets)
    skin_info = apply_photo_skin(
        body.data, ref_buf, ref_w, ref_h, scene, proj_cam, to_local, centers,
        accept_projection=(proj_err < 45.0),
    )
    skin_info["projection_error_px"] = proj_err
    knit_color, knit_pink = make_knit_color(ref_buf, ref_w, ref_h)
    bpy.data.images.remove(ref_img)

    sub.levels = 2
    knit = knit_image()
    sweater = build_sweater(body, regions, col_cloth, knit)
    cap, strands, fly = build_hair(body, col_hair, to_local, fwd, up, right, head_pivot)

    skin = material_skin()
    eye_mat = material_eye()
    cornea_mat = material_cornea()
    hair_dark = material_hair("Hair", (0.11, 0.043, 0.028, 1.0))
    hair_cap_mat = material_hair("HairCap", (0.09, 0.034, 0.020, 1.0))
    sweater_mat = material_sweater(knit, knit_color)
    lash_mat = material_lash()

    body.data.materials.append(skin)
    sweater.data.materials.append(sweater_mat)
    cap.data.materials.append(hair_cap_mat)
    strands.data.materials.append(hair_dark)
    fly.data.materials.append(hair_dark)
    lashes.data.materials.append(lash_mat)
    for eye, cornea in eye_pairs:
        eye.data.materials.append(eye_mat)
        cornea.data.materials.append(cornea_mat)

    bg = build_world(scene)
    # Large window from her right (image left), warm and soft, like the cafe.
    add_area(col_lights, "KeyLight", (-0.65, -0.75, 1.70), 16.0, 1.2, (1.0, 0.93, 0.85), (0.0, -0.08, 1.50))
    add_area(col_lights, "FillLight", (0.50, -0.60, 1.28), 6.0, 1.4, (1.0, 0.94, 0.90), (0.0, -0.06, 1.45))
    add_area(col_lights, "RimLight", (0.10, 0.50, 1.68), 8.0, 0.55, (1.0, 0.86, 0.74), (0.0, 0.0, 1.55))
    env_objs = build_environment(col_env)

    char_objs = [body, sweater, cap, strands, fly, lashes] + [o for pair in eye_pairs for o in pair]
    cams, eye_mid = build_cameras(col_cams, centers)
    bpy.context.view_layer.update()
    mins, maxs = character_bounds(char_objs)
    frame_diagnostic(cams, mins, maxs)
    scene.camera = cams["beauty"]

    # Save before render so a render failure keeps the geometry.
    for image in bpy.data.images:
        if image.packed_file is None and image.source == "FILE":
            try:
                image.pack()
            except Exception:
                pass
    blend_path = os.path.join(ROOT, "character.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    print("saved", blend_path)

    try:
        total_eval, per_obj = evaluated_vertex_count(char_objs)
    except Exception:
        traceback.print_exc()
        total_eval, per_obj = None, {}
    print("evaluated", total_eval, per_obj)
    fly_pts = sum(len(sp.points) for sp in fly.data.splines) if fly.type == "CURVE" else None
    info = {
        "hair_curves": {
            "HairStrands": {
                "splines": len(strands.data.splines),
                "points": sum(len(sp.points) for sp in strands.data.splines),
                "bevel_depth_m": strands.data.bevel_depth,
            },
            "HairFlyaways": {
                "splines": len(fly.data.splines),
                "points": fly_pts,
                "bevel_depth_m": fly.data.bevel_depth,
                "note": "Native poly curve points. Beveled evaluated meshes are also listed under evaluated_vertices_by_object.",
            },
        },
        "textures": [
            {
                "name": "KnitHeight",
                "size": 1024,
                "source": "procedural",
                "projection": "sweater UVMap, repeat",
                "description": "Procedural stockinette height, packed, drives sweater displacement and bump.",
            },
            {
                "name": "KnitColor",
                "size": 1024,
                "source": "input/reference.png crop x=780 y=860 w=160 h=160",
                "projection": "sweater UVMap with mapping scale (3.2, 8.0, 1), repeat",
                "description": "Dusty-pink knit patch. Hair-colored pixels replaced, low-frequency photo light divided out, edges feathered for tiling.",
            },
            {
                "name": "SkinColor",
                "size": None,
                "source": "input/reference.png",
                "projection": "hidden ProjCamera solved to eye, nose and mouth landmarks; vertex colors on Body, not a UV image",
                "description": "Front-facing skin samples with hair, sweater and eye-socket pixels rejected, luminance flattened toward a cheek albedo. Sides, ears, neck and scalp use the sampled cheek color. Lips use the sampled lip color.",
            },
        ],
        "reference_note": (
            "input/reference.png was opened as a decoded image (direct visual read of the cafe selfie, plus numerical pixel sampling of eyes, lips, cheek and knit). "
            "Colors and a masked projection come from that file. It is not a background plate."
        ),
        "projection_error_px": skin_info.get("projection_error_px"),
        "limitations": [
            "Not a scan. The head is a sculpt on MakeHuman hm08 with a masked, lighting-flattened projection of the reference onto front-facing skin vertices.",
            "Eye alignment of that projection is about 5 pixels; the mouth landmark is about 50 pixels off, so lip color is mostly the sampled average rather than a pasted mouth.",
            "The scalp cap is still a smooth surface under the strands and reads as a wig base. Hair is about a thousand beveled curves, not a full groom, and the waves are inferred.",
            "Eyes are spheres sunk into sculpted lids, with a procedural iris using colors sampled from the photograph. Photo eyelids are not projected, to avoid double eyes.",
            "Side, back, ears, and sweater folds away from the camera are inferred. Hands stay MakeHuman topology.",
            "The sweater color is a feathered crop of the reference knit; stitch relief is a procedural height map. The shoulder and neck boundaries can show gaps.",
            "The cafe is a few primitives, not the photographed room, and is hidden in the turnaround views.",
            "The portrait is still visibly CG.",
        ],
    }

    samples_preview = 8
    if preview:
        os.makedirs(os.path.join(ROOT, "preview"), exist_ok=True)
        render_still(scene, cams["beauty"], os.path.join(ROOT, "preview", "beauty.png"), 480, 520, samples_preview)
        # Hide the cafe so diagnostic previews read the character.
        for o in env_objs:
            o.hide_render = True
        bg.inputs[0].default_value = (0.78, 0.78, 0.80, 1.0)
        bg.inputs[1].default_value = 0.9
        render_still(scene, cams["front"], os.path.join(ROOT, "preview", "front.png"), 420, 420, samples_preview)
        render_still(scene, cams["three_quarter"], os.path.join(ROOT, "preview", "three-quarter.png"), 420, 420, samples_preview)
        render_still(scene, cams["side"], os.path.join(ROOT, "preview", "side.png"), 420, 420, samples_preview)
        render_still(scene, cams["back"], os.path.join(ROOT, "preview", "back.png"), 420, 420, samples_preview)
        for o in env_objs:
            o.hide_render = False
        # Resave with cameras framed.
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
        write_report(scene, char_objs, mins, maxs, total_eval, per_obj, True, info)
        return

    os.makedirs(os.path.join(ROOT, "renders"), exist_ok=True)
    render_still(scene, cams["beauty"], os.path.join(ROOT, "renders", "beauty.png"), 960, 1040, 32)
    for o in env_objs:
        o.hide_render = True
    bg.inputs[0].default_value = (0.80, 0.80, 0.82, 1.0)
    bg.inputs[1].default_value = 1.0
    render_still(scene, cams["front"], os.path.join(ROOT, "renders", "front.png"), 768, 768, 32)
    render_still(scene, cams["three_quarter"], os.path.join(ROOT, "renders", "three-quarter.png"), 768, 768, 32)
    render_still(scene, cams["side"], os.path.join(ROOT, "renders", "side.png"), 768, 768, 32)
    render_still(scene, cams["back"], os.path.join(ROOT, "renders", "back.png"), 768, 768, 32)
    for o in env_objs:
        o.hide_render = False
    bg.inputs[0].default_value = (0.55, 0.46, 0.38, 1.0)
    bg.inputs[1].default_value = 0.35
    scene.camera = cams["beauty"]
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    write_report(scene, char_objs, mins, maxs, total_eval, per_obj, False, info)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
