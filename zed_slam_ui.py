import json
import os
import threading
import time
import signal
import sys

from queue import Queue

import cv2
import numpy as np
import pyzed.sl as sl
from flask import Flask, Response, render_template, jsonify, request

HOST = '0.0.0.0'
PORT = 5000
MAX_PATH = 3000
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

lock = threading.Lock()
S = {
    'running': True,
    'frame': None,
    'frame_id': 0,
    'depth_frame': None,
    'depth_frame_id': 0,
    'pose': None,
    'path': [],
    'tracking_state': 'OFF',
    'mapping_active': True,
    'mapping_state': 'NOT_ENABLED',
    'pc_vertices': [],
    'reset_pending': False,
    'imu': None,
    'fps': 0.0,
    'camera_ready': False,
    'camera_model': '',
    'serial': '',
    'settings_restart_pending': False,
    'cfg': {
        'init': {
            'camera_resolution': 'VGA',
            'camera_fps': 60,
            'depth_mode': 'NEURAL_LIGHT',
            'depth_minimum_distance': -1,
            'depth_maximum_distance': -1,
            'coordinate_units': 'METER',
            'coordinate_system': 'RIGHT_HANDED_Z_UP',
            'camera_disable_self_calib': False,
            'camera_image_flip': 'AUTO',
            'depth_stabilization': 30,
            'enable_image_enhancement': True,
            'sensors_required': False,
        },
        'runtime': {
            'confidence_threshold': 30,
            'texture_confidence_threshold': 100,
            'enable_depth': True,
            'remove_saturated_areas': False,
            'enable_fill_mode': False,
        },
        'tracking': {
            'enable_imu_fusion': True,
            'enable_pose_smoothing': False,
            'set_as_static': False,
            'set_floor_as_origin': False,
            'set_gravity_as_origin': True,
            'depth_min_range': -1,
            'mode': 'GEN_3',
            'enable_area_memory': True,
            'enable_2d_ground_mode': False,
        },
        'mapping': {
            'resolution': 'LOW',
            'range_meter': -1,
            'max_memory_usage': 2048,
            'map_type': 'FUSED_POINT_CLOUD',
            'save_texture': False,
            'use_chunk_only': True,
            'reverse_vertex_order': False,
            'stability_counter': 0,
        },
    },
}


def save_settings():
    with lock:
        data = dict(S['cfg'])
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[settings] save failed: {e}")


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
    except Exception as e:
        print(f"[settings] load failed: {e}")
        return
    with lock:
        for section in ('init', 'runtime', 'tracking', 'mapping'):
            if section in saved:
                for key, value in saved[section].items():
                    if key in S['cfg'].get(section, {}):
                        cur_type = type(S['cfg'][section][key])
                        if cur_type is int:
                            value = int(value)
                        elif cur_type is float:
                            value = float(value)
                        elif cur_type is bool and not isinstance(value, bool):
                            value = str(value).lower() == 'true'
                        S['cfg'][section][key] = value
        print(f"[settings] loaded from {SETTINGS_FILE}")


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen():
        last_id = -1
        while True:
            with lock:
                fid = S['frame_id']
                frame = S['frame'] if fid != last_id else None
            if frame:
                last_id = fid
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            else:
                time.sleep(0.016)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/depth_feed')
def depth_feed():
    def gen():
        last_id = -1
        while True:
            with lock:
                fid = S['depth_frame_id']
                frame = S['depth_frame'] if fid != last_id else None
            if frame:
                last_id = fid
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            else:
                time.sleep(0.016)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with lock:
        return jsonify({
            'pose': S['pose'],
            'tracking_state': S['tracking_state'],
            'mapping_active': S['mapping_active'],
            'mapping_state': S['mapping_state'],
            'imu': S['imu'],
            'fps': S['fps'],
            'camera_ready': S['camera_ready'],
            'camera_model': S['camera_model'],
            'serial': S['serial'],
            'path_length': len(S['path']),
            'restart_pending': S['settings_restart_pending'],
        })


@app.route('/path')
def path():
    with lock:
        return jsonify(S['path'])


@app.route('/pc')
def pc():
    with lock:
        return jsonify({'vertices': S['pc_vertices']})


@app.route('/reset', methods=['POST'])
def reset():
    with lock:
        S['reset_pending'] = True
    return jsonify({'status': 'resetting'})


def _settings_options():
    return {
        'init': {
            'camera_resolution': ['VGA', 'SVGA', 'HD720', 'HD1080', 'HD1200', 'HD2K', 'AUTO'],
            'camera_fps': [0, 15, 30, 60, 100, 120],
            'depth_mode': ['PERFORMANCE', 'QUALITY', 'ULTRA', 'NEURAL', 'NEURAL_LIGHT', 'NEURAL_PLUS', 'NONE'],
            'depth_minimum_distance': {'min': -1, 'max': 10, 'step': 0.1},
            'depth_maximum_distance': {'min': -1, 'max': 40, 'step': 0.5},
            'coordinate_units': ['MILLIMETER', 'CENTIMETER', 'METER', 'INCH', 'FOOT'],
            'coordinate_system': ['IMAGE', 'LEFT_HANDED_Y_UP', 'LEFT_HANDED_Z_UP', 'RIGHT_HANDED_Y_UP', 'RIGHT_HANDED_Z_UP', 'RIGHT_HANDED_Z_UP_X_FWD'],
            'camera_disable_self_calib': 'bool',
            'camera_image_flip': [v for v in dir(sl.FLIP_MODE) if v.isupper()],
            'depth_stabilization': {'min': 0, 'max': 100, 'step': 1},
            'enable_image_enhancement': 'bool',
            'sensors_required': 'bool',
        },
        'runtime': {
            'confidence_threshold': {'min': 0, 'max': 100, 'step': 1},
            'texture_confidence_threshold': {'min': 0, 'max': 200, 'step': 1},
            'enable_depth': 'bool',
            'remove_saturated_areas': 'bool',
            'enable_fill_mode': 'bool',
        },
        'tracking': {
            'enable_imu_fusion': 'bool',
            'enable_pose_smoothing': 'bool',
            'set_as_static': 'bool',
            'set_floor_as_origin': 'bool',
            'set_gravity_as_origin': 'bool',
            'depth_min_range': {'min': -1, 'max': 10, 'step': 0.1},
            'mode': ['GEN_1', 'GEN_2', 'GEN_3'],
            'enable_area_memory': 'bool',
            'enable_2d_ground_mode': 'bool',
        },
        'mapping': {
            'resolution': ['LOW', 'MEDIUM', 'HIGH'],
            'range_meter': {'min': -1, 'max': 20, 'step': 0.5},
            'max_memory_usage': {'min': 256, 'max': 8192, 'step': 256},
            'map_type': ['FUSED_POINT_CLOUD', 'MESH'],
            'save_texture': 'bool',
            'use_chunk_only': 'bool',
            'reverse_vertex_order': 'bool',
            'stability_counter': {'min': 0, 'max': 100, 'step': 1},
        },
    }


def _settings_descriptions():
    return {
        'init': {
            'camera_resolution': 'Sensor output resolution. Higher = more detail but lower max FPS and higher GPU load.',
            'camera_fps': 'Target frame rate. 0 = auto-selects the maximum FPS for the chosen resolution.',
            'depth_mode': 'Depth computation algorithm. NEURAL = best quality. NEURAL_LIGHT = Jetson-optimized. PERFORMANCE = fastest but deprecated. NONE disables depth.',
            'depth_minimum_distance': 'Closest depth returned (-1 = camera default, typically ~0.3m). Cannot exceed 3m.',
            'depth_maximum_distance': 'Farthest depth returned (-1 = camera default, up to 20m). Affects depth map range only.',
            'coordinate_units': 'Unit for all spatial data: depth, point cloud, tracking, and mesh.',
            'coordinate_system': 'Axis convention for positional tracking and 3D measures.',
            'camera_disable_self_calib': 'Skip self-calibration on open. Disable for repeatable calibration across runs; keep enabled for best accuracy.',
            'camera_image_flip': 'Flip images horizontally. AUTO uses IMU gravity to detect orientation. Use ON if camera is mounted upside-down.',
            'depth_stabilization': 'Temporal depth smoothness (0=off, 100=max). Reduces flicker on low-texture surfaces. Enables positional tracking automatically when > 0.',
            'enable_image_enhancement': 'Enhanced Contrast Technology via camera ISP (firmware 1523+). Improves image quality in low-light scenes.',
            'sensors_required': 'Fail camera open if IMU sensors are not detected. Disable to use USB3-only cables without sensor connection.',
        },
        'runtime': {
            'confidence_threshold': 'Depth confidence filter (0=strict, 100=permissive). Lower values remove noisy depth but may create holes.',
            'texture_confidence_threshold': 'Texture-based confidence filter (0=strict, 200=permissive). Removes depth on low-texture areas.',
            'enable_depth': 'Enable or disable depth map computation. Disabling saves GPU resources when depth is not needed.',
            'remove_saturated_areas': 'Mask out overexposed (saturated) pixels from depth computation. Improves depth quality in bright scenes.',
            'enable_fill_mode': 'Interpolate depth values in small holes and gaps. Produces denser depth maps at the cost of accuracy at object edges.',
        },
        'tracking': {
            'enable_imu_fusion': 'Fuse IMU accelerometer/gyroscope data with visual odometry. Provides robust tracking during fast motion or low texture.',
            'enable_pose_smoothing': 'Apply temporal smoothing to camera pose. Reduces jitter but adds a small latency to pose updates.',
            'set_as_static': 'Optimize tracking for a stationary camera. Disables motion compensation, useful for fixed-mount setups.',
            'set_floor_as_origin': 'Use detected floor plane as the world origin (Z=0 or Y=0). Must see the floor at startup.',
            'set_gravity_as_origin': 'Align world frame with gravity using IMU. ZED 2i gravity direction is detected from IMU at startup.',
            'depth_min_range': 'Minimum depth range for features used in tracking (-1 = auto). Reduces influence of very close objects.',
            'mode': 'Tracking algorithm generation. GEN_3 = latest with best accuracy. GEN_1/GEN_2 = legacy modes for compatibility.',
            'enable_area_memory': 'Remember visual landmarks across sessions for relocalization. Useful for returning to a previously mapped area.',
            'enable_2d_ground_mode': 'Constrain tracking to a 2D ground plane (XZ or XY). For wheeled robots or vehicles on flat surfaces.',
        },
        'mapping': {
            'resolution': 'Spatial mapping voxel resolution. LOW = faster, less detailed. HIGH = slower, more detailed mesh/point cloud.',
            'range_meter': 'Maximum mapping range in meters (-1 = auto). Longer range captures more distant geometry but uses more memory.',
            'max_memory_usage': 'Maximum memory budget for the spatial map in MB. Higher = more detail retained. Lower = chunks are recycled sooner.',
            'map_type': 'Output representation. FUSED_POINT_CLOUD = lightweight 3D points. MESH = textured/solid surface reconstruction.',
            'save_texture': 'Apply camera texture to mesh faces (MESH mode only). Produces visually rich models but increases memory and file size.',
            'use_chunk_only': 'Only process the most recently updated map chunks. Reduces CPU load during live streaming of incremental updates.',
            'reverse_vertex_order': 'Flip triangle winding for mesh faces. Needed if the mesh appears inside-out or has incorrect normals in your viewer.',
            'stability_counter': 'Number of observations before a voxel is locked (0 = instant). Higher values reduce noise but delay map convergence.',
        },
    }


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'GET':
        with lock:
            return jsonify({
                'current': S['cfg'],
                'options': _settings_options(),
                'descriptions': _settings_descriptions(),
            })
    data = request.get_json(force=True)
    needs_restart = False
    with lock:
        for section in ('init', 'runtime', 'tracking', 'mapping'):
            if section not in data:
                continue
            for key, value in data[section].items():
                if key in S['cfg'].get(section, {}):
                    if section in ('init', 'tracking', 'mapping'):
                        needs_restart = True
                    S['cfg'][section][key] = value
        if needs_restart:
            S['settings_restart_pending'] = True
    save_settings()
    return jsonify({'status': 'accepted', 'needs_restart': needs_restart})


zed = None


def _init_params(cfg):
    init = sl.InitParameters(
        camera_resolution=getattr(sl.RESOLUTION, cfg['camera_resolution']),
        camera_fps=int(cfg['camera_fps']),
        depth_mode=getattr(sl.DEPTH_MODE, cfg['depth_mode']),
        depth_minimum_distance=cfg['depth_minimum_distance'],
        depth_maximum_distance=cfg['depth_maximum_distance'],
        coordinate_units=getattr(sl.UNIT, cfg['coordinate_units']),
        coordinate_system=getattr(sl.COORDINATE_SYSTEM, cfg['coordinate_system']),
    )
    init.camera_disable_self_calib = cfg['camera_disable_self_calib']
    init.camera_image_flip = getattr(sl.FLIP_MODE, cfg['camera_image_flip'])
    init.depth_stabilization = cfg['depth_stabilization']
    init.enable_image_enhancement = cfg['enable_image_enhancement']
    init.sensors_required = cfg['sensors_required']
    return init


def _tracking_params(cfg):
    p = sl.PositionalTrackingParameters()
    p.enable_imu_fusion = cfg['enable_imu_fusion']
    p.enable_pose_smoothing = cfg['enable_pose_smoothing']
    p.set_as_static = cfg['set_as_static']
    p.set_floor_as_origin = cfg['set_floor_as_origin']
    p.set_gravity_as_origin = cfg['set_gravity_as_origin']
    p.depth_min_range = cfg['depth_min_range']
    p.mode = getattr(sl.POSITIONAL_TRACKING_MODE, cfg['mode'])
    p.enable_area_memory = cfg['enable_area_memory']
    p.enable_2d_ground_mode = cfg['enable_2d_ground_mode']
    return p


def _mapping_params(cfg):
    return sl.SpatialMappingParameters(
        map_type=getattr(sl.SPATIAL_MAP_TYPE, cfg['map_type']),
        resolution=getattr(sl.MAPPING_RESOLUTION, cfg['resolution']),
        max_memory_usage=cfg['max_memory_usage'],
        save_texture=cfg['save_texture'],
        use_chunk_only=cfg['use_chunk_only'],
        reverse_vertex_order=cfg['reverse_vertex_order'],
    )


def encode_loop(raw_queue):
    while True:
        with lock:
            if not S['running']:
                break
        try:
            img = raw_queue.get(timeout=1.0)
        except:
            continue
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ret:
            with lock:
                S['frame'] = jpeg.tobytes()
                S['frame_id'] += 1


def encode_depth_loop(depth_queue):
    while True:
        with lock:
            if not S['running']:
                break
        try:
            img = depth_queue.get(timeout=1.0)
        except:
            continue
        ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ret:
            with lock:
                S['depth_frame'] = jpeg.tobytes()
                S['depth_frame_id'] += 1


def camera_loop(raw_queue, depth_queue):
    global zed
    needs_restart = False

    while True:
        if needs_restart:
            with lock:
                S['settings_restart_pending'] = False
                S['mapping_active'] = False
                S['mapping_state'] = 'NOT_ENABLED'
                S['pc_vertices'] = []
                S['path'] = []
                S['camera_ready'] = False
            needs_restart = False
            if zed is not None:
                try:
                    zed.disable_positional_tracking()
                    zed.disable_spatial_mapping()
                    zed.close()
                except:
                    pass
            zed = None
            time.sleep(0.5)

        with lock:
            if not S['running']:
                if zed is not None:
                    zed.close()
                return
            if S['settings_restart_pending']:
                needs_restart = True
                continue
            init_cfg = dict(S['cfg']['init'])
            tracking_cfg = dict(S['cfg']['tracking'])
            mapping_cfg = dict(S['cfg']['mapping'])

        zed = sl.Camera()
        err = zed.open(_init_params(init_cfg))
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"[cam] open failed: {err}, retrying in 2s")
            time.sleep(2)
            continue

        err = zed.enable_positional_tracking(_tracking_params(tracking_cfg))
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"[cam] tracking enable failed: {err}, retrying in 2s")
            zed.close()
            time.sleep(2)
            continue

        runtime = sl.RuntimeParameters()
        image = sl.Mat()
        depth = sl.Mat()
        zpose = sl.Pose()
        sensors = sl.SensorsData()
        fpc = sl.FusedPointCloud()
        sm = _mapping_params(mapping_cfg)
        zed.enable_spatial_mapping(sm)

        cam_info = zed.get_camera_information()
        with lock:
            S['camera_ready'] = True
            S['camera_model'] = str(cam_info.camera_model)
            S['serial'] = str(cam_info.serial_number)
            S['mapping_active'] = True

        frame_count = 0
        fps_timer = time.time()
        last_pc_update = 0
        inner_restart = False

        while True:
            with lock:
                if not S['running']:
                    inner_restart = True
                    break
                if S['settings_restart_pending']:
                    inner_restart = True
                    needs_restart = True
                    break

            now = time.time()

            with lock:
                rc = S['cfg']['runtime']
            runtime.confidence_threshold = rc['confidence_threshold']
            runtime.texture_confidence_threshold = rc['texture_confidence_threshold']
            runtime.enable_depth = rc['enable_depth']
            runtime.remove_saturated_areas = rc['remove_saturated_areas']
            runtime.enable_fill_mode = rc['enable_fill_mode']

            with lock:
                if S['reset_pending']:
                    S['reset_pending'] = False
                    S['mapping_active'] = False
                    zed.disable_spatial_mapping()
                    zed.reset_positional_tracking(sl.Transform())
                    fpc.clear()
                    S['pc_vertices'] = []
                    S['path'] = []
                    S['mapping_state'] = 'NOT_ENABLED'
                    if zed.enable_spatial_mapping(sl.SpatialMappingParameters(
                            map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD,
                            resolution=sl.MAPPING_RESOLUTION.LOW,
                            max_memory_usage=2048,
                            save_texture=False,
                            use_chunk_only=True,
                        )) <= sl.ERROR_CODE.SUCCESS:
                        S['mapping_active'] = True
                        last_pc_update = 0

            if zed.grab(runtime) > sl.ERROR_CODE.SUCCESS:
                time.sleep(0.001)
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            try:
                raw_queue.put_nowait(image.get_data())
            except:
                pass

            if frame_count % 2 == 0:
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                try:
                    d = depth.get_data()
                    if d.size > 0:
                        d_norm = np.clip(d / 5.0, 0, 1) * 255
                        np.nan_to_num(d_norm, copy=False)
                        d_8u = d_norm.astype(np.uint8)
                        d_color = cv2.applyColorMap(d_8u, cv2.COLORMAP_TURBO)
                        depth_queue.put_nowait(d_color)
                except:
                    pass

            track_state = zed.get_position(zpose)
            trans = zpose.get_translation().get()
            orient = zpose.get_orientation().get()

            imu_data = None
            if zed.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE) <= sl.ERROR_CODE.SUCCESS:
                imu = sensors.get_imu_data()
                acc = [0, 0, 0]
                imu.get_linear_acceleration(acc)
                ang = [0, 0, 0]
                imu.get_angular_velocity(ang)
                imu_data = {
                    'acceleration': [round(float(acc[0]), 2), round(float(acc[1]), 2), round(float(acc[2]), 2)],
                    'angular_velocity': [round(float(ang[0]), 2), round(float(ang[1]), 2), round(float(ang[2]), 2)],
                }

            frame_count += 1
            if now - fps_timer >= 1.0:
                with lock:
                    S['fps'] = round(frame_count / (now - fps_timer), 1)
                frame_count = 0
                fps_timer = now

            tx, ty, tz = float(trans[0]), float(trans[1]), float(trans[2])
            ox, oy, oz, ow = float(orient[0]), float(orient[1]), float(orient[2]), float(orient[3])

            with lock:
                S['pose'] = {
                    'translation': [tx, tz, ty],
                    'orientation': [ox, oz, oy, ow],
                }
                ts = str(track_state).split('.')[-1] if '.' in str(track_state) else str(track_state)
                S['tracking_state'] = ts
                S['imu'] = imu_data

                if track_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    S['path'].append(dict(S['pose']))
                    if len(S['path']) > MAX_PATH:
                        S['path'] = S['path'][-MAX_PATH:]

                if S['mapping_active']:
                    mstate = zed.get_spatial_mapping_state()
                    ms = str(mstate).split('.')[-1] if '.' in str(mstate) else str(mstate)
                    S['mapping_state'] = ms

                    if now - last_pc_update > 2.0:
                        zed.request_spatial_map_async()
                        last_pc_update = now

                    if zed.get_spatial_map_request_status_async() <= sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_spatial_map_async(fpc)
                        try:
                            all_verts = []
                            for ch in fpc.chunks:
                                if len(ch.vertices) > 0:
                                    v = np.asarray(ch.vertices, dtype=np.float32)
                                    v_cvt = np.empty((len(v), 3), dtype=np.float32)
                                    v_cvt[:, 0] = v[:, 0]
                                    v_cvt[:, 1] = v[:, 2]
                                    v_cvt[:, 2] = v[:, 1]
                                    all_verts.append(v_cvt)
                            if all_verts:
                                S['pc_vertices'] = np.concatenate(all_verts, axis=0).round(4).tolist()
                        except:
                            pass

        zed.disable_positional_tracking()
        if S['mapping_active']:
            zed.disable_spatial_mapping()
        zed.close()
        zed = None


def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    load_settings()

    raw_queue: "Queue[np.ndarray]" = Queue(maxsize=2)
    depth_queue: "Queue[np.ndarray]" = Queue(maxsize=2)

    t1 = threading.Thread(target=camera_loop, args=(raw_queue, depth_queue), daemon=True)
    t2 = threading.Thread(target=encode_loop, args=(raw_queue,), daemon=True)
    t3 = threading.Thread(target=encode_depth_loop, args=(depth_queue,), daemon=True)
    t1.start()
    t2.start()
    t3.start()

    time.sleep(2)
    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()
