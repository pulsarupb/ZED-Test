import threading
import time
import signal
import sys

import cv2
import numpy as np
import pyzed.sl as sl
from flask import Flask, Response, render_template, jsonify

HOST = '0.0.0.0'
PORT = 5000
MAX_PATH = 3000

lock = threading.Lock()
S = {
    'running': True,
    'frame': None,
    'frame_id': 0,
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
}

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


zed = None


def camera_loop():
    global zed
    zed = sl.Camera()

    init = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.VGA,
        depth_mode=sl.DEPTH_MODE.NEURAL_LIGHT,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP,
        camera_disable_self_calib=False,
    )

    err = zed.open(init)
    if err > sl.ERROR_CODE.SUCCESS:
        with lock:
            S['running'] = False
        return

    track_params = sl.PositionalTrackingParameters()
    track_params.enable_imu_fusion = True
    track_params.enable_pose_smoothing = False

    err = zed.enable_positional_tracking(track_params)
    if err > sl.ERROR_CODE.SUCCESS:
        zed.close()
        with lock:
            S['running'] = False
        return

    runtime = sl.RuntimeParameters(confidence_threshold=30)
    image = sl.Mat()
    zpose = sl.Pose()
    sensors = sl.SensorsData()
    fpc = sl.FusedPointCloud()

    cam_info = zed.get_camera_information()
    with lock:
        S['camera_ready'] = True
        S['camera_model'] = str(cam_info.camera_model)
        S['serial'] = str(cam_info.serial_number)

    frame_count = 0
    fps_timer = time.time()
    last_pc_update = 0

    sm = sl.SpatialMappingParameters(
        map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD,
        resolution=sl.MAPPING_RESOLUTION.MEDIUM,
        max_memory_usage=2048,
        save_texture=False,
        use_chunk_only=True,
    )
    if zed.enable_spatial_mapping(sm) <= sl.ERROR_CODE.SUCCESS:
        S['mapping_active'] = True

    while True:
        with lock:
            if not S['running']:
                break

        now = time.time()

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
                if zed.enable_spatial_mapping(sm) <= sl.ERROR_CODE.SUCCESS:
                    S['mapping_active'] = True
                    last_pc_update = 0

        if zed.grab(runtime) > sl.ERROR_CODE.SUCCESS:
            time.sleep(0.001)
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        img = image.get_data()
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret = True
        jpeg = None
        if frame_count % 2 == 0:
            ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 55])

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
            if ret and jpeg is not None:
                S['frame'] = jpeg.tobytes()
                S['frame_id'] += 1
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
                            print(f"[pc] {len(fpc.chunks)} chunks, {len(S['pc_vertices'])} points")
                    except Exception as e:
                        print(f"[pc] error: {e}")

    zed.disable_positional_tracking()
    if S['mapping_active']:
        zed.disable_spatial_mapping()
    zed.close()


def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()

    time.sleep(2)
    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()
