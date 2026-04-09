"""
CARLA 자동 데이터 수집 스크립트
================================
실행 방법:
    1. CARLA 서버 먼저 실행: CarlaUE4.exe -quality-level=Low
    2. 이 스크립트 실행:
       conda activate carla_env
       python carla_data_collector.py --frames 500 --town Town01

저장 구조:
    data_collection/carla_dataset/
        images/     ← RGB 이미지 (.jpg)
        depth/      ← 깊이 맵 (.npy, 미터 단위)
        semantic/   ← 시맨틱 마스크 (.png, 클래스 ID)
        labels/     ← YOLO 형식 GT 바운딩 박스 (.txt)
        metadata.json ← 카메라 파라미터 + 수집 설정
"""

import carla
import numpy as np
import cv2
import json
import time
import argparse
import queue
import threading
from pathlib import Path
from datetime import datetime

from carla_utils import (
    build_camera_intrinsics,
    depth_to_meters,
    semantic_image_to_mask,
    carla_image_to_bgr,
    get_actor_bboxes_2d,
    save_yolo_annotation,
    semantic_mask_to_rgb,
)

# ── 설정 ────────────────────────────────────────────────────────────
IMAGE_W   = 1280
IMAGE_H   = 720
FOV       = 90
SAVE_DIR  = Path(__file__).parent / 'carla_dataset'


def setup_sensors(world, vehicle, sensor_queue):
    """RGB / Depth / Semantic 카메라 생성 후 차량에 부착"""
    bp_lib = world.get_blueprint_library()
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    sensors = {}

    def make_camera(bp_name, tag):
        bp = bp_lib.find(bp_name)
        bp.set_attribute('image_size_x', str(IMAGE_W))
        bp.set_attribute('image_size_y', str(IMAGE_H))
        bp.set_attribute('fov',          str(FOV))
        sensor = world.spawn_actor(bp, cam_transform, attach_to=vehicle)
        sensor.listen(lambda img, t=tag: sensor_queue.put((t, img)))
        return sensor

    sensors['rgb']      = make_camera('sensor.camera.rgb',                   'rgb')
    sensors['depth']    = make_camera('sensor.camera.depth',                  'depth')
    sensors['semantic'] = make_camera('sensor.camera.semantic_segmentation',  'semantic')

    return sensors


def collect_synchronized_frame(sensor_queue, timeout=2.0):
    """세 카메라의 동일 timestamp 프레임 수집"""
    frame_data = {}
    deadline = time.time() + timeout
    while len(frame_data) < 3 and time.time() < deadline:
        try:
            tag, img = sensor_queue.get(timeout=0.1)
            frame_data[tag] = img
        except queue.Empty:
            continue
    return frame_data if len(frame_data) == 3 else None


def save_frame(frame_idx, frame_data, K, world, sensors, save_dirs):
    """단일 프레임 데이터 저장"""
    stem = f'{frame_idx:06d}'

    # RGB 이미지
    bgr = carla_image_to_bgr(frame_data['rgb'])
    cv2.imwrite(str(save_dirs['images'] / f'{stem}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Depth 맵 (미터 단위 numpy)
    depth_m = depth_to_meters(frame_data['depth'])
    np.save(str(save_dirs['depth'] / f'{stem}.npy'), depth_m)

    # Semantic 마스크 (클래스 ID)
    sem_mask = semantic_image_to_mask(frame_data['semantic'])
    cv2.imwrite(str(save_dirs['semantic'] / f'{stem}.png'), sem_mask)

    # GT 바운딩 박스
    bboxes = get_actor_bboxes_2d(world, sensors['rgb'], K, IMAGE_W, IMAGE_H)
    save_yolo_annotation(save_dirs['labels'] / f'{stem}.txt', bboxes, IMAGE_W, IMAGE_H)

    return len(bboxes), depth_m.mean()


def main(args):
    # ── 저장 디렉토리 생성 ──
    save_dirs = {
        'images':   SAVE_DIR / 'images',
        'depth':    SAVE_DIR / 'depth',
        'semantic': SAVE_DIR / 'semantic',
        'labels':   SAVE_DIR / 'labels',
    }
    for d in save_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    K = build_camera_intrinsics(IMAGE_W, IMAGE_H, FOV)

    # ── CARLA 연결 ──
    print(f'CARLA 서버 연결 중 (localhost:{args.port})...')
    client = carla.Client('localhost', args.port)
    client.set_timeout(15.0)
    print(f'연결 완료. CARLA 버전: {client.get_server_version()}')

    # 맵 로드
    print(f'맵 로드: {args.town}')
    world = client.load_world(args.town)
    time.sleep(2.0)

    # 날씨 설정
    weather = carla.WeatherParameters(
        cloudiness=20, precipitation=0, sun_altitude_angle=60
    )
    world.set_weather(weather)

    # 동기 모드 (센서 동기화)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    actors_to_cleanup = []
    try:
        # ── 차량 스폰 ──
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        vehicle.set_autopilot(True)
        actors_to_cleanup.append(vehicle)
        print(f'차량 스폰 완료: {vehicle.type_id}')

        # ── Traffic Manager 설정 ──
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)
        vehicle.set_autopilot(True, args.tm_port)

        # ── NPC 차량 스폰 (승용차 + 트럭 + 버스 + 오토바이 + 자전거) ──
        all_vehicle_bps = list(bp_lib.filter('vehicle.*'))
        # 차종별 비율 조절: 4륜 70%, 2륜 30%
        four_wheel = [b for b in all_vehicle_bps if int(b.get_attribute('number_of_wheels')) == 4]
        two_wheel  = [b for b in all_vehicle_bps if int(b.get_attribute('number_of_wheels')) == 2]
        # 4륜 70%, 2륜 30% 비율로 섞기
        import random
        mixed_bps = four_wheel * 7 + two_wheel * 3
        random.shuffle(mixed_bps)

        for i in range(min(args.n_npcs, len(spawn_points) - 1)):
            bp = mixed_bps[i % len(mixed_bps)]
            npc = world.try_spawn_actor(bp, spawn_points[i + 1])
            if npc:
                npc.set_autopilot(True, args.tm_port)
                actors_to_cleanup.append(npc)
        n_spawned = len(actors_to_cleanup) - 1
        print(f'NPC 차량 {n_spawned}대 스폰 (4륜/2륜 혼합)')

        # ── NPC 보행자 스폰 ──
        # walker blueprint 목록
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        # walker controller blueprint
        walker_ctrl_bp = bp_lib.find('controller.ai.walker')

        spawned_walkers = 0
        for _ in range(args.n_pedestrians):
            # 보행자 전용 랜덤 위치 (내비게이션 메시 기반)
            spawn_loc = world.get_random_location_from_navigation()
            if spawn_loc is None:
                continue
            walker_bp = walker_bps[spawned_walkers % len(walker_bps)]
            # 성별/복장 랜덤 설정
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
            if walker is None:
                continue
            actors_to_cleanup.append(walker)
            # AI 컨트롤러 부착 (자율 보행)
            ctrl = world.spawn_actor(walker_ctrl_bp, carla.Transform(), attach_to=walker)
            actors_to_cleanup.append(ctrl)
            spawned_walkers += 1

        # 보행자 AI 활성화 (tick 후에 start 필요)
        world.tick()
        for actor in actors_to_cleanup:
            if 'controller.ai.walker' in actor.type_id:
                actor.start()
                actor.go_to_location(world.get_random_location_from_navigation())
                actor.set_max_speed(1.4)  # 보통 걷는 속도 ~1.4 m/s
        print(f'NPC 보행자 {spawned_walkers}명 스폰')

        # ── 센서 생성 ──
        sensor_queue = queue.Queue()
        sensors = setup_sensors(world, vehicle, sensor_queue)
        actors_to_cleanup.extend(sensors.values())
        print('센서 부착 완료 (RGB + Depth + Semantic)')

        # 워밍업 (카메라 안정화)
        print('워밍업 중 (10 tick)...')
        for _ in range(10):
            world.tick()
            while not sensor_queue.empty():
                sensor_queue.get()

        # ── 데이터 수집 루프 ──
        print(f'\n수집 시작: {args.frames}프레임 목표')
        print('-' * 50)
        collected = 0
        tick_count = 0
        start_time = time.time()

        while collected < args.frames:
            world.tick()
            tick_count += 1

            # 매 tick마다 수집하지 않고 N tick마다 저장 (저장 부하 조절)
            if tick_count % args.save_every != 0:
                while not sensor_queue.empty():
                    sensor_queue.get()
                continue

            frame_data = collect_synchronized_frame(sensor_queue)
            if frame_data is None:
                print(f'  [경고] 프레임 {collected} 동기화 실패, 스킵')
                continue

            n_boxes, mean_depth = save_frame(
                collected, frame_data, K, world, sensors, save_dirs
            )
            collected += 1

            if collected % 50 == 0 or collected == 1:
                elapsed = time.time() - start_time
                fps = collected / elapsed
                eta  = (args.frames - collected) / fps if fps > 0 else 0
                print(f'  [{collected:4d}/{args.frames}] bbox={n_boxes}개  '
                      f'mean_depth={mean_depth:.1f}m  '
                      f'FPS={fps:.1f}  ETA={eta:.0f}s')

        # ── 메타데이터 저장 ──
        metadata = {
            'collected_at': datetime.now().isoformat(),
            'n_frames': collected,
            'town': args.town,
            'image_size': [IMAGE_W, IMAGE_H],
            'fov': FOV,
            'camera_intrinsics': K.tolist(),
            'camera_mount': {'x': 1.5, 'z': 2.4},
            'n_npcs': len(actors_to_cleanup) - 1,
            'carla_version': client.get_server_version(),
        }
        with open(SAVE_DIR / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - start_time
        print(f'\n수집 완료!')
        print(f'  총 프레임: {collected}장')
        print(f'  소요 시간: {elapsed:.1f}초 ({collected/elapsed:.1f} FPS)')
        print(f'  저장 경로: {SAVE_DIR.resolve()}')

    finally:
        # ── 정리 ──
        print('\n정리 중...')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in reversed(actors_to_cleanup):
            if actor.is_alive:
                actor.destroy()
        print('완료.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA 자동 데이터 수집')
    parser.add_argument('--frames',     type=int, default=500,  help='수집할 프레임 수')
    parser.add_argument('--town',       type=str, default='Town01', help='CARLA 맵 이름')
    parser.add_argument('--port',       type=int, default=2000, help='CARLA 서버 포트')
    parser.add_argument('--tm_port',    type=int, default=8000, help='Traffic Manager 포트')
    parser.add_argument('--n_npcs',        type=int, default=20,  help='NPC 차량 수')
    parser.add_argument('--n_pedestrians', type=int, default=15,  help='NPC 보행자 수')
    parser.add_argument('--save_every', type=int, default=4,    help='N tick마다 저장 (5fps 기준)')
    args = parser.parse_args()
    main(args)
