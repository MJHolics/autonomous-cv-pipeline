"""
CARLA 유틸리티 함수 모음
- 카메라 내부 파라미터 생성
- 깊이 인코딩 변환 (CARLA log → meters)
- 3D 바운딩 박스 → 2D 투영
- 시맨틱 세그멘테이션 처리
"""

import numpy as np
import carla
import cv2
from pathlib import Path


# ── CARLA Semantic 클래스 정의 ──────────────────────────────────────
# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
# 0.9.15에서 28(Car specific) 등 확장 태그 추가됨 — check_dataset.py로 실측 확인 필요
CARLA_SEMANTIC = {
    0:  ('Unlabeled',    (0,   0,   0)),
    1:  ('Building',     (70,  70,  70)),
    2:  ('Fence',        (190, 153, 153)),
    3:  ('Other',        (250, 170, 160)),
    4:  ('Pedestrian',   (220, 20,  60)),
    5:  ('Pole',         (153, 153, 153)),
    6:  ('RoadLine',     (157, 234, 50)),
    7:  ('Road',         (128, 64,  128)),
    8:  ('SideWalk',     (244, 35,  232)),
    9:  ('Vegetation',   (107, 142, 35)),
    10: ('Vehicles',     (0,   0,   142)),
    11: ('Wall',         (102, 102, 156)),
    12: ('TrafficSign',  (220, 220, 0)),
    13: ('Sky',          (70,  130, 180)),
    14: ('Ground',       (81,  0,   81)),
    15: ('Bridge',       (150, 100, 100)),
    16: ('RailTrack',    (230, 150, 140)),
    17: ('GuardRail',    (180, 165, 180)),
    18: ('TrafficLight', (250, 170, 30)),
    19: ('Static',       (110, 190, 160)),
    20: ('Dynamic',      (170, 120, 50)),
    21: ('Water',        (45,  60,  150)),
    22: ('Terrain',      (145, 170, 100)),
    # CARLA 0.9.15 확장 태그 (실측으로 확인됨)
    28: ('Car',          (0,   0,   142)),  # Car specific (0.9.15+)
}

# ── CARLA blueprint → COCO 클래스명 매핑 ────────────────────────────
# COCO80 기준: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
BLUEPRINT_TO_CLASS = {
    # 승용차
    'vehicle.tesla':        'car',
    'vehicle.audi':         'car',
    'vehicle.bmw':          'car',
    'vehicle.chevrolet':    'car',
    'vehicle.citroen':      'car',
    'vehicle.dodge':        'car',
    'vehicle.ford':         'car',
    'vehicle.jeep':         'car',
    'vehicle.lincoln':      'car',
    'vehicle.mercedes':     'car',
    'vehicle.mini':         'car',
    'vehicle.mustang':      'car',
    'vehicle.nissan':       'car',
    'vehicle.seat':         'car',
    'vehicle.toyota':       'car',
    'vehicle.volkswagen':   'car',
    # 트럭/밴
    'vehicle.carlamotors':  'truck',
    'vehicle.mitsubishi.fusorosa': 'bus',
    # 오토바이
    'vehicle.harley-davidson': 'motorcycle',
    'vehicle.kawasaki':     'motorcycle',
    'vehicle.yamaha':       'motorcycle',
    'vehicle.vespa':        'motorcycle',
    # 자전거
    'vehicle.bh':           'bicycle',
    'vehicle.diamondback':  'bicycle',
    'vehicle.gazelle':      'bicycle',
}

def get_vehicle_class(blueprint_id: str) -> str:
    """CARLA blueprint ID → COCO 클래스명 반환"""
    for prefix, cls in BLUEPRINT_TO_CLASS.items():
        if blueprint_id.startswith(prefix):
            return cls
    # 바퀴 수로 fallback
    return 'car'

# 자율주행 핵심 클래스 (COCO 클래스와 매핑)
AV_CLASSES = {
    4:  'person',
    10: 'car',
    18: 'traffic light',
    12: 'stop sign',
    28: 'car',  # CARLA 0.9.15 Car specific
}


def build_camera_intrinsics(image_w: int, image_h: int, fov: float) -> np.ndarray:
    """
    CARLA 카메라 내부 파라미터 행렬 K 생성.
    CARLA는 수평 FOV 기준으로 초점 거리 계산.

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    fx = image_w / (2.0 * np.tan(np.radians(fov / 2.0)))
    fy = fx
    cx = image_w / 2.0
    cy = image_h / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    return K


def depth_to_meters(depth_image: carla.Image) -> np.ndarray:
    """
    CARLA Depth Camera 출력 → 실제 거리(미터) 변환.

    CARLA 깊이 인코딩:
        R + G*256 + B*256^2
        ──────────────────  × 1000 (미터)
              256^3 - 1

    반환: (H, W) float32 ndarray, 단위 = 미터
    """
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = array.reshape((depth_image.height, depth_image.width, 4))  # BGRA
    # R, G, B 채널 분리
    B = array[:, :, 0].astype(np.float32)
    G = array[:, :, 1].astype(np.float32)
    R = array[:, :, 2].astype(np.float32)
    # 디코딩
    depth_m = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1) * 1000.0
    return depth_m


def semantic_image_to_mask(sem_image: carla.Image) -> np.ndarray:
    """
    CARLA Semantic Segmentation Image → 클래스 ID 맵.
    CARLA는 R 채널에 클래스 ID 저장.

    반환: (H, W) uint8 ndarray, 값 = 클래스 ID (0~22)
    """
    array = np.frombuffer(sem_image.raw_data, dtype=np.uint8)
    array = array.reshape((sem_image.height, sem_image.width, 4))  # BGRA
    return array[:, :, 2].copy()  # R 채널 = class ID


def semantic_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """클래스 ID 맵 → RGB 컬러 시각화"""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, (_, color) in CARLA_SEMANTIC.items():
        rgb[mask == cls_id] = color
    return rgb


def carla_image_to_bgr(carla_img: carla.Image) -> np.ndarray:
    """CARLA RGB Image → OpenCV BGR ndarray"""
    array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
    array = array.reshape((carla_img.height, carla_img.width, 4))  # BGRA
    return array[:, :, :3][:, :, ::-1].copy()  # BGRA → BGR


def get_actor_bboxes_2d(
    world: carla.World,
    camera: carla.Actor,
    K: np.ndarray,
    image_w: int,
    image_h: int,
    max_distance: float = 50.0
) -> list:
    """
    월드 내 차량/보행자의 3D 바운딩 박스를 카메라 2D 좌표로 투영.

    반환: [{'class': str, 'bbox': [x1,y1,x2,y2], 'distance': float}, ...]
    """
    results = []
    cam_transform = camera.get_transform()
    cam_world_matrix = np.array(cam_transform.get_matrix())

    # 월드 → 카메라 변환 (역행렬)
    world_to_cam = np.linalg.inv(cam_world_matrix)

    actors = world.get_actors()
    vehicles   = actors.filter('vehicle.*')
    pedestrians = actors.filter('walker.pedestrian.*')

    for actor_list, cls_name in [(vehicles, 'car'), (pedestrians, 'person')]:
        for actor in actor_list:
            # 자아 차량 제외 (카메라가 붙어있는 차량)
            if actor.id == camera.parent.id:
                continue

            # 거리 필터
            dist = cam_transform.location.distance(actor.get_location())
            if dist > max_distance:
                continue

            # 3D 바운딩 박스 8개 꼭짓점 추출
            bbox = actor.bounding_box
            verts = bbox.get_world_vertices(actor.get_transform())
            pts_2d = []

            for v in verts:
                # 월드 좌표 → 카메라 좌표
                pt_world = np.array([v.x, v.y, v.z, 1.0])
                pt_cam = world_to_cam @ pt_world

                # CARLA 카메라 좌표계 조정 (y → -y, z → x)
                pt_cam_cv = np.array([pt_cam[1], -pt_cam[2], pt_cam[0]])

                # 뒤에 있는 점 제외
                if pt_cam_cv[2] <= 0:
                    continue

                # 투영
                px = K[0, 0] * pt_cam_cv[0] / pt_cam_cv[2] + K[0, 2]
                py = K[1, 1] * pt_cam_cv[1] / pt_cam_cv[2] + K[1, 2]
                pts_2d.append((px, py))

            if len(pts_2d) < 4:
                continue

            xs = [p[0] for p in pts_2d]
            ys = [p[1] for p in pts_2d]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(image_w, max(xs)), min(image_h, max(ys))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            # 차량은 blueprint로 세부 클래스 분류
            if cls_name == 'car':
                cls_name = get_vehicle_class(actor.type_id)
            results.append({
                'class': cls_name,
                'bbox':  [int(x1), int(y1), int(x2), int(y2)],
                'distance': dist
            })

    return results


def save_yolo_annotation(
    save_path: Path,
    bboxes: list,
    image_w: int,
    image_h: int,
    # COCO80 기준 class ID
    class_map: dict = None
):
    """
    GT 바운딩 박스를 YOLO 형식으로 저장.
    COCO80: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
    """
    if class_map is None:
        class_map = {
            'person':     0,
            'bicycle':    1,
            'car':        2,
            'motorcycle': 3,
            'bus':        5,
            'truck':      7,
        }
    lines = []
    for b in bboxes:
        cls_id = class_map.get(b['class'], -1)
        if cls_id < 0:
            continue
        x1, y1, x2, y2 = b['bbox']
        cx = (x1 + x2) / 2.0 / image_w
        cy = (y1 + y2) / 2.0 / image_h
        w  = (x2 - x1) / image_w
        h  = (y2 - y1) / image_h
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
