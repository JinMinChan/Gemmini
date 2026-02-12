"""
OCR 결과 파싱 및 게임 상태 변환 모듈
"""
import os
import cv2
import numpy as np

# detect.py에서 함수들 import
from .detect import (
    detect_option,
    normalize_possible,
    normalize_cost,
    normalize_count,
    read_numeric_text_with_fallback,
    crop_center,
    ANNOTATION
)

# 옵션명 매핑 (OCR 결과 -> UI 표시)
OPTION_NAME_MAP = {
    '공격력': '공격력',
    '추가피해': '추가피해',
    '아군피해강화': '아군피해강화',
    '낙인력': '낙인력',
    '아군공격력강화': '아군공격력강화',
    '보스피해': '보스피해',
    '의지력': '의지력',
    '질서혼돈': '질서혼돈',
    '비용': '비용',
    '상태유지': '상태 유지',
    '새로고침': '새로고침',
    '세로고침': '새로고침',
}

# 카테고리 매핑
CATEGORY_TO_STAT = {
    'willpower': 'willpower',
    'points': 'points',
    'effect1': 'effect1',
    'effect2': 'effect2',
    'special': 'special'
}

class GameStateParser:
    def __init__(self):
        self.last_ocr_result = None
        # NOTE: Saving ROI debug images on every request is expensive and clutters the repo.
        # Enable explicitly only when needed.
        self.debug_save_rois = str(os.getenv("GEMMINI_OCR_DEBUG_ROI", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    
    def parse_ocr_result(self, img):
        """
        이미지에서 OCR로 게임 상태 추출
        Returns: dict with keys: options, possible, cost, count
        """
        try:
            results = {}
            
            img_h, img_w = img.shape[:2]
            scale_x = img_w / ANNOTATION["width"]
            scale_y = img_h / ANNOTATION["height"]
            
            print(f"이미지 크기: {img.shape}")
            print(f"기준 해상도: {ANNOTATION['width']}x{ANNOTATION['height']}")
            print(f"스케일 비율: X={scale_x:.3f}, Y={scale_y:.3f}")
            
            # Optional: ROI debug visualization (disabled by default).
            if self.debug_save_rois:
                try:
                    debug_img = img.copy()
                    for box in ANNOTATION["boxes"]:
                        scale_x_box = img_w / ANNOTATION["width"]
                        scale_y_box = img_h / ANNOTATION["height"]

                        cx = int(box["x"] * scale_x_box)
                        cy = int(box["y"] * scale_y_box)
                        w = int(box["w"] * scale_x_box)
                        h = int(box["h"] * scale_y_box)

                        x1 = max(0, cx - w // 2)
                        y1 = max(0, cy - h // 2)
                        x2 = min(img_w, cx + w // 2)
                        y2 = min(img_h, cy + h // 2)

                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            debug_img,
                            box["label"],
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

                    cv2.imwrite("debug_roi_positions.png", debug_img)
                    print("ROI 위치 디버그 이미지 저장: debug_roi_positions.png")
                except Exception as e:
                    print(f"디버그 이미지 저장 실패: {e}")
            
            # 각 박스별로 OCR 수행
            for box in ANNOTATION["boxes"]:
                try:
                    roi = crop_center(img, box, debug=False)
                    label = box["label"]
                    
                    # ROI 유효성 검사
                    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                        print(f"처리 중: {label} - ROI가 너무 작음 (크기: {roi.shape}), 건너뜀")
                        continue
                    
                    print(f"처리 중: {label} (ROI 크기: {roi.shape})")
                    
                    # Optional: per-ROI image dumps (disabled by default).
                    if self.debug_save_rois:
                        try:
                            cv2.imwrite(f"debug_roi_{label}.png", roi)
                        except Exception:
                            pass
                    
                    if label.startswith("option"):
                        # 옵션 감지
                        option_info = detect_option(roi)
                        print(f"  옵션 감지: {option_info}")
                        results.setdefault("options", []).append(option_info)
                    
                    elif label == "possible":
                        # "X회 가능" 영역은 숫자(왼쪽) + 한글(오른쪽)로 구성되어 OCR이
                        # 한글 글리프까지 같이 읽으면서 0 -> 3 같은 오인식이 자주 발생한다.
                        # 숫자 영역만 잘라서 OCR을 수행한다.
                        num_roi = roi
                        try:
                            h, w = roi.shape[:2]
                            # NOTE: 기존 crop(아이콘+한글 일부 포함)에서는 회색 0회 가능 상태가
                            #       아이콘/한글에 끌려 '302' 같은 형태로 오인식되는 케이스가 있었다.
                            #       아이콘/한글을 최대한 배제하고 숫자 중심부만 취한다.
                            x1 = int(w * 0.30)
                            x2 = int(w * 0.50)
                            y1 = int(h * 0.10)
                            y2 = int(h * 0.95)
                            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                                num_roi = roi[y1:y2, x1:x2]
                        except Exception:
                            num_roi = roi

                        raw = read_numeric_text_with_fallback(num_roi, allowlist="012345", expected_len=1)
                        results["possible"] = normalize_possible(raw)
                        print(f"  possible: {results['possible']}")
                    
                    elif label == "cost":
                        raw = read_numeric_text_with_fallback(roi, allowlist="0189")
                        results["cost"] = normalize_cost(raw)
                        print(f"  cost: {results['cost']}")
                    
                    elif label == "count":
                        raw = read_numeric_text_with_fallback(roi, allowlist="0123456789/", expected_len=3)
                        results["count"] = normalize_count(raw)
                        print(f"  count: {results['count']}")
                
                except Exception as e:
                    print(f"  오류 발생 ({label}): {e}")
                    import traceback
                    traceback.print_exc()

            # 시작 상태에서는 possible을 count 기준으로 고정
            count_text = results.get("count")
            if isinstance(count_text, str) and "/" in count_text:
                try:
                    left_str, right_str = count_text.split("/", 1)
                    left = int(left_str)
                    right = int(right_str)
                    if left == right == 7:
                        results["possible"] = 1
                    elif left == right == 9:
                        results["possible"] = 2
                except Exception:
                    pass
            
            self.last_ocr_result = results
            return results
        
        except Exception as e:
            print(f"OCR 처리 중 전체 오류: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def convert_to_ui_state(self, ocr_result):
        """
        OCR 결과를 UI 업데이트용 상태로 변환
        Returns: dict with UI 변수 업데이트 정보
        """
        ui_state = {
            'options': [],
            'rerolls': 1,  # 기본값 1 (희귀), count에서 재설정됨
            'attempts_left': 0,  # count에서 파싱
            'cost_state': 0,  # cost 값에서 계산
        }
        
        # count / possible 파싱
        count_now = None
        max_attempts = None
        if 'count' in ocr_result and ocr_result['count']:
            try:
                parts = ocr_result['count'].split('/')
                if len(parts) == 2:
                    count_now = int(parts[0])
                    max_attempts = int(parts[1])
                    ui_state['attempts_left'] = count_now
            except:
                pass

        possible_value = ocr_result.get('possible')

        # 시작 턴(7/7 또는 9/9)만 리롤 고정, 진행 중에는 OCR possible 우선
        if count_now is not None and max_attempts is not None and count_now == max_attempts:
            if max_attempts == 7:
                ui_state['rerolls'] = 1
            elif max_attempts == 9:
                ui_state['rerolls'] = 2
            elif isinstance(possible_value, int):
                ui_state['rerolls'] = max(0, possible_value)
        elif isinstance(possible_value, int):
            ui_state['rerolls'] = max(0, possible_value)
        elif max_attempts == 7:
            ui_state['rerolls'] = 1
        elif max_attempts == 9:
            ui_state['rerolls'] = 2
        
        # cost -> gold_state (-1: 0G, 0: 900G, 1: 1800G)
        if 'cost' in ocr_result:
            cost_value = ocr_result['cost']
            if cost_value == 0:
                ui_state['cost_state'] = -1
            elif cost_value == 900:
                ui_state['cost_state'] = 0
            elif cost_value == 1800:
                ui_state['cost_state'] = 1
        
        # count -> attempts_left (위 파싱 실패 시 보조)
        if ui_state['attempts_left'] == 0 and 'count' in ocr_result and ocr_result['count']:
            try:
                parts = ocr_result['count'].split('/')
                if len(parts) == 2:
                    ui_state['attempts_left'] = int(parts[0])
            except:
                pass
        
        # options 변환 - 카테고리 기반으로 표준 UI 텍스트 생성
        if 'options' in ocr_result:
            for opt_info in ocr_result['options']:
                option_name = opt_info.get('option')
                value = opt_info.get('value', 0)
                category = opt_info.get('category')
                position = opt_info.get('position')
                
                # 옵션이 None이면 건너뜀 (인식 실패)
                if option_name is None:
                    continue
                
                # 카테고리 + position 기반으로 표준 UI 텍스트 생성
                if category == 'effect1':
                    is_change_opt = isinstance(option_name, str) and ('변경' in option_name)
                    # position이 right이면 잘못 분류된 것 -> effect2로 수정
                    if position == 'right':
                        if is_change_opt or value == 0:
                            display_text = "부옵션2 변환"
                        else:
                            display_text = f"부옵션2 {value:+d}"
                        standard_name = "부옵션2"
                        print(f"  -> 수정: effect1 + right position => 부옵션2")
                    else:
                        if is_change_opt or value == 0:
                            display_text = "부옵션1 변환"
                        else:
                            display_text = f"부옵션1 {value:+d}"
                        standard_name = "부옵션1"
                elif category == 'effect2':
                    is_change_opt = isinstance(option_name, str) and ('변경' in option_name)
                    # position이 left이면 잘못 분류된 것 -> effect1로 수정
                    if position == 'left':
                        if is_change_opt or value == 0:
                            display_text = "부옵션1 변환"
                        else:
                            display_text = f"부옵션1 {value:+d}"
                        standard_name = "부옵션1"
                        print(f"  -> 수정: effect2 + left position => 부옵션1")
                    else:
                        if is_change_opt or value == 0:
                            display_text = "부옵션2 변환"
                        else:
                            display_text = f"부옵션2 {value:+d}"
                        standard_name = "부옵션2"
                elif category == 'willpower':
                    display_text = f"의지력 {value:+d}"
                    standard_name = "의지력"
                elif category == 'points':
                    # 질서/혼돈 구분 (값의 부호로 판단 가능하지만 일단 통합)
                    display_text = f"질서혼돈 {value:+d}"
                    standard_name = "질서혼돈"
                elif category == 'special':
                    # 특수 옵션 처리
                    if '비용' in option_name:
                        if value > 0:
                            display_text = f"비용 +100%"
                        else:
                            display_text = f"비용 -100%"
                        standard_name = "비용"
                    elif '새로고침' in option_name or '세로고침' in option_name:
                        display_text = f"새로고침 +{value}"
                        standard_name = "새로고침"
                    elif '상태' in option_name:
                        display_text = "상태 유지"
                        standard_name = "상태유지"
                    else:
                        display_text = option_name
                        standard_name = option_name
                else:
                    # unknown 카테고리: OCR 텍스트 + position으로 추정
                    print(f"알 수 없는 카테고리 ({category}), 텍스트+위치로 추정: {option_name}, position={position}")
                    if any(keyword in option_name for keyword in ['공격', '피해', '낙인', '보스']):
                        if position == 'right':
                            display_text = f"부옵션2 {value:+d}"
                            standard_name = "부옵션2"
                        else:
                            display_text = f"부옵션1 {value:+d}"
                            standard_name = "부옵션1"
                    else:
                        # 그래도 모르면 건너뜀
                        print(f"  -> 건너뜀")
                        continue
                
                ui_state['options'].append({
                    'text': display_text,
                    'category': category,
                    'position': position,
                    'raw_option': option_name,
                    'value': value
                })
        
        # option=None인 special 카테고리도 처리 (OCR 인식 실패한 비용 등)
        if 'options' in ocr_result:
            for opt_info in ocr_result['options']:
                if opt_info.get('option') is None and opt_info.get('category') == 'special':
                    raw_ocr = opt_info.get('raw_ocr', '')
                    print(f"option=None인 special 추정: {raw_ocr}")
                    
                    # OCR 텍스트로 추정
                    if '비용' in raw_ocr:
                        if '증가' in raw_ocr:
                            display_text = "비용 +100%"
                        else:
                            display_text = "비용 -100%"
                        ui_state['options'].append({
                            'text': display_text,
                            'category': 'special',
                            'position': 'other',
                            'raw_option': '비용',
                            'value': 100 if '증가' in raw_ocr else -100
                        })
                    elif '상태' in raw_ocr:
                        ui_state['options'].append({
                            'text': '상태 유지',
                            'category': 'special',
                            'position': 'other',
                            'raw_option': '상태유지',
                            'value': 0
                        })
                    elif '새로고침' in raw_ocr or '세로고침' in raw_ocr:
                        import re
                        match = re.search(r'[+\-]?\d+', raw_ocr)
                        val = int(match.group()) if match else 1
                        ui_state['options'].append({
                            'text': f"새로고침 +{val}",
                            'category': 'special',
                            'position': 'other',
                            'raw_option': '새로고침',
                            'value': val
                        })
        
        return ui_state


# 테스트
if __name__ == "__main__":
    parser = GameStateParser()
    
    # 테스트 이미지 로드
    img = cv2.imread("./arkgrid.png")
    if img is not None:
        print("OCR 처리 중...")
        result = parser.parse_ocr_result(img)
        
        print("\n===== OCR 결과 =====")
        for k, v in result.items():
            print(f"{k}: {v}")
        
        print("\n===== UI 상태 변환 =====")
        ui_state = parser.convert_to_ui_state(result)
        for k, v in ui_state.items():
            print(f"{k}: {v}")
