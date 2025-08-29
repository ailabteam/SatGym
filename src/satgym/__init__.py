# src/satgym/__init__.py
from pathlib import Path
import logging

# --- ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN CỐT LÕI MÀ CÁC MODULE KHÁC CÓ THỂ SỬ DỤNG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STARPERF_PATH = PROJECT_ROOT / "deps" / "StarPerf_Simulator"
UAV_REPO_PATH = PROJECT_ROOT / "deps" / "UAV_Obstacle_Avoiding_DRL"

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Chúng ta sẽ thêm các lệnh register sau khi các môi trường được tạo
# from gymnasium.envs.registration import register
# ...
