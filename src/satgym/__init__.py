# src/satgym/__init__.py

import logging
from gymnasium.envs.registration import register
from pathlib import Path

# ======================================================================
# --- CÁC HẰNG SỐ TOÀN CỤC CỦA PACKAGE ---
# Định nghĩa các đường dẫn gốc để các module khác có thể import và sử dụng.
# Đây là nơi duy nhất trong thư viện biết về cấu trúc thư mục bên ngoài.
#
# LƯU Ý: File này KHÔNG thay đổi sys.path.
# Trách nhiệm thiết lập môi trường import thuộc về các script thực thi.
# ======================================================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    STARPERF_PATH = PROJECT_ROOT / "deps" / "StarPerf_Simulator"
    UAV_REPO_PATH = PROJECT_ROOT / "deps" / "UAV_Obstacle_Avoiding_DRL"
except NameError:
    # Xảy ra khi chạy trong một số môi trường REPL nhất định
    PROJECT_ROOT, STARPERF_PATH, UAV_REPO_PATH = None, None, None


# Cấu hình logging cơ bản cho toàn bộ thư viện
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ======================================================================
# --- ĐĂNG KÝ CÁC MÔI TRƯỜNG VỚI GYMNASIUM ---
# ======================================================================

# Import các class môi trường
# Đặt trong try...except để không gây lỗi nếu dependency chưa sẵn sàng
try:
    from .envs.routing_env import RoutingEnv
    register(id="SatGym-Routing-v0", entry_point="satgym.envs:RoutingEnv")

    #from .envs.resource_alloc_env import ResourceAllocationEnv
    #register(id="SatGym-ResourceAllocation-v0", entry_point="satgym.envs:ResourceAllocationEnv", max_episode_steps=100)

    #from .envs.handover_env import HandoverEnv
    #register(id="SatGym-Handover-v0", entry_point="satgym.envs:HandoverEnv", max_episode_steps=1000)

    #from .envs.task_offloading_env import TaskOffloadingEnv
    #register(id="SatGym-TaskOffloading-v0", entry_point="satgym.envs:TaskOffloadingEnv", max_episode_steps=50)

    #from .envs.beam_hopping_env import BeamHoppingEnv
    #register(id="SatGym-BeamHopping-v0", entry_point="satgym.envs:BeamHoppingEnv", max_episode_steps=200)

    #from .envs.vertical_handover_env import VerticalHandoverEnv
    #register(id="SatGym-VerticalHandover-v0", entry_point="satgym.envs:VerticalHandoverEnv", max_episode_steps=1000)
    
    # Môi trường Multi-agent sẽ được xử lý riêng vì nó dùng PettingZoo
    #from .envs.multi_agent_routing_env import MultiAgentRoutingEnv

except ImportError as e:
    logging.warning(f"Could not register all environments due to missing dependency or setup issue: {e}")
