# tests/test_routing_env.py
import sys
from pathlib import Path

# ======================================================================
# === KHỐI THIẾT LẬP MÔI TRƯỜNG ("Boilerplate" cho mọi script) ===
# Phải được thực thi TRƯỚC TẤT CẢ các import khác từ satgym

# 1. Thêm thư mục src/ của dự án vào path để 'import satgym' hoạt động
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 2. Thêm thư mục GỐC của StarPerf vào path để 'from src...' hoạt động
STARPERF_ROOT_PATH = PROJECT_ROOT / "deps" / "StarPerf_Simulator"
if str(STARPERF_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(STARPERF_ROOT_PATH))
# ======================================================================

# Bây giờ import mới an toàn
import gymnasium as gym
import satgym

print("\n--- Testing RoutingEnv with Entry Point Setup ---")
try:
    # Xóa file HDF5 cũ để buộc tính toán lại
    import os
    h5_path = satgym.STARPERF_PATH / "data" / "XML_constellation" / "Starlink.h5"
    if h5_path.exists():
        os.remove(h5_path)
        
    env = gym.make("SatGym-Routing-v0", simulation_steps=10)
    print("✅ Environment created successfully!")
    
    obs, info = env.reset()
    print("✅ Environment reset successful.")
    
    env.step(env.action_space.sample())
    print("✅ Environment step successful.")
    
    print("\n🎉   Test PASSED!   🎉")

except Exception as e:
    print(f"\n❌   Test FAILED! An error occurred: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    pass # Toàn bộ logic đã ở trong scope chính
