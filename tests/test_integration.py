# tests/test_integration.py
import sys
from pathlib import Path
import logging
import os

# ======================================================================
# === KHỐI THIẾT LẬP MÔI TRƯỜNG (PHIÊN BẢN ĐÚNG) ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thêm thư mục src/ của SatGym để có thể `import satgym`
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Thêm thư mục GỐC của StarPerf để `from src...` hoạt động
STARPERF_ROOT_PATH = PROJECT_ROOT / "deps" / "StarPerf_Simulator"
if str(STARPERF_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(STARPERF_ROOT_PATH))
# ======================================================================

# Bây giờ import mới an toàn
from satgym.simulators.satellite_simulator import SatelliteSimulator
from satgym import STARPERF_PATH 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simulator_initialization():
    """
    Tests if the SatelliteSimulator can be initialized correctly.
    """
    print("\n" + "="*50)
    print("--- Testing SatelliteSimulator Initialization ---")
    print("="*50 + "\n")

    try:
        # --- 1. Chuẩn bị Config ---
        config = {
            "constellation_name": "Starlink",
            "dT": 578, # ~10 steps
        }
        
        # Xóa file HDF5 cũ để buộc tính toán lại
        h5_path = STARPERF_PATH / "data" / "XML_constellation" / "Starlink.h5"
        if h5_path.exists():
            logger.info(f"Removing existing HDF5 file: {h5_path}")
            os.remove(h5_path)
        
        # --- 2. Khởi tạo Simulator ---
        logger.info("Initializing SatelliteSimulator...")
        sim = SatelliteSimulator(starperf_path=STARPERF_PATH, config=config)
        
        # --- 3. Kiểm tra các thuộc tính cơ bản ---
        assert sim.num_satellites > 0, "Number of satellites should be greater than 0"
        assert sim.simulation_steps > 5, "Simulation steps should be > 5 for this config"
        logger.info(f"✅ Initialization successful. Found {sim.num_satellites} satellites and {sim.simulation_steps} time steps.")

        # --- 4. Kiểm tra các hàm API ---
        logger.info("\n--- Testing API methods ---")
        graph = sim.get_network_graph(time_step=1)
        assert graph.number_of_nodes() == sim.num_satellites
        logger.info(f"✅ get_network_graph() returned a valid graph.")
        
        print("\n" + "="*40)
        print("🎉   SatelliteSimulator Test PASSED!   🎉")
        print("="*40)

    except Exception as e:
        print(f"\n❌   Test FAILED! An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simulator_initialization()
