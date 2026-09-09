import os
import urllib.request
import spiceypy as spice

# ==============================================================================
# SPICE KERNEL MANAGEMENT
# ==============================================================================
KERNEL_DIR = "./spice_kernels"
os.makedirs(KERNEL_DIR, exist_ok=True)

KERNEL_URLS = {
    "pck00010.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de440s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
    "Gravity.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/Gravity.tpc"
}

def load_spice_kernels():
    """Download and furnish necessary SPICE kernels if not present."""
    kernels_to_load = []
    for filename, url in KERNEL_URLS.items():
        filepath = os.path.join(KERNEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from NAIF...")
            urllib.request.urlretrieve(url, filepath)
        kernels_to_load.append(filepath)
        spice.furnsh(filepath)
    
    #spice.furnsh(" ".join(kernels_to_load))
    print("SPICE kernels furnished successfully.")
