import os
import platform

# Define the base directory based on the hostname
hostname = platform.node()
if hostname == 'Nates-MacBook-Pro.local' or hostname == 'UM-C02RQ1S8FVH7':
    BASE_DIR = '/Users/nate/Documents/SFWMD_work/'
elif hostname == 'mango.rsmas.miami.edu':
    BASE_DIR = '/home/ntaminger/SFWMD_work/'
elif hostname == 'aztidal01p':
    BASE_DIR = '/home/ntaminge/SFWMD_work/'
else:
    raise ValueError(f"Unknown hostname: {hostname}")

pwd = os.path.join(BASE_DIR, 'predictions/')
os.chdir(pwd)

def delete_grib_files():
    for file in os.listdir():
        if file.endswith(".grib2"):
            try:
                os.remove(file)
                print(f"Deleted {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")
                
#------------------------------------------------------------------------------
#                 DELETE GRIB
#------------------------------------------------------------------------------
delete_grib_files()
