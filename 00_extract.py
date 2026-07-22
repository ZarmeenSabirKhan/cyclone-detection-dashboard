import zipfile
import os
import shutil

# Sahi path — yahan zip files hain
ZIP_DIR = r"C:\Users\Tmart\Desktop\cy\Future"
OUT_DIR = r"C:\Users\Tmart\Desktop\cy\extracted"
os.makedirs(OUT_DIR, exist_ok=True)

short_names = {
    'tasmax': 'tasmax_future.nc',
    'pr': 'pr_future.nc',
    'hurs': 'hurs_future.nc',
    'rsds': 'rsds_future.nc',
    'sfcWind': 'sfcWind_future.nc'
}

zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
print(f"📦 Found {len(zip_files)} zip files in: {ZIP_DIR}\n")

print("🔓 Extracting...")
for zip_file in zip_files:
    var = zip_file.split('_')[0]
    if var not in short_names:
        continue
    
    zip_path = os.path.join(ZIP_DIR, zip_file)
    out_path = os.path.join(OUT_DIR, short_names[var])
    
    zip_size_mb = os.path.getsize(zip_path) / 1024**2
    print(f"  📦 {var}: extracting ({zip_size_mb:.0f} MB zip)...", end=" ", flush=True)
    
    try:
        with zipfile.ZipFile(zip_path) as z:
            nc_members = [m for m in z.namelist() if m.endswith('.nc')]
            if not nc_members:
                print("❌ No .nc inside!")
                continue
            
            with z.open(nc_members[0]) as src, open(out_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        
        size_mb = os.path.getsize(out_path) / 1024**2
        print(f"✅ {short_names[var]} ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n📁 Extracted files in: {OUT_DIR}")
for f in os.listdir(OUT_DIR):
    size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024**2
    print(f"   - {f}  ({size_mb:.0f} MB)")