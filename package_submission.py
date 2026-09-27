import os
import sys
import zipfile

ROOT_DIR = r"c:\jaishree_projects\amazon_ml_challenge"
ZIP_NAME = "Bugs2_submission.zip"
ZIP_PATH = os.path.join(ROOT_DIR, ZIP_NAME)

print("=================================================================", flush=True)
print("PACKAGING FINAL SUBMISSION ZIP ARTIFACT", flush=True)
print("=================================================================", flush=True)

# Required files check
matching_tsv = os.path.join(ROOT_DIR, "output", "matching_results.tsv")
candidate_tsv = os.path.join(ROOT_DIR, "output", "candidate_pairs.tsv")
doc_template = os.path.join(ROOT_DIR, "Documentation_template.md")
readme = os.path.join(ROOT_DIR, "README.md")
reqs = os.path.join(ROOT_DIR, "requirements.txt")
exec_model = os.path.join(ROOT_DIR, "execute_model.py")
src_dir = os.path.join(ROOT_DIR, "src")

assert os.path.isfile(matching_tsv), f"Missing: {matching_tsv}"
assert os.path.isfile(candidate_tsv), f"Missing: {candidate_tsv}"
assert os.path.isfile(doc_template), f"Missing: {doc_template}"
assert os.path.isfile(readme), f"Missing: {readme}"
assert os.path.isfile(reqs), f"Missing: {reqs}"
assert os.path.isdir(src_dir), f"Missing: {src_dir}"

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # 1. output/
    zipf.write(matching_tsv, arcname="output/matching_results.tsv")
    zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")
    print("Added output/ files to zip.")
    
    # 2. code/business_entity_resolution/
    zipf.write(readme, arcname="code/business_entity_resolution/README.md")
    zipf.write(reqs, arcname="code/business_entity_resolution/requirements.txt")
    if os.path.isfile(exec_model):
        zipf.write(exec_model, arcname="code/business_entity_resolution/execute_model.py")
        
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            fpath = os.path.join(src_dir, fname)
            zipf.write(fpath, arcname=f"code/business_entity_resolution/src/{fname}")
    print("Added code/business_entity_resolution/ files to zip.")
    
    # 3. Documentation_template.md
    zipf.write(doc_template, arcname="Documentation_template.md")
    print("Added Documentation_template.md to zip.")

# Verify contents
print("\n--- Verifying Zip Archive Contents ---")
has_dataset = False
total_files = 0
with zipfile.ZipFile(ZIP_PATH, 'r') as zipf:
    for info in zipf.infolist():
        total_files += 1
        print(f"  {info.filename} ({info.file_size:,} bytes)")
        if "dataset" in info.filename.lower() or "train_" in info.filename.lower() or "test_" in info.filename.lower():
            has_dataset = True

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"\nTotal Files in ZIP: {total_files}")
print(f"ZIP Archive Size  : {zip_size_mb:.2f} MB")
print(f"Dataset Excluded  : {'PASS (No dataset files included)' if not has_dataset else 'FAIL (Dataset detected!)'}")
print(f"ZIP Path          : {ZIP_PATH}")
