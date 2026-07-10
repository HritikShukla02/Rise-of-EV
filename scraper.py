from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os

# Setup download folder and new filename
download_dir = "/path/to/your/download/folder"
expected_filename = "original_name_from_server.extension"
new_filename = "renamed_file.extension"

# Chrome preferences
chrome_options = Options()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)

# Create driver
driver = webdriver.Chrome(options=chrome_options)


driver.get("https://example.com/download-page")

# Click the download button
download_button = driver.find_element(By.ID, "downloadBtn")
download_button.click()

# Wait until download completes
def wait_for_download(filename, timeout=30):
    download_path = os.path.join(download_dir, filename)
    for _ in range(timeout):
        if os.path.exists(download_path):
            # Wait a bit more in case it's still writing
            time.sleep(1)
            return True
        time.sleep(1)
    return False

# Wait and rename
if wait_for_download(expected_filename):
    os.rename(
        os.path.join(download_dir, expected_filename),
        os.path.join(download_dir, new_filename)
    )
    print("File renamed successfully.")
else:
    print("Download failed or timed out.")
