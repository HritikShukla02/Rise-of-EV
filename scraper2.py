from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import os

# Setup download folder and new filename
# download_dir = "D:\Vahan parivahan\Rise-of-EV\data\Vehicle_data"
download_dir = "D:\Vahan parivahan\Rise-of-EV\data\State_ev_data"
expected_filename = "reportTable.xlsx"
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}

chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_experimental_option('detach', True)


def click_element(path):
    for attempt in range(10):
        try:
            # download_button = driver.find_element(By.ID, "downloadBtn")
            # download_button.click()
            # Setting X- variable
            element = wait.until(EC.element_to_be_clickable((By.XPATH,path)))

            # print(x_div.text)
            time.sleep(1)
            element.click()
            break
        except StaleElementReferenceException:
            print(f"Stale element — retrying... (attempt {attempt + 1})")
            time.sleep(1)

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

# Wait and locate button fresh before clicking

for i in range(2016, 2026):
    new_filename = f"{i}.xlsx"

    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://vahan.parivahan.gov.in/vahan4dashboard/vahan/view/reportview.xhtml')
    wait = WebDriverWait(driver, 10)

    # Setting Y- variable
    click_element('//*[@id="yaxisVar"]/div[3]/span')
    click_element('//*[@id="yaxisVar_5"]')


    # Setting X- variable
    click_element('//*[@id="xaxisVar"]/div[3]/span')
    click_element('//*[@id="xaxisVar_7"]')

    
    # Setting year
    click_element('//*[@id="selectedYear"]/div[3]/span')
    year = driver.find_element(By.CSS_SELECTOR, value=f'#selectedYear_items li[data-label="{i}"]')
    year.click()

    # Hit refresh
    click_element('//*[@id="j_idt72"]/span[2]')

    #  Click Filter
    click_element('//*[@id="filterLayout-toggler"]/span/a/span')

    # Select 4WIC
    click_element('//*[@id="VhCatg"]/tbody/tr[7]/td/label')

    # Select HMV
    click_element('//*[@id="VhCatg"]/tbody/tr[9]/td/label')

    # Select LMV
    click_element('//*[@id="VhCatg"]/tbody/tr[12]/td/label')

    # Select MMV
    click_element('//*[@id="VhCatg"]/tbody/tr[15]/td/label')

    #------------------------- Set Fuel Type -------------------------#
    click_element('//*[@id="fuel"]/tbody/tr[8]/td/label') # Electriv(BOV)
    click_element('//*[@id="fuel"]/tbody/tr[22]/td/label') # Pure EV
    # click_element('//*[@id="fuel"]/tbody/tr[10]/td/label') #Fuel cell Hydrogen
    click_element('//*[@id="fuel"]/tbody/tr[21]/td/label') # Plug-in Hybrid
    click_element('//*[@id="fuel"]/tbody/tr[24]/td/label') # strong Hybrid


    # Hit refresh
    click_element('//*[@id="j_idt79"]/span[2]')

    # hit Download
    click_element('//*[@id="groupingTable:xls"]')

    # Wait and rename
    if wait_for_download(expected_filename):
        os.rename(
            os.path.join(download_dir, expected_filename),
            os.path.join(download_dir, new_filename)
        )
        print(f"{i} File renamed successfully.")
    else:
        print(f"{i} Download failed or timed out.")

    driver.quit()