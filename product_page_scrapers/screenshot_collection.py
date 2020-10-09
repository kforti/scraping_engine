import time
import os
import logging

from selenium import webdriver
from selenium.common.exceptions import TimeoutException

def custom_screenshots(driver, shot_dimensions=None, fullpage=False):
    if not shot_dimensions and fullpage is False:
        raise AttributeError
    if shot_dimensions:
        images = []
        for dim in shot_dimensions:
            img = get_dimension_screenshot(driver, dim['width'], dim['height'])
            time.sleep(1)
            images.append(img)

    elif fullpage:
        images = fullpage_screenshot(driver)

    return images


def get_dimension_screenshot(driver, width, height):
    #driver.set_window_size(width, height)
    driver.execute_script(f"window.scrollTo({width}, {height})")
    img = driver.get_screenshot_as_png()
    return img


def fullpage_screenshot(driver: webdriver.Chrome) -> None:

    original_size = driver.get_window_size()
    required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
    required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
    height = original_size['height']
    driver.set_window_size(required_width)
    count = 0
    screenshots = []
    while height < required_height:
        #photo_path = os.path.join(path, name + f"_{count}.png")
        #driver.save_screenshot(photo_path)  # has scrollbar
        img = driver.get_screenshot_as_png()
        count += 1
        height += height
        driver.execute_script(f"window.scrollTo(0, {height})")
        screenshots.append(img)
        time.sleep(.5)

    return screenshots

import json
import pandas as pd
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

SAVE_PATH = "/data/product_images"
#urls = pd.read_csv("/home/kevin/bin/scraping_engine/product_urls.csv")["URL"].tolist()

base_url = "https://www.walmart.com"
with open("/data/products.json", "r") as f:
    products = json.load(f)
urls = [(p["productId"], base_url + p["productPageUrl"]) for p in products]


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("window-size=1800,1500")
chrome_options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183")
driver = webdriver.Chrome(options=chrome_options)
driver.set_page_load_timeout(10)

for _id, u in tqdm(urls[20:100]):
    logging.info("Getting Product ID: " + _id)
    try:
        driver.get(u)#"https://www.walmart.com/ip/Kellogg-s-Raisin-Bran-Breakfast-Cereal-Original-Family-Size-24-Oz/844236833?wpa_bd=&wpa_pg_seller_id=F55CDC31AB754BB68FE0B39041159D63&wpa_ref_id=wpaqs:oY8Uq2-8EsahZtHhSEzEeVwHB6KFQ-2wPkD5VyawtMxIYPgQAsp6jmUmkwzZFwGZ9YYJoTUwhNTE3zkULQJUAwZ6-I_svxLJlxdcfY2bU_opWocBK2pCFMjzs5mU6gFz3aFZCMVRxXNGYPCn89XFvZXbUWzN_Fo0dZx_hmGeXnbjl0cZ9Vv_6nDckrTxwmaplNjY_3WoYkVJIUxhnI3EXQ&wpa_tag=&wpa_aux_info=&wpa_pos=1&wpa_plmt=1145x1145_T-C-IG_TI_1-2_HL-INGRID-GRID-NY&wpa_aduid=690c098b-8b3c-417e-9f77-433c1632d561&wpa_pg=browse&wpa_pg_id=976759&wpa_st=__searchterms__&wpa_tax=976759&wpa_bucket=__bkt__")
    except TimeoutException as e:
        print("Skipped: ", _id)
        continue

    imgs = custom_screenshots(driver, [{'width': 0,  'height': 0}, {'width': 0, 'height': 2000}])

    for i, img in enumerate(imgs):
        with open(f'{SAVE_PATH}/{_id}_{i}.png', 'wb') as image_file:
            image_file.write(img)
    time.sleep(.5)
driver.close()