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
    driver.set_window_size(required_width, height)
    count = 0
    screenshots = []
    scroll_height = 0
    while scroll_height < required_height:
        #photo_path = os.path.join(path, name + f"_{count}.png")
        #driver.save_screenshot(photo_path)  # has scrollbar
        img = driver.get_screenshot_as_png()
        count += 1
        scroll_height += height
        driver.execute_script(f"window.scrollTo(0, {scroll_height})")
        screenshots.append(img)
        time.sleep(.5)

    return screenshots

import json
import pandas as pd
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm


def get_walmart_urls(start=0, end=100):
    base_url = "https://www.walmart.com"
    with open("/home/kevin/bin/scraping_engine/data/products.json", "r") as f:
        products = json.load(f)
    urls = [(p["productId"], base_url + p["productPageUrl"]) for p in products]
    if start and end:
        urls = urls[start:end]
    return urls


def get_amazon_urls(path, start=None, end=None):
    with open(path, "r") as file:
        data = json.load(file)
    urls = []
    while True:
        urls.extend([(i['url'].split('/')[5], i['url']) for i in data["products"]])
        try:
            data = data['selection2'][0]
        except:
            break
    if start and end:
        urls = urls[start: end]
    return urls


def collect_screenshots(urls, save_path):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("window-size=1800,1500")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183")
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(10)

    for _id, u in tqdm(urls):
        logging.info("Getting Product ID: " + _id)
        try:
            driver.get(u)#"https://www.walmart.com/ip/Kellogg-s-Raisin-Bran-Breakfast-Cereal-Original-Family-Size-24-Oz/844236833?wpa_bd=&wpa_pg_seller_id=F55CDC31AB754BB68FE0B39041159D63&wpa_ref_id=wpaqs:oY8Uq2-8EsahZtHhSEzEeVwHB6KFQ-2wPkD5VyawtMxIYPgQAsp6jmUmkwzZFwGZ9YYJoTUwhNTE3zkULQJUAwZ6-I_svxLJlxdcfY2bU_opWocBK2pCFMjzs5mU6gFz3aFZCMVRxXNGYPCn89XFvZXbUWzN_Fo0dZx_hmGeXnbjl0cZ9Vv_6nDckrTxwmaplNjY_3WoYkVJIUxhnI3EXQ&wpa_tag=&wpa_aux_info=&wpa_pos=1&wpa_plmt=1145x1145_T-C-IG_TI_1-2_HL-INGRID-GRID-NY&wpa_aduid=690c098b-8b3c-417e-9f77-433c1632d561&wpa_pg=browse&wpa_pg_id=976759&wpa_st=__searchterms__&wpa_tax=976759&wpa_bucket=__bkt__")
        except TimeoutException as e:
            print("Skipped: ", _id)
            continue

        imgs = custom_screenshots(driver, fullpage=True ) #[{'width': 0,  'height': 0}, {'width': 0, 'height': 2000}]

        for i, img in enumerate(imgs):
            with open(f'{save_path}/{_id}_{i}.png', 'wb') as image_file:
                image_file.write(img)
        time.sleep(.5)
    driver.close()

walmart_path = "/home/kevin/bin/scraping_engine/data/products.json"
path = "/home/kevin/bin/scraping_engine/data/run_results.json"
#urls = get_amazon_urls(path, start=0, end=100)
urls = get_walmart_urls(223, 300)
collect_screenshots(urls, "/home/kevin/bin/scraping_engine/data/walmart_product_images_100_200")