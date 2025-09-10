import time
from logzero import logger, logfile
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64
# from webdriver_manager.chrome import ChromeDriverManager

logfile("pogodynka.log")

# Create a browser instance running in the background
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
driver = webdriver.Chrome(service=Service(), options=options)
driver.set_window_size(1280, 720)


# Function used to debug browser not working properly
def debug_screenshot(browser, name):
    screenshot = browser.get_screenshot_as_base64()
    img = Image.open(BytesIO(base64.b64decode(screenshot)))
    area = img.crop((0, 0, 1280, 720))
    area.save(name, 'PNG')


def get_page(url):
    # Define headers as a dictionary
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/139.0.0.0 Mobile Safari/537.36 OPR/91.0.0.0',
        'Content-Type': 'application/json'
    }

    # A GET request to the API
    try:
        response = requests.get(url, headers=headers)
    except Exception:
        time.sleep(60)
        response = requests.get(url, headers=headers)

    time.sleep(10)

    return response


def round_to_closest_hour(time_str):
    target_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    time_obj = datetime.strptime(time_str, "%H:%M")
    hour = time_obj.hour + time_obj.minute / 60  # Convert to decimal hours

    # Find the closest hour in the target list
    closest_hour = min(target_hours, key=lambda x: abs(x - hour))
    return closest_hour


def get_data(response):
    try:
        if len(response.text) > 3000:

            # Parse response data
            datagodz = response.text.split("</td><td align=center>dane z ")[1].split("""<a href="javascript:
            window.open('https://www.traxelektronik.pl/pogoda/logging.php'""")[0].split("</td><td width")[0]

            data = datagodz.split(" ")[0]
            godz = datagodz.split(" ")[1]

            temp_odcz = response.text.split("&name=Temperatura,odczuwalna")[0].split("?temp=")[-1].split("&sr=")[0]
            temp_pow = response.text.split("&name=Temperatura,powietrza")[0].split("?temp=")[-1].split("&sr=")[0]
            temp_naw0 = response.text.split("&name=Temperatura,nawierzchni,0cm")[0].split("?temp=")[-1].split("&sr=")[0]
            temp_na5 = response.text.split("&name=Temperatura,nawierzchni,-5cm")[0].split("?temp=")[-1].split("&sr=")[0]
            temp_pod = response.text.split("&name=Temperatura,podbudowy,-30cm")[0].split("?temp=")[-1].split("&sr=")[0]
            temp_pun = response.text.split("&name=Temperatura,punktu,rosy")[0].split("?temp=")[-1].split("&sr=")[0]

            wiatr = response.text.split("<br>&nbsp;&nbsp;")
            wiatr_ms = wiatr[1]
            wiatr_kmh = wiatr[2]
            wiatr_wezly = wiatr[3]
            wiatr_opis = wiatr[4].split("<br><b>Kierunek wiatru:</b>")[0].split("<br>&nbsp;&nbsp")[1].replace("\n", "")

            if "cisza" not in wiatr[4]:
                wiatr_kierunek = wiatr[5].split("&deg;, ")[0]
            else:
                wiatr_opis = wiatr_opis.split("<br></td>")[0]
                wiatr_kierunek = "Błąd..."

            return (f"{data};{godz};{temp_odcz};{temp_pow};{temp_naw0};{temp_na5};{temp_pod};{temp_pun};"
                    f"{wiatr_ms};{wiatr_kmh};{wiatr_wezly};{wiatr_opis};{wiatr_kierunek}")
    except Exception as e:
        # Just in case something fails, the whole script won't fail, it just aborts
        logger.error('{}: {})'.format(e.__class__.__name__, e))

    return


def main():
    first = get_data(get_page("https://www.traxelektronik.pl/pogoda/stacja/stacja.php?w_ekr=1920&stid=1219"))
    filename = " ".join(first.split(';')[0:2]).replace(":", "")
    with open(f"{filename}.csv", "w") as f:
        f.write("Data;Godzina;Temperatura odczuwalna;Temperatura powietrza;Temperatura nawierzchni 0cm;"
                "Temperatura nawierzchni -5cm;Temperatura podbudowy -30cm;Temperatura punktu rosy;Wiatr m/s;"
                "Wiatr km/h;Wiatr węzły;Wiatr opis;Kierunek wiatru;Windy temperatura;Windy prędkość wiatru m/s;"
                "Windy powiewy wiatru m/s\n")

    logger.info("File created, starting writing data")

    while True:
        # Load Windy.com to get wind info
        driver.get(
            "https://embed.windy.com/embed2.html?lat=52.229676&lon=21.012229&zoom=5&level=surface&overlay=wind&menu="
            "&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=true&detailLat="
            "52.229676&detailLon=21.012229&metricWind=default&metricTemp=default&radarRange=-1")

        time.sleep(30)

        hours = "/html/body/span/div/span/div/div[2]/div[2]/div/div[1]/table/tbody/tr[2]"
        temperature = "/html/body/span/div/span/div/div[2]/div[2]/div/div[1]/table/tbody/tr[4]"
        wind_speed = "/html/body/span/div/span/div/div[2]/div[2]/div/div[1]/table/tbody/tr[6]"
        wind_gusts = "/html/body/span/div/span/div/div[2]/div[2]/div/div[1]/table/tbody/tr[7]"

        # Find and parse info about the wind
        hours = driver.find_element(By.XPATH, hours)
        temperature = driver.find_element(By.XPATH, temperature)
        wind_speed = driver.find_element(By.XPATH, wind_speed)
        wind_gusts = driver.find_element(By.XPATH, wind_gusts)

        now_index = hours.text.split(' ').index(str(round_to_closest_hour(datetime.now().strftime("%H:%M"))))

        now_temp = temperature.text.split(' ')[now_index].replace("°", "")
        air_speed = float(wind_speed.text.split(' ')[now_index])*0.51445
        gust_speed = float(wind_gusts.text.split(' ')[now_index])*0.51445
        # Load info from Lazurowa station and parse it
        current_data = get_data(get_page('https://www.traxelektronik.pl/pogoda/stacja/stacja.php?w_ekr=1920&stid=1219'))

        if current_data is not None:
            with open(f"{filename}.csv", "a") as f:
                f.write(f"{current_data};{now_temp};{air_speed};{gust_speed}\n")
            logger.info("Data written")

            # Load a random website to free up resources when waiting for next loop
            driver.get("https://one.one.one.one")

        time.sleep(10 * 60)


if __name__ == "__main__":
    main()
