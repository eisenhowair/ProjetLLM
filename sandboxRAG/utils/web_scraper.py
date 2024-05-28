from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

import os
import time
from dotenv import load_dotenv
from selenium.webdriver.common.keys import Keys


def authenticate_firefox(driver, login_url):
    """
    Authenticate with the website using Firefox.
    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env_path = os.path.join(parent_dir, '.env')
    load_dotenv(dotenv_path=env_path)

    username = os.getenv("USERNAME_CAS")
    password = os.getenv("PASSWORD_CAS")

    driver.get(login_url)
    username_input = driver.find_element(
        By.ID, "username")
    password_input = driver.find_element(
        By.ID, "password")
    submit_button = driver.find_element(
        By.NAME, "submit")

    username_input.send_keys(username)
    password_input.send_keys(password)
    submit_button.click()


def extract_moodle_links(driver):
    """
    Extracts the links with the "aalink stretched-link" class from a Moodle page.
    """
    links = []
    pdf_files = []

    for link in driver.find_elements(By.CSS_SELECTOR, "a.aalink.stretched-link"):
        href = link.get_attribute("href")
        if href.endswith(".pdf"):
            print(f"{href} a été ajouté")
        else:
            links.append(href)

    return links


def recupere_liste_cours(driver):
    liste_cours = []
    wait = WebDriverWait(driver, 6)  # Attendre jusqu'à 10 secondes
    elements = wait.until(EC.presence_of_all_elements_located(
        (By.CSS_SELECTOR, "a.aalink.coursename.mr-2.mb-1")))
    for link in elements:
        href = link.get_attribute("href")
        print(f"{href} ajouté")
        liste_cours.append(href)

    print(liste_cours)
    return liste_cours


def prepare_options():
    options = FirefoxOptions()
    options.headless = True

    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.manager.focusWhenStarting", False)
    options.set_preference("browser.download.alwaysOpenPanel", False)
    options.set_preference("browser.download.animateNotifications", False)
    options.set_preference("browser.download.panel.shown", False)
    options.set_preference("browser.download.manager.useWindow", False)
    options.set_preference("dom.webnotifications.enabled", False)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk",
                           "application/msword, application/csv, application/ris, text/csv, image/png, application/pdf, text/html, text/plain, application/zip, application/x-zip, application/x-zip-compressed, application/download, application/octet-stream")
    options.set_preference("browser.download.manager.alertOnEXEOpen", False)
    options.set_preference("browser.download.useDownloadDir", True)
    options.set_preference("browser.helperApps.alwaysAsk.force", False)
    options.set_preference("browser.download.manager.closeWhenDone", True)
    options.set_preference("browser.download.animateNotifications", False)

    options.set_preference(
        "browser.download.manager.showAlertOnComplete", False)
    options.set_preference(
        "services.sync.prefs.sync.browser.download.manager.showWhenStarting", False)
    options.set_preference(
        "browser.download.dir", "/home/UHA/e2303253/ProjetLLM/sandboxRAG/differents_textes/moodle")
    # Example:options.set_preference("browser.download.dir", "C:\Tutorial\down")
    mime_types = [
        'text/plain',
        'application/vnd.ms-excel',
        'text/csv',
        'application/csv',
        'text/comma-separated-values',
        'application/download',
        'application/octet-stream',
        'binary/octet-stream',
        'application/binary',
        'application/x-unknown',
        'application/octet-stream',
        'application/pdf',
        'application/x-pdf'
    ]
    options.set_preference(
        "browser.helperApps.neverAsk.saveToDisk", ",".join(mime_types))
    # Disable built-in PDF viewer
    options.set_preference("pdfjs.disabled", True)

    return options


def load_web_documents_firefox(urls, login_url=None):
    """
    Fetch and parse the HTML content from a list of URLs using Firefox.
    """
    documents = {
        "web_result": [],
        "pdf_to_read": []
    }

    with webdriver.Firefox(options=prepare_options()) as driver:
        if login_url is not None:
            authenticate_firefox(driver, login_url)

            # page post-connexion
            driver.get(
                "https://e-services.uha.fr/_authenticate?requestedURL=/index.html")
            driver.execute_script(
                "window.addEventListener('load', function() { Notification.requestPermission(permission => { if (permission === 'granted') { Notification.permission = 'default'; } }); }, false);")

        for url in urls:
            try:
                # print(f"Lien ouvert : {url['url']}")
                if url["url"].endswith(".pdf") or "/mod/resource/view.php" in url["url"]:
                    documents["pdf_to_read"].append(url["url"])
                    continue
                if (url["type"] == "webpage_from_moodle"):
                    # pour les liens menant vers du web depuis moodle
                    driver.get(url["url"]+"&redirect=1")
                else:
                    driver.get(url["url"])  # pour le reste

                if (url["type"] == "accueil_moodle"):
                    print("accueil_moodle fonctionne bien")
                    liste_cours = recupere_liste_cours(driver)
                    for cours in liste_cours:
                        urls.append(
                            {"url": cours, "type": "cours_moodle"})

                if (url["type"] == "cours_moodle"):
                    # récupérer les liens et pdf de la page moodle ici
                    links = extract_moodle_links(driver)

                    for link in links:
                        urls.append(
                            {"url": link, "type": "webpage_from_moodle"})
                        # on met les liens récupérés dans la liste
                        # pour les récupérer
                time.sleep(1)
                html_content = driver.page_source

                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()
                doc = Document(page_content=text, metadata={
                               "source": url["url"]})
                documents["web_result"].append(doc)
            except Exception as e:
                print(f"Error fetching {url['url']}: {e}")

        for pdf_url in documents["pdf_to_read"]:
            driver.get(pdf_url)

            time.sleep(1)
            # driver.switch_to.window(driver.window_handles[0])

            driver.execute_script(
                "window.setTimeout(function(){ window.location.reload(); }, 1000);")

    print("Récupération web terminée")
    return documents
