from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

import os
from dotenv import load_dotenv


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


def load_web_documents_firefox(urls, login_url=None):
    """
    Fetch and parse the HTML content from a list of URLs using Firefox.
    """
    documents = {
        "web_result": [],
        "pdf_to_read": []
    }
    options = FirefoxOptions()
    options.headless = True

    with webdriver.Firefox(options=options) as driver:
        if login_url is not None:
            authenticate_firefox(driver, login_url)

            # page post-connexion
            driver.get(
                "https://e-services.uha.fr/_authenticate?requestedURL=/index.html")

        for url in urls:
            try:
                # print(f"Lien ouvert : {url['url']}")
                if url["url"].endswith(".pdf") or "/mod/resource/view.php" in url["url"]:
                    documents["pdf_to_read"].append(url["url"])
                    continue
                driver.get(url["url"])

                if (url["type"] == "moodle"):
                    # récupérer les liens et pdf de la page moodle ici
                    links = extract_moodle_links(driver)

                    for link in links:
                        urls.append({"url": link, "type": "plain"})
                        # on met les liens récupérés dans la liste
                        # pour les récupérer

                html_content = driver.page_source

                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()
                doc = Document(page_content=text, metadata={
                               "source": url["url"]})
                documents["web_result"].append(doc)
            except Exception as e:
                print(f"Error fetching {url['url']}: {e}")

    print("Récupération web terminée")
    return documents
