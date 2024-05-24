from PyPDF2 import PdfReader
from spire.pdf import *
from spire.pdf.common import *

# pip install Spire.pdf


def extract_metadata(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        metadata = reader.metadata
        print(str(metadata)+"\n")


def modify_metadata_Peter_Pan(pdf_path):
    doc = PdfDocument()
    doc.LoadFromFile(pdf_path)

    information = doc.DocumentInformation

    information.Author = "J. M. Barrie"
    information.Keywords = "Peter-Pan, book, story, children, adventure, tale"
    information.Subject = "Story of Peter-Pan"
    information.Title = "Peter-Pan"
    # information.SetCustomProperty("Telephone", "845-3-111")

    doc.SaveToFile(pdf_path)
    doc.Close()

    extract_metadata(pdf_path)


def modify_metadata_Walk_Egyptian(pdf_path):
    doc = PdfDocument()
    doc.LoadFromFile(pdf_path)

    information = doc.DocumentInformation

    information.Author = "Caroline Tully"
    information.Keywords = "Egypt, occult, Aleister Crowley, Book of The Law, Article, Florence Farr, Rose Kelly, Samuel Mathers"
    information.Subject = "article investigates the story of Aleister Crowley’s reception of  The  Book  of  the  Law  in \
    Cairo, Egypt, in 1904, focusing on the question of why it occurred in Egypt"
    information.Title = "Walk like an Egyptian:\
    Egypt as authority in Aleister Crowley’s reception of The Book of the Law "
    # information.SetCustomProperty("Telephone", "845-3-111")

    doc.SaveToFile(pdf_path)
    doc.Close()

    extract_metadata(pdf_path)


"""
modify_metadata_Peter_Pan("differents_textes/peter-pan.pdf")
extract_metadata("differents_textes/Dumas_Les_trois_mousquetaires_1.pdf")
modify_metadata_Walk_Egyptian(
    "differents_textes/Tully Walk like an Egyptian .pdf")
"""
