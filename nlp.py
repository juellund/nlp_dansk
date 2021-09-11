import PyPDF2
import os

def read_pdfs(sti, tekster):
  files = os.listdir(sti)
  alle_tekster = []
  for file in files:
    if file in tekster:
      pdf = open(sti + '/' + file, 'rb')
      read_pdf = PyPDF2.PdfFileReader(pdf)
      tekst = []
      for page in range(read_pdf.getNumPages()):
        tekst.append(read_pdf.getPage(page).extractText())
      alle_tekster.extend(tekst)
  tekst_raw = ''
  for side in alle_tekster:
    side_raw = side.split('\n')
    tekst_raw += ' '.join(side_raw)
  return tekst_raw