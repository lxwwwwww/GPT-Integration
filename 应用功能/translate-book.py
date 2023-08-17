import pdfplumber
import PyPDF2
import openai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

openai.api_key = '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'

pdf = pdfplumber.open('.pdf')

pdf_writer = PyPDF2.PdfWriter()

for page_num, page in enumerate(pdf.pages):
    text = page.extract_text()
    lines = text.splitlines()
    translated_text = ''
    for line in lines:
        print(line)
        messages = []
        while len(line) > 0:
            messages.append({
                'role': 'user',
                'content': 'Help me translate the text to English, only return the translated text: '+ line[:250]
            })
            line = line[250:]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages
            )
            translated_text += response.choices[0].message.content
            print(response.choices[0].message.content)

    # Create a new PDF page with the translated text
    c = canvas.Canvas("temp_page.pdf", pagesize=letter)
    c.setFont("Helvetica", 12)
    text_object = c.beginText(40, 750)
    for line in translated_text.split('\n'):
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()

    # Add the new PDF page to the PDF writer
    with open("temp_page.pdf", "rb") as f_temp:
        temp_reader = PyPDF2.PdfReader(f_temp)
        pdf_writer.add_page(temp_reader.pages[0])

# Save the translated PDF
with open("translated.pdf", "wb") as f:
    pdf_writer.write(f)

pdf.close()