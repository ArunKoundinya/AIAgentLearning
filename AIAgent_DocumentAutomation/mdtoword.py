import spire.doc as sd
from spire.doc.common import *

# Create a Document object
document = sd.Document()

# Load a Markdown file
document.LoadFromFile("example.md")

# Save it as a docx file
document.SaveToFile("ToWord.docx", sd.FileFormat.Docx2016)

# Dispose resources
document.Dispose()